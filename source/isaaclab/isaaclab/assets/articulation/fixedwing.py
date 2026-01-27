# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.string as string_utils
from isaaclab.actuators import Thruster
from isaaclab.assets.articulation.fixedwing_data import FixedWingData
from isaaclab.utils.types import ArticulationThrustActions
from isaaclab.utils.math import quat_apply_inverse, quat_apply

from .articulation import Articulation

if TYPE_CHECKING:
    from .fixedwing_cfg import FixedWingCfg


class FixedWing(Articulation):
    """A fixed-wing articulation asset class.

    This class extends the base articulation class to support multirotor vehicles
    with thruster actuators that can be applied at specific body locations.
    """

    cfg: FixedWingCfg
    """Configuration instance for the multirotor."""

    def __init__(self, cfg: FixedWingCfg):
        """Initialize the multirotor articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._dt = 1.0 / 120.0  # default to 60 Hz

        self._device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    @property
    def data(self) -> FixedWingData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    def _process_aero_cfg(self) -> None:
        for link_name, wing_cfg in self.cfg.wings.items():
            body_idx, link_names = self.find_bodies(link_name)
            if len(body_idx) == 0:
                raise ValueError(
                    f"Could not find body '{link_name}' for fixed-wing aerodynamics."
                )
            self._aerodata.aero_link_mapping[link_names[0]] = body_idx[0]
            self._aerodata.wing_drag_tensor[link_name] = torch.tensor(
                [wing_cfg.C_rdr, wing_cfg.C_rdp, 0.0], device=self._device
            )

            if wing_cfg.has_controlsurface:
                actuator_name = wing_cfg.connected_actuator
                actuator_idx = self._data.joint_names.index(actuator_name)
                if actuator_idx is None:
                    raise ValueError(
                        f"Could not find actuator joint '{actuator_name}' for fixed-wing aerodynamics."
                    )
                self._aerodata.aero_actuator_link_mapping[link_name] = actuator_idx

            if wing_cfg.stallable:
                self._aerodata.aero_stall_hyst[link_name] = torch.zeros(
                    self.num_instances, device=self._device
                )

        for link_name, engine_cfg in self.cfg.engines.items():
            body_idx, link_names = self.find_bodies(link_name)
            if len(body_idx) == 0:
                raise ValueError(
                    f"Could not find body '{link_name}' for fixed-wing engines."
                )
            self._aerodata.engine_link_mapping[link_names[0]] = body_idx[0]
            actuator_name = engine_cfg.connected_actuator
            actuator_idx = self._data.joint_names.index(actuator_name)
            if actuator_idx is None:
                raise ValueError(
                    f"Could not find actuator joint '{actuator_name}' for fixed-wing aerodynamics."
                )
            self._aerodata.engine_actuator_idx_mapping[link_name] = actuator_idx

    def update(self, dt: float):
        self._dt = dt
        self._data.update(dt)

    def _initialize_impl(self):
        """Initialize the multirotor implementation."""
        # call parent initialization
        # self._data = FixedWingData(self.root_physx_view, self.device)

        super()._initialize_impl()

        # container for data access
        self._aerodata = FixedWingData()

        # Create thruster buffers with correct size (SINGLE PHASE)
        self._process_aero_cfg()

    def write_data_to_sim(self):
        # self._apply_drag()
        self._apply_aerodynamics()
        self._apply_thrust()
        super().write_data_to_sim()

    def _apply_aerodynamics(self):
        # Get world-frame velocities for base link (body index 0)

        forces = torch.zeros_like(self._data.body_lin_vel_w)
        torques = torch.zeros_like(forces)
        positions = torch.zeros_like(forces)
        ones = torch.ones(forces.shape[0], device=self.device)
        unit_z = torch.zeros_like(positions[:, 0, :])
        unit_z[:, 2] = 1.0

        for link_name, wing_cfg in self.cfg.wings.items():
            body_idx = self._aerodata.aero_link_mapping[link_name]
            v_world = self._data.body_lin_vel_w[:, body_idx, :]
            w_world = self._data.body_ang_vel_w[:, body_idx, :]
            quat_w = self._data.body_quat_w[:, body_idx, :]

            if wing_cfg.mixed_airflow:
                influenced_by_body_idx = self._aerodata.engine_link_mapping[
                    wing_cfg.influenced_by
                ]
                influenced_by_actuator_idx = self._aerodata.engine_actuator_idx_mapping[
                    wing_cfg.influenced_by
                ]
                v_world += quat_apply(
                    self._data.body_quat_w[:, influenced_by_body_idx, :],
                    unit_z
                    * (
                        torch.abs(self._data.joint_vel[:, influenced_by_actuator_idx])
                        * wing_cfg.mixed_airflow_coefficient
                    ).unsqueeze(-1),
                )
                # print(v_world)

            v = quat_apply_inverse(quat_w, v_world)
            w = quat_apply_inverse(quat_w, w_world)

            v_projected = -v.clone()
            v_projected[:, 1] = 0.0  # project onto x-z plane
            v_projected_flipped = v_projected.clone()
            v_projected_flipped[:, 0] = -v_projected[:, 2]
            v_projected_flipped[:, 2] = v_projected[:, 0]

            aoa = torch.atan2(v[:, 2], v[:, 0])  # angle of attack

            if wing_cfg.has_controlsurface:
                delta_q = self._data.joint_pos[
                    :, self._aerodata.aero_actuator_link_mapping[link_name]
                ]
            else:
                delta_q = 0.0

            drag_coeff = (
                (
                    wing_cfg.C_ds
                    + wing_cfg.C_d * torch.sin(aoa) ** 2
                    + (delta_q * wing_cfg.q_drag) * torch.cos(aoa) ** 2
                )
                * self.cfg.rho
                * wing_cfg.wing_area_projected
                / 2
            )
            drag = drag_coeff.unsqueeze(-1) * v_projected * torch.abs(v_projected)

            if wing_cfg.stallable:
                blend_factor = (
                    torch.clamp(
                        (torch.abs(aoa) - wing_cfg.stall_angle) / wing_cfg.stall_range,
                        0,
                        1,
                    )
                    ** 0.3
                )
                blend_factor = blend_factor**3 * (
                    blend_factor * (blend_factor * 6 - 15) + 10
                )
            else:
                blend_factor = ones

            lift_blended = (1 - blend_factor) * wing_cfg.C_ll * torch.sin(
                1.9 * aoa
            ) + blend_factor * wing_cfg.C_lt * torch.sin(1.9 * aoa)

            lift_q_blended = (1 - blend_factor) * delta_q * wing_cfg.q_lift

            lift_coeff = (
                (lift_blended + lift_q_blended)
                * self.cfg.rho
                * wing_cfg.wing_area_projected
                / 2
            )
            lift = (
                lift_coeff.unsqueeze(-1)
                * v_projected_flipped
                * torch.abs(v_projected_flipped)
            )

            moment_q_blended = (
                (1 - blend_factor * 0.8)
                * delta_q
                * wing_cfg.q_torque
                * torch.cos(2 * aoa)
            )

            moment_coeff = (
                (wing_cfg.C_m * torch.sin(2 * aoa) + moment_q_blended)
                * self.cfg.rho
                * wing_cfg.wing_area_projected
                / 2
            )

            angl_drag = -torch.mul(
                w * torch.abs(w), self._aerodata.wing_drag_tensor[link_name]
            )

            torque = moment_coeff.unsqueeze(-1) * torch.norm(
                v_projected**2, dim=-1, keepdim=True
            )

            forces[:, body_idx, :] = drag + lift
            torques[:, body_idx, :] = angl_drag
            torques[:, body_idx, 1] = -torque[:, 0]

        self._instantaneous_wrench_composer.add_forces_and_torques(
            self._ALL_INDICES_WP,
            self._ALL_BODY_INDICES_WP,
            forces=wp.from_torch(forces, dtype=wp.vec3f),
            torques=wp.from_torch(torques, dtype=wp.vec3f),
            positions=wp.from_torch(positions, dtype=wp.vec3f),
            is_global=False,
        )

    def _apply_thrust(self):
        # Get world-frame velocities for base link (body index 0)

        forces = torch.zeros_like(self._data.body_lin_vel_w)
        torques = torch.zeros_like(forces)
        positions = torch.zeros_like(forces)
        unit_z = torch.zeros_like(positions[:, 0, :])
        unit_z[:, 2] = 1.0

        for link_name, engine_cfg in self.cfg.engines.items():
            body_idx = self._aerodata.engine_link_mapping[link_name]
            quat_w = self._data.body_quat_w[:, body_idx, :]
            v_world = self._data.body_lin_vel_w[:, body_idx, :]
            v = quat_apply_inverse(quat_w, v_world)

            rpm = (
                self._data.joint_vel[
                    :, self._aerodata.engine_actuator_idx_mapping[link_name]
                ]
                / engine_cfg.max_rpm
            )

            forces[:, body_idx, 2] = torch.clamp(
                rpm * engine_cfg.thrust_coefficient
                - torch.abs(v[:, 2]) * v[:, 2] * engine_cfg.effectiveness,
                0,
                engine_cfg.max_thrust,
            )
            torques[:, body_idx, :] += (
                -rpm.unsqueeze(-1)
                * engine_cfg.torque_coefficient
                * unit_z
                * engine_cfg.spin_direction
            )

        self._instantaneous_wrench_composer.add_forces_and_torques(
            self._ALL_INDICES_WP,
            self._ALL_BODY_INDICES_WP,
            forces=wp.from_torch(forces, dtype=wp.vec3f),
            torques=wp.from_torch(torques, dtype=wp.vec3f),
            positions=wp.from_torch(positions, dtype=wp.vec3f),
            is_global=False,
        )
