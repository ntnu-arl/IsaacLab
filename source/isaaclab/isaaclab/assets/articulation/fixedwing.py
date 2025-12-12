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

import omni.log
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

        self.aero_link_mapping: dict[str, int] = {}

    def _process_aero_cfg(self) -> None:
        for link_name in self.cfg.aero_link_names:
            body_indice, link_names = self.find_bodies(link_name)

            self.aero_link_mapping[link_names[0]] = body_indice[0]

    def _initialize_impl(self):
        """Initialize the multirotor implementation."""
        # call parent initialization
        # self._data = FixedWingData(self.root_physx_view, self.device)

        super()._initialize_impl()

        # Replace data container with MultirotorData

        # Create thruster buffers with correct size (SINGLE PHASE)
        self._process_aero_cfg()

    def write_data_to_sim(self):
        # self._apply_drag()
        self._apply_aerodynamics()
        super().write_data_to_sim()

    def _apply_aerodynamics(self):
        # Get world-frame velocities for base link (body index 0)

        forces = torch.zeros_like(self.data.body_lin_vel_w)
        torques = torch.zeros_like(forces)
        quat_w = self.data.body_quat_w[:, 0, :]

        for _, body_idx in self.aero_link_mapping.items():
            v_world = self.data.body_lin_vel_w[:, body_idx, :]
            v = quat_apply_inverse(quat_w, v_world)
            v_projected = -v.clone()
            v_projected[:, 1] = 0.0  # project onto x-z plane

            aoa = torch.atan2(v[:, 2], v[:, 0])  # angle of attack
            sideslip = torch.atan2(v[:, 1], v[:, 0])  # sideslip angle
            drag = (
                self.cfg.C_d
                * torch.abs(torch.sin(aoa))
                * v_projected
                * torch.abs(v_projected)
                * self.cfg.rho
                * self.cfg.wing_area_projected
                / 2
            )

            v_projected_flipped = v_projected.clone()
            v_projected_flipped[:, 0] = v_projected_flipped[:, 2]
            v_projected_flipped[:, 2] = v_projected_flipped[:, 0]

            lift_turbulent = (
                self.cfg.C_lt
                * torch.sin(2 * aoa)
                * v_projected_flipped
                * torch.abs(v_projected_flipped)
                * self.cfg.rho
                * self.cfg.wing_area_projected
                / 2
            )

            forces[:, body_idx, :] = drag + lift_turbulent
        forces = quat_apply_inverse(quat_w, forces)
        self._instantaneous_wrench_composer.add_forces_and_torques(
            self._ALL_INDICES_WP,
            self._ALL_BODY_INDICES_WP,  # base_link only
            forces=wp.from_torch(forces, dtype=wp.vec3f),
            torques=wp.from_torch(torques, dtype=wp.vec3f),
            is_global=False,
        )

    def _apply_drag(self):
        """Apply aerodynamic drag forces and torques to the base link.

        The drag model follows:
        - Linear drag: F = -C_lin * v - C_quad * |v| * v (world-frame)
        - Angular drag: T = -C_lin * w - C_quad * |w| * w (body-frame)
        """
        # Get world-frame velocities for base link (body index 0)
        v = self.data.body_lin_vel_w[:, 0, :]
        w_world = self.data.body_ang_vel_w[:, 0, :]

        # Convert angular velocity from world to body frame
        quat_w = self.data.body_quat_w[:, 0, :]
        w = quat_apply_inverse(quat_w, w_world)

        # Compute linear drag in world frame
        lin_drag = -self.cfg.lin_drag_linear_coef * v
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        lin_drag = lin_drag - self.cfg.lin_drag_quadratic_coef * v_norm * v

        # Compute angular drag in body frame
        ang_drag = -self.cfg.ang_drag_linear_coef * w
        w_norm = torch.linalg.vector_norm(w, dim=-1, keepdim=True)
        ang_drag = ang_drag - self.cfg.ang_drag_quadratic_coef * w_norm * w

        # Reshape for wrench composer
        forces = lin_drag.unsqueeze(1)  # world-frame
        torques = ang_drag.unsqueeze(1)  # body-frame

        # Add drag to instantaneous wrench composer
        self._instantaneous_wrench_composer.add_forces_and_torques(
            self._ALL_INDICES_WP,
            self._ALL_BODY_INDICES_WP[:1],  # base_link only
            forces=wp.from_torch(forces, dtype=wp.vec3f),
            torques=wp.from_torch(torques, dtype=wp.vec3f),
            is_global=False,
        )
