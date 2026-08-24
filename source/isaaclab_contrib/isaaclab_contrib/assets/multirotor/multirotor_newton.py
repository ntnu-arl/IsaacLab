# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton implementation of the multirotor asset API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.assets.articulation import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.kernels import split_state_to_root_pose_and_vel

from .multirotor_base import MultirotorBase
from .multirotor_data_base import MultirotorDataBase

if TYPE_CHECKING:
    from .multirotor_cfg import MultirotorCfg


@wp.kernel
def _write_body_frame_wrench_to_newton(
    forces_b: wp.array2d(dtype=wp.vec3f),
    torques_b: wp.array2d(dtype=wp.vec3f),
    body_link_quat_w: wp.array2d(dtype=wp.quatf),
    wrench_w: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
):
    """Rotate body-frame loads into Newton's world-frame external-wrench buffer."""
    env_index, body_index = wp.tid()
    if env_mask[env_index] and body_mask[body_index]:
        link_quat_w = body_link_quat_w[env_index, body_index]
        force_w = wp.quat_rotate(link_quat_w, forces_b[env_index, body_index])
        torque_w = wp.quat_rotate(link_quat_w, torques_b[env_index, body_index])
        wrench_w[env_index, body_index] = wp.spatial_vector(force_w, torque_w, wp.float32)


class MultirotorDataNewton(MultirotorDataBase, ArticulationData):
    """Newton data container implementing the common multirotor data API."""


class MultirotorNewton(MultirotorBase, Articulation):
    """Newton implementation of :class:`MultirotorBase`."""

    __backend_name__: str = "newton"

    def __init__(self, cfg: MultirotorCfg):
        """Initialize the Newton multirotor.

        Args:
            cfg: A multirotor configuration instance.
        """
        super().__init__(cfg)

    def _create_multirotor_data(self) -> MultirotorDataNewton:
        """Create the Newton multirotor data container."""
        return MultirotorDataNewton(self.root_view, self.device)

    def _set_default_root_state(self, default_root_state: torch.Tensor) -> None:
        """Convert and store the configured root state in Newton storage."""
        default_root_state_wp = wp.from_torch(default_root_state, dtype=wp.float32)
        pose_output = wp.zeros(self.num_instances, dtype=wp.transformf, device=self.device)
        vel_output = wp.zeros(self.num_instances, dtype=wp.spatial_vectorf, device=self.device)
        wp.launch(
            split_state_to_root_pose_and_vel,
            dim=self.num_instances,
            inputs=[default_root_state_wp],
            outputs=[pose_output, vel_output],
            device=self.device,
        )
        self.data.default_root_pose = pose_output
        self.data.default_root_vel = vel_output

    def _write_external_wrenches_to_sim(self) -> None:
        """Compose and apply pending external wrenches to Newton."""
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                composer = self._instantaneous_wrench_composer
                composer.add_raw_buffers_from(self._permanent_wrench_composer)
            else:
                composer = self._permanent_wrench_composer
            # Newton's external-wrench state stores world-frame loads, while the shared wrench composer outputs
            # body-frame loads. Rotate the composed loads using the current link orientations before binding them.
            composer.compose_to_body_frame()
            wp.launch(
                _write_body_frame_wrench_to_newton,
                dim=(self.num_instances, self.num_bodies),
                device=self.device,
                inputs=[
                    composer.out_force_b,
                    composer.out_torque_b,
                    self.data.body_link_quat_w.warp,
                    self._data._sim_bind_body_external_wrench,
                    self._ALL_ENV_MASK,
                    self._ALL_BODY_MASK,
                ],
            )
