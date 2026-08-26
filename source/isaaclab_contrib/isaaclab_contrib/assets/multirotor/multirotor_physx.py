# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX implementation of the multirotor asset API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_physx.assets.articulation import Articulation
from isaaclab_physx.assets.articulation.articulation_data import ArticulationData
from isaaclab_physx.assets.kernels import split_state_to_root_pose_and_vel

from .multirotor import Multirotor
from .multirotor_data import MultirotorData

if TYPE_CHECKING:
    from .multirotor_cfg import MultirotorCfg


class MultirotorDataPhysx(MultirotorData, ArticulationData):
    """PhysX data container implementing the common multirotor data API."""


class MultirotorPhysx(Multirotor, Articulation):
    """PhysX implementation of :class:`Multirotor`."""

    __backend_name__: str = "physx"

    def __init__(self, cfg: MultirotorCfg):
        """Initialize the PhysX multirotor.

        Args:
            cfg: A multirotor configuration instance.
        """
        super().__init__(cfg)

    def _create_multirotor_data(self) -> MultirotorDataPhysx:
        """Create the PhysX multirotor data container."""
        return MultirotorDataPhysx(self.root_view, self.device)

    def _set_default_root_state(self, default_root_state: torch.Tensor) -> None:
        """Convert and store the configured root state in PhysX storage."""
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
        """Compose and apply pending external wrenches to PhysX."""
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                composer = self._instantaneous_wrench_composer
                composer.add_raw_buffers_from(self._permanent_wrench_composer)
            else:
                composer = self._permanent_wrench_composer
            composer.compose_to_body_frame()
            self.root_view.apply_forces_and_torques_at_position(
                force_data=composer.out_force_b.warp.flatten().view(wp.float32),
                torque_data=composer.out_torque_b.warp.flatten().view(wp.float32),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )
