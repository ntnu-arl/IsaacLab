# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton implementation scaffold for the multirotor asset API."""

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


class MultirotorDataNewton(MultirotorDataBase, ArticulationData):
    """Newton data container implementing the common multirotor data API."""


class MultirotorNewton(MultirotorBase, Articulation):
    """Newton implementation scaffold for :class:`MultirotorBase`.

    The common multirotor API and state handling are available, but the Newton wrench write remains part of the next
    MR-04 implementation step.
    """

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
        # TODO: write the composed body wrenches through Newton's bound external-wrench array.
        raise NotImplementedError("Newton multirotor wrench application is not implemented yet.")
