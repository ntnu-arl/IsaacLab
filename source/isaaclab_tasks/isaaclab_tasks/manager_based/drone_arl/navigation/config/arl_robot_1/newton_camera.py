# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton depth-camera adapter for the ARL navigation task."""

import torch
import warp as wp

from isaaclab.sensors.ray_caster.kernels import copy_mesh_poses_to_table_kernel

from isaaclab_newton.sensors import MultiMeshRayCasterCamera


class ArlNewtonMultiMeshRayCasterCamera(MultiMeshRayCasterCamera):
    """Adapt clone-plan globs and consolidate obstacle pose tracking."""

    def _resolve_target_owner_exprs(self, prim_expr: str) -> list[str]:
        owner_exprs = super()._resolve_target_owner_exprs(prim_expr)
        return [owner_expr.replace("env_*", "env_.*") for owner_expr in owner_exprs]

    def _initialize_warp_meshes(self) -> None:
        super()._initialize_warp_meshes()

        if any(view is None for view in self._mesh_views):
            return
        if any(self._num_meshes_per_env[cfg.prim_expr] != 1 for cfg in self._raycast_targets_cfg):
            return
        if any(view.shape[0] != self._num_envs for view in self._mesh_views):
            return

        # Each target view is ordered by environment
        site_indices = (
            torch.stack([wp.to_torch(view) for view in self._mesh_views], dim=1).reshape(-1).contiguous()
        )
        self._arl_mesh_site_indices = wp.from_torch(site_indices, dtype=wp.int32)
        site_count = site_indices.shape[0]
        self._arl_mesh_pose_w = wp.empty(site_count, dtype=wp.transformf, device=self._device)
        self._arl_mesh_pos_w = wp.empty(site_count, dtype=wp.vec3f, device=self._device)
        self._arl_mesh_quat_w = wp.empty(site_count, dtype=wp.quatf, device=self._device)

    def _update_mesh_transforms(self) -> None:
        if not hasattr(self, "_arl_mesh_site_indices"):
            super()._update_mesh_transforms()
            return

        self._update_newton_site_transforms(
            self._arl_mesh_site_indices,
            self._arl_mesh_pose_w,
            self._arl_mesh_pos_w,
            self._arl_mesh_quat_w,
        )
        num_meshes = len(self._mesh_views)
        wp.launch(
            copy_mesh_poses_to_table_kernel,
            dim=(self._num_envs, num_meshes),
            inputs=[
                self._arl_mesh_pos_w,
                self._arl_mesh_quat_w,
                num_meshes,
                0,
                False,
                self._mesh_positions_w,
                self._mesh_orientations_w,
            ],
            device=self._device,
        )
