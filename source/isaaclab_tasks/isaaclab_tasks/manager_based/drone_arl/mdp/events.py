# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions specific to the drone ARL environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_obstacles_with_individual_ranges(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    obstacle_configs: dict,
    wall_configs: dict,
    env_size: tuple[float, float, float],
    use_curriculum: bool = True,
    min_num_obstacles: int = 1,
    max_num_obstacles: int = 10,
    ground_offset: float = 0.1,
) -> None:
    """Reset obstacle and wall positions for specified environments without collision checking.

    This function repositions all walls and a curriculum-determined subset of obstacles
    within the specified environment bounds.

    Walls are positioned at fixed locations based on their configuration ratios. Obstacles
    are randomly placed within their designated zones, with the number of active obstacles
    determined by the curriculum difficulty level. Inactive obstacles are moved far below
    the scene (-1000m in Z) to effectively remove them from the environment.

    The curriculum scaling works as:
        num_obstacles = min + (difficulty / max_difficulty) * (max - min)

    Args:
        env: The manager-based RL environment instance.
        env_ids: Tensor of environment indices to reset.
        asset_cfg: Scene entity configuration identifying the obstacle collection.
        obstacle_configs: Dictionary mapping obstacle type names to their BoxCfg
            configurations, specifying size and placement ranges.
        wall_configs: Dictionary mapping wall names to their BoxCfg configurations.
        env_size: Tuple of (length, width, height) defining the environment bounds in meters.
        use_curriculum: If True, number of obstacles scales with curriculum difficulty.
            If False, spawns max_num_obstacles in every environment. Defaults to True.
        min_num_obstacles: Minimum number of obstacles to spawn per environment.
            Defaults to 1.
        max_num_obstacles: Maximum number of obstacles to spawn per environment.
            Defaults to 10.
        ground_offset: Z-axis offset to prevent obstacles from spawning at z=0.
            Defaults to 0.1 meters.

    Note:
        This function expects the environment to have `_obstacle_difficulty_levels` and
        `_max_obstacle_difficulty` attributes when `use_curriculum=True`. These are
        typically set by :func:`obstacle_density_curriculum`.
    """
    obstacles: RigidObjectCollection = env.scene[asset_cfg.name]

    num_objects = obstacles.num_objects
    num_envs = len(env_ids)
    object_names = obstacles.object_names

    # Get difficulty levels per environment
    if use_curriculum and hasattr(env, "_obstacle_difficulty_levels"):
        difficulty_levels = env._obstacle_difficulty_levels[env_ids]
        max_difficulty = env._max_obstacle_difficulty
    else:
        difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
        max_difficulty = max_num_obstacles

    # Calculate active obstacles per env based on difficulty
    obstacles_per_env = (
        min_num_obstacles + (difficulty_levels / max_difficulty) * (max_num_obstacles - min_num_obstacles)
    ).long()

    # Prepare tensors
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)

    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    env_size_t = torch.tensor(env_size, device=env.device)

    # place walls
    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)

            min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)

            if torch.allclose(min_ratio, max_ratio):
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                ratios = torch.rand(num_envs, 3, device=env.device)
                center_ratios = ratios * (max_ratio - min_ratio) + min_ratio

            positions = (center_ratios - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            positions += env.scene.env_origins[env_ids]

            all_poses[:, wall_idx, 0:3] = positions
            all_poses[:, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    # Get obstacle indices
    obstacle_indices = [idx for idx, name in enumerate(object_names) if name not in wall_names]

    if len(obstacle_indices) == 0:
        obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
        obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
        return

    # Determine which obstacles are active per env
    active_masks = torch.zeros(num_envs, len(obstacle_indices), dtype=torch.bool, device=env.device)
    for env_idx in range(num_envs):
        num_active = obstacles_per_env[env_idx].item()
        perm = torch.randperm(len(obstacle_indices), device=env.device)[:num_active]
        active_masks[env_idx, perm] = True

    # place obstacles
    for obj_list_idx in range(len(obstacle_indices)):
        obj_idx = obstacle_indices[obj_list_idx]

        # Which envs need this obstacle?
        envs_need_obstacle = active_masks[:, obj_list_idx]

        if not envs_need_obstacle.any():
            # Move all to -1000
            all_poses[:, obj_idx, 0:3] = env.scene.env_origins[env_ids] + torch.tensor(
                [0.0, 0.0, -1000.0], device=env.device
            )
            all_poses[:, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue

        # Get obstacle config
        config_idx = obj_list_idx % len(obstacle_types)
        obs_cfg = obstacle_types[config_idx]

        min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
        max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)

        # sample object positions
        num_active_envs = envs_need_obstacle.sum().item()
        ratios = torch.rand(num_active_envs, 3, device=env.device)
        positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
        positions[:, 2] += ground_offset

        # Add env origins
        active_env_indices = torch.where(envs_need_obstacle)[0]
        positions += env.scene.env_origins[env_ids[active_env_indices]]

        # Generate quaternions
        quats = math_utils.random_orientation(num_envs, device=env.device)

        # Write poses
        all_poses[envs_need_obstacle, obj_idx, 0:3] = positions
        all_poses[envs_need_obstacle, obj_idx, 3:7] = quats[envs_need_obstacle]

        # Move inactive obstacles far away
        inactive = ~envs_need_obstacle
        all_poses[inactive, obj_idx, 0:3] = env.scene.env_origins[env_ids[inactive]] + torch.tensor(
            [0.0, 0.0, -1000.0], device=env.device
        )
        all_poses[inactive, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)


def apply_disturbance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_range: tuple[float, float] = (-2.0, 2.0),
    torque_range: tuple[float, float] = (-0.3, 0.3),
    body_name: str = "base_link",
    duration_steps: int = 1,
):
    """Apply a random disturbance to the drone body.

    This function applies a force and torque disturbance to the drone body. The duration
    determines whether it's an impulse (1 step) or continuous (multiple steps).

    Args:
        env: The environment instance.
        env_ids: Environment indices to apply the disturbance to.
        asset_cfg: Configuration for the asset to apply the disturbance to.
        force_range: Range for force magnitude in each axis (x, y, z). Defaults to (-2.0, 2.0).
        torque_range: Range for torque magnitude in each axis (x, y, z). Defaults to (-0.3, 0.3).
        body_name: Name of the body to apply the disturbance to. Defaults to "base_link".
        duration_steps: Number of simulation steps to apply the disturbance.
            If 1, uses instantaneous (impulse). If > 1, uses permanent (continuous).
            Defaults to 1.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # find body ID
    body_ids, _ = asset.find_bodies(body_name, preserve_order=True)
    if len(body_ids) == 0:
        return

    # Convert env_ids to int32 if it's a tensor (warp expects int32)
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.to(torch.int32)

    # sample random forces and torques - shape: (len(env_ids), 1, 3)
    size = (len(env_ids), 1, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)

    # choose composer based on duration
    if duration_steps == 1:
        # impulse: use instantaneous (auto-clears after 1 step)
        asset.add_instantaneous_external_wrench(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=False,
        )
    else:
        # continuous: use permanent (persists until cleared)
        asset.set_permanent_external_wrench(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=False,
        )
        # Note: For duration_steps > 1, you'll need a separate clear event
        # or manually clear it after the desired duration


def clear_disturbance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Clear permanent disturbances from the drone.

    This function clears any permanent external wrenches that were previously applied
    via :func:`apply_disturbance` with duration_steps > 1.

    Args:
        env: The environment instance.
        env_ids: Environment indices to clear the disturbance from.
        asset_cfg: Configuration for the asset to clear the disturbance from.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Convert env_ids to int32 if it's a tensor (warp expects int32)
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.to(torch.int32)

    # clear permanent wrench by setting zeros with correct shape
    # Determine the number of environments and bodies
    num_envs = len(env_ids)
    body_ids, _ = asset.find_bodies("base_link", preserve_order=True)
    num_bodies = len(body_ids)

    # Create zero tensors with correct shape: (num_envs, num_bodies, 3)
    zero_forces = torch.zeros((num_envs, num_bodies, 3), device=asset.device)
    zero_torques = torch.zeros((num_envs, num_bodies, 3), device=asset.device)

    asset.set_permanent_external_wrench(
        forces=zero_forces,
        torques=zero_torques,
        body_ids=body_ids,
        env_ids=env_ids,
    )


class apply_continuous_disturbance_with_duration(ManagerTermBase):
    """Apply a continuous disturbance that auto-clears after a specified duration.

    This class-based event applies a disturbance once when triggered, then automatically
    clears it after the specified duration. Unlike function-based events, this maintains
    state to track when disturbances were applied.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Extract parameters
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.force_range = cfg.params.get("force_range", (-1.0, 1.0))
        self.torque_range = cfg.params.get("torque_range", (-0.2, 0.2))
        self.body_name = cfg.params.get("body_name", "base_link")
        self.duration_s = cfg.params.get("duration_s", 1.0)  # Duration in seconds

        # Track when disturbances were applied (per environment)
        # Stores the episode time when disturbance was applied, or -1 if not applied
        self._disturbance_start_time = torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float32)

        # Store the asset reference
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.body_ids, _ = self.asset.find_bodies(self.body_name, preserve_order=True)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset the disturbance tracking.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
        """
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, (slice, torch.Tensor)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # Reset tracking
        if isinstance(env_ids, slice):
            self._disturbance_start_time[env_ids] = -1.0
        else:
            self._disturbance_start_time[env_ids] = -1.0

        # Clear any active disturbances by setting zeros with correct shape
        # Determine the number of environments and bodies
        if isinstance(env_ids, slice):
            num_envs = self.num_envs
        elif isinstance(env_ids, torch.Tensor):
            num_envs = len(env_ids)
        else:
            num_envs = len(env_ids)
        num_bodies = len(self.body_ids)

        # Create zero tensors with correct shape: (num_envs, num_bodies, 3)
        zero_forces = torch.zeros((num_envs, num_bodies, 3), device=self.device)
        zero_torques = torch.zeros((num_envs, num_bodies, 3), device=self.device)

        self.asset.set_permanent_external_wrench(
            forces=zero_forces,
            torques=zero_torques,
            body_ids=self.body_ids,
            env_ids=env_ids,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        body_name: str,
        duration_s: float,
    ) -> None:
        """Apply or clear disturbance based on elapsed time.

        This method is called by the event manager. It checks if disturbances need to be
        applied (when triggered) or cleared (after duration has elapsed).

        Note: The parameters are already stored in __init__, but they must be listed here
        to match the event configuration for validation purposes.

        Args:
            env: The environment instance.
            env_ids: Environment indices. If None, uses all environments.
            asset_cfg: Configuration for the asset (stored in __init__, ignored here).
            force_range: Range for force magnitude (stored in __init__, ignored here).
            torque_range: Range for torque magnitude (stored in __init__, ignored here).
            body_name: Name of the body (stored in __init__, ignored here).
            duration_s: Duration in seconds (stored in __init__, ignored here).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Convert env_ids to int32 if it's a tensor (warp expects int32)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(torch.int32)

        # Get current episode time for each environment
        current_time = env.episode_length_buf.float() * env.step_dt

        # Check which environments need disturbance applied (not yet applied)
        needs_apply = self._disturbance_start_time[env_ids] < 0

        # Check which environments need disturbance cleared (duration exceeded)
        elapsed_time = current_time[env_ids] - self._disturbance_start_time[env_ids]
        needs_clear = (self._disturbance_start_time[env_ids] >= 0) & (elapsed_time >= self.duration_s)

        # Apply disturbances to environments that need them
        if torch.any(needs_apply):
            apply_env_ids = env_ids[needs_apply]
            size = (len(apply_env_ids), 1, 3)
            forces = math_utils.sample_uniform(*self.force_range, size, self.device)
            torques = math_utils.sample_uniform(*self.torque_range, size, self.device)

            self.asset.set_permanent_external_wrench(
                forces=forces,
                torques=torques,
                body_ids=self.body_ids,
                env_ids=apply_env_ids,
                is_global=False,
            )
            # Record when disturbance was applied
            self._disturbance_start_time[apply_env_ids] = current_time[apply_env_ids]

        # Clear disturbances that have exceeded duration
        if torch.any(needs_clear):
            clear_env_ids = env_ids[needs_clear]
            # Create zero tensors with correct shape: (len(clear_env_ids), num_bodies, 3)
            num_bodies = len(self.body_ids)
            zero_forces = torch.zeros((len(clear_env_ids), num_bodies, 3), device=self.device)
            zero_torques = torch.zeros((len(clear_env_ids), num_bodies, 3), device=self.device)

            self.asset.set_permanent_external_wrench(
                forces=zero_forces,
                torques=zero_torques,
                body_ids=self.body_ids,
                env_ids=clear_env_ids,
            )
            # Reset tracking
            self._disturbance_start_time[clear_env_ids] = -1.0
