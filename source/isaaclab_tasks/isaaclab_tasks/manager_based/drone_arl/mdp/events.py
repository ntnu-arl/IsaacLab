# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions specific to the drone ARL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .curriculums import get_obstacle_curriculum_term

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection
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
    determined by the curriculum difficulty level. Inactive obstacles are parked at distinct
    locations far below the scene to avoid overlapping collision geometry.

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

    num_objects = obstacles.num_bodies
    num_envs = len(env_ids)
    object_names = obstacles.body_names

    # Get difficulty levels per environment
    if use_curriculum:
        curriculum_term = get_obstacle_curriculum_term(env)
        if curriculum_term is not None:
            # Get difficulty levels for the specific environments being reset
            difficulty_levels = curriculum_term.difficulty_levels[env_ids]
            max_difficulty = curriculum_term.max_difficulty
        else:
            # Fallback: use max obstacles if curriculum not found
            difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
            max_difficulty = max_num_obstacles
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
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    all_poses[..., 3:7] = identity_quat

    # Place walls
    wall_entries = [(object_names.index(name), cfg) for name, cfg in wall_configs.items() if name in object_names]
    if wall_entries:
        wall_indices = [entry[0] for entry in wall_entries]
        wall_min_ratios = torch.tensor(
            [entry[1].center_ratio_min for entry in wall_entries], dtype=torch.float32, device=env.device
        )
        wall_max_ratios = torch.tensor(
            [entry[1].center_ratio_max for entry in wall_entries], dtype=torch.float32, device=env.device
        )
        wall_center_ratios = wall_min_ratios.unsqueeze(0).expand(num_envs, -1, -1).clone()
        variable_wall_indices = [
            i
            for i, (_, wall_cfg) in enumerate(wall_entries)
            if wall_cfg.center_ratio_min != wall_cfg.center_ratio_max
        ]
        if variable_wall_indices:
            wall_ratios = torch.rand(num_envs, len(variable_wall_indices), 3, device=env.device)
            wall_center_ratios[:, variable_wall_indices] = (
                wall_ratios
                * (wall_max_ratios[variable_wall_indices] - wall_min_ratios[variable_wall_indices])
                + wall_min_ratios[variable_wall_indices]
            )
        wall_positions = (wall_center_ratios - 0.5) * env_size_t
        wall_positions[..., 2] += ground_offset
        wall_positions += env.scene.env_origins[env_ids].unsqueeze(1)
        all_poses[:, wall_indices, 0:3] = wall_positions

    # Get obstacle indices
    obstacle_indices = [idx for idx, name in enumerate(object_names) if name not in wall_names]

    if len(obstacle_indices) == 0:
        obstacles.write_body_pose_to_sim_index(body_poses=all_poses, env_ids=env_ids)
        obstacles.write_body_com_velocity_to_sim_index(body_velocities=all_velocities, env_ids=env_ids)
        return

    num_obstacles = len(obstacle_indices)

    # Select the requested number of unique obstacles per environment
    random_order = torch.argsort(torch.rand(num_envs, num_obstacles, device=env.device), dim=1)
    active_by_rank = torch.arange(num_obstacles, device=env.device).unsqueeze(0) < obstacles_per_env.unsqueeze(1)
    active_masks = torch.zeros(num_envs, num_obstacles, dtype=torch.bool, device=env.device)
    active_masks.scatter_(1, random_order, active_by_rank)

    # Sample every obstacle in one operation.
    obstacle_min_ratios = torch.tensor(
        [obstacle_types[i % len(obstacle_types)].center_ratio_min for i in range(num_obstacles)],
        dtype=torch.float32,
        device=env.device,
    )
    obstacle_max_ratios = torch.tensor(
        [obstacle_types[i % len(obstacle_types)].center_ratio_max for i in range(num_obstacles)],
        dtype=torch.float32,
        device=env.device,
    )
    obstacle_ratios = torch.rand(num_envs, num_obstacles, 3, device=env.device)
    obstacle_positions = (
        obstacle_ratios * (obstacle_max_ratios - obstacle_min_ratios) + obstacle_min_ratios - 0.5
    ) * env_size_t
    obstacle_positions[..., 2] += ground_offset
    obstacle_positions += env.scene.env_origins[env_ids].unsqueeze(1)

    # Inactive samples are discarded
    inactive_positions = env.scene.env_origins[env_ids].unsqueeze(1).expand(-1, num_obstacles, -1).clone()
    inactive_positions[..., 2] += -1000.0 - 5.0 * torch.arange(num_obstacles, device=env.device)
    obstacle_positions = torch.where(active_masks.unsqueeze(-1), obstacle_positions, inactive_positions)

    obstacle_quats = math_utils.random_orientation(num_envs * num_obstacles, device=env.device).view(
        num_envs, num_obstacles, 4
    )
    obstacle_quats = torch.where(active_masks.unsqueeze(-1), obstacle_quats, identity_quat)

    all_poses[:, obstacle_indices, 0:3] = obstacle_positions
    all_poses[:, obstacle_indices, 3:7] = obstacle_quats

    # Write to sim
    obstacles.write_body_pose_to_sim_index(body_poses=all_poses, env_ids=env_ids)
    obstacles.write_body_com_velocity_to_sim_index(body_velocities=all_velocities, env_ids=env_ids)
