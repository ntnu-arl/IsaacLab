# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

from .obstacle_scene_cfg import ObstaclesSceneCfg

"""Obstacle scene generation and reset functionality for drone navigation environments.

This module provides utilities for generating dynamic 3D obstacle courses with walls and
floating obstacles. The obstacle configurations support curriculum learning where difficulty
can be progressively increased by adjusting the number of active obstacles.
"""

OBSTACLE_SCENE_CFG = ObstaclesSceneCfg(
    env_size=(12.0, 8.0, 6.0),
    min_num_obstacles=20,
    max_num_obstacles=40,
    ground_offset=3.0,
)


def generate_obstacle_collection(cfg: ObstaclesSceneCfg, kinematic: bool = False) -> RigidObjectCollectionCfg:
    """Generate a rigid object collection configuration for walls and obstacles.

    Creates a complete scene with boundary walls and a variety of floating obstacles
    (panels, cubes, rods, etc.) based on the provided configuration. Each obstacle is
    assigned random colors and configured with appropriate physics properties.

    PhysX uses the task's original massive, velocity-limited rigid bodies. Newton can use
    kinematic bodies to represent the same stationary geometry without the numerically
    problematic mass and damping values.

    Args:
        cfg: Configuration object specifying obstacle types, sizes, quantities, and
            positioning constraints.
        kinematic: Whether to represent walls and obstacles as kinematic bodies.

    Returns:
        A RigidObjectCollectionCfg containing all wall and obstacle configurations,
        ready to be added to a scene.

    Note:
        Objects are initially parked at distinct positions below the scene so the physics
        solver never observes overlapping geometry before the reset event places them.
        All collection members live under the ``Obstacles`` prim so the collection's
        combined path pattern cannot also match sibling assets such as the robot.
    """
    max_num_obstacles = cfg.max_num_obstacles

    rigid_objects = {}

    if kinematic:
        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            disable_gravity=True,
            kinematic_enabled=True,
        )
        wall_mass_props = None
        obstacle_rigid_props = wall_rigid_props
        obstacle_mass_props = None
    else:
        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            disable_gravity=True,
            kinematic_enabled=False,
            linear_damping=9999.0,
            angular_damping=9999.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        )
        wall_mass_props = sim_utils.MassPropertiesCfg(mass=10000000.0)
        obstacle_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            disable_gravity=True,
            kinematic_enabled=False,
            linear_damping=1.0,
            angular_damping=1.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        )
        obstacle_mass_props = sim_utils.MassPropertiesCfg(mass=100.0)

    for wall_name, wall_cfg in cfg.wall_cfgs.items():
        center_ratio = np.asarray(wall_cfg.center_ratio_min)
        default_center = (center_ratio - 0.5) * np.asarray(cfg.env_size)
        default_center[2] += cfg.ground_offset
        color = float(np.random.randint(0, 256, dtype=np.uint8)) / 255.0

        rigid_objects[wall_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacles/obstacle_{wall_name}",
            spawn=sim_utils.CuboidCfg(
                size=wall_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, color), metallic=0.2),
                rigid_props=wall_rigid_props,
                mass_props=wall_mass_props,
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center.tolist())),
            collision_group=0,
        )

    obstacle_types = list(cfg.obstacle_cfgs.values())
    for i in range(max_num_obstacles):
        obj_name = f"obstacle_{i}"
        obs_cfg = obstacle_types[i % len(obstacle_types)]

        default_center = [0.0, 0.0, -1000.0 - 10.0 * (len(cfg.wall_cfgs) + i)]
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        color_normalized = tuple(float(c) / 255.0 for c in color)

        rigid_objects[obj_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacles/{obj_name}",
            spawn=sim_utils.CuboidCfg(
                size=obs_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_normalized, metallic=0.2),
                rigid_props=obstacle_rigid_props,
                mass_props=obstacle_mass_props,
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=0,
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)
