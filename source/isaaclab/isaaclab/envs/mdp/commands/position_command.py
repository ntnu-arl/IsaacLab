# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformWaypointCommandCfg


class UniformWaypointCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformWaypointCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformWaypointCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers
        # -- commands: (x, y, altitude, radius) in root frame
        self.waypoint_command = torch.zeros(self.num_envs, 4, device=self.device)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["altitude_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformWaypointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.waypoint_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        root_link_pos = self.robot.data.root_com_pos_w
        pos_error_z = root_link_pos[:, 2] - self.waypoint_command[:, 2]
        pos_error_xy = (
            torch.norm(root_link_pos[:, :2] - self.waypoint_command[:, :2], dim=-1)
            - self.waypoint_command[:, 3]
        )

        self.metrics["position_error"] = torch.abs(pos_error_xy)
        self.metrics["altitude_error"] = torch.abs(pos_error_z)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.waypoint_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.x)
        self.waypoint_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.y)
        self.waypoint_command[env_ids, :2] += self._env.scene.env_origins[env_ids, :2]
        self.waypoint_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.altitude)
        if self.cfg.ranges.radius is not None:
            self.waypoint_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.radius)
        else:
            self.waypoint_command[env_ids, 3] = 0.0

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pos_visualizer = VisualizationMarkers(
                    self.cfg.goal_pos_visualizer_cfg
                )
                # -- current body pose
                self.current_pos_visualizer = VisualizationMarkers(
                    self.cfg.current_pos_visualizer_cfg
                )
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.current_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.current_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers

        # -- goal pose
        self.goal_pos_visualizer.visualize(
            self.waypoint_command[:, :3],
            None,
        )
        # -- current body pose
        target_vector = (
            self.robot.data.root_com_pos_w[:, :2] - self.waypoint_command[:, :2]
        )
        heading_angle = wrap_to_pi(
            torch.atan2(target_vector[:, 1], target_vector[:, 0]) + torch.pi
        )
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        self.current_pos_visualizer.visualize(
            self.robot.data.root_com_pos_w, arrow_quat
        )
