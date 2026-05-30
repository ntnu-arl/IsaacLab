# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import (
        Figure8AirspeedHeadingCommandCfg,
        NormalVelocityCommandCfg,
        UniformVelocityCommandCfg,
        UniformAirspeedHeadingCommandCfg,
    )


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        # update standing envs
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_att_w = math_utils.quat_inv(self.robot.data.root_quat_w.clone())

        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_w[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w,
            math_utils.quat_mul(base_att_w, vel_des_arrow_quat),
            vel_des_arrow_scale,
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, math_utils.quat_mul(base_att_w, vel_arrow_quat), vel_arrow_scale
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(
            mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0]
        )
        self.vel_command_b[env_ids, 0] *= torch.where(
            r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0
        )
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(
            mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1]
        )
        self.vel_command_b[env_ids, 1] *= torch.where(
            r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0
        )
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(
            mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2]
        )
        self.vel_command_b[env_ids, 2] *= torch.where(
            r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0
        )

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        )
        self.is_zero_vel_y_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        )
        self.is_zero_vel_yaw_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]
        )

        # update standing envs
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(
            as_tuple=False
        ).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(
            as_tuple=False
        ).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0


class UniformAirspeedHeadingCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformAirspeedHeadingCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformAirspeedHeadingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.alt_command and self.cfg.ranges.altitude is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.altitude and not self.cfg.alt_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.altitude' attribute set to '{self.cfg.ranges.altitude}'"
                " but the altitude command is not active. Consider setting the flag for the altitude command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.ahv_command = torch.zeros(self.num_envs, 3, device=self.device)

        # -- metrics
        self.metrics["error_airspeed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vertical"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.ahv_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_airspeed"] += (
            self.ahv_command[:, 0] - self.robot.data.root_lin_vel_b[:, 0]
        ) / max_command_step
        self.metrics["error_heading"] += (
            self.ahv_command[:, 1]
            - torch.tanh(
                self.robot.data.root_lin_vel_w[:, 1]
                / self.robot.data.root_lin_vel_w[:, 0]
                + 1e-6,
            )
        ) / max_command_step
        if self.cfg.alt_command:
            self.metrics["error_vertical"] += (
                self.ahv_command[:, 2] - self.robot.data.body_pos_w[:, 0, 2]
            ) / max_command_step
        else:
            self.metrics["error_vertical"] += (
                self.ahv_command[:, 2] - self.robot.data.root_lin_vel_w[:, 2]
            ) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.ahv_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.airspeed)
        # -- linear velocity - y direction
        self.ahv_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.heading)
        # -- ang vel yaw - rotation around z
        if self.cfg.alt_command:
            self.ahv_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.altitude)
        else:
            self.ahv_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.vertical_speed)

    def _update_command(self):
        """Post-processes the velocity command."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_att_w = math_utils.quat_inv(self.robot.data.root_quat_w.clone())

        base_pos_w[:, 2] += 0.1
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_ah_velocity_to_arrow(
            self.ahv_command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_w[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w,
            math_utils.quat_mul(base_att_w, vel_des_arrow_quat),
            vel_des_arrow_scale,
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, math_utils.quat_mul(base_att_w, vel_arrow_quat), vel_arrow_scale
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1)
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_ah_velocity_to_arrow(
        self, ah_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            ah_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= ah_velocity[:, 0]
        # arrow-direction
        heading_angle = ah_velocity[:, 1]
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class Figure8AirspeedHeadingCommand(CommandTerm):
    r"""Command generator that generates a figure-8 target position, airspeed, and heading.

    The command comprises a target position in world frame, an airspeed command, and a heading
    command. The target position follows a time-parameterized figure-8 curve, while the airspeed
    and altitude are sampled when the command is resampled. The trajectory center is slowly
    adapted toward the robot position and is reset to the robot position when the separation gets
    too large.

    The figure-8 geometry is controlled by the sampled ``size`` parameter. Larger sizes produce
    slower turns for the same commanded airspeed.
    """

    cfg: Figure8AirspeedHeadingCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: Figure8AirspeedHeadingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the altitude command is active but the altitude range is not provided.
        """
        super().__init__(cfg, env)

        if self.cfg.ranges.altitude is None:
            raise ValueError(
                "The figure-8 command requires the `ranges.altitude` parameter to be set."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.sha_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._figure8_center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._figure8_size = torch.ones(self.num_envs, device=self.device)
        self._figure8_phase_offset = torch.zeros(self.num_envs, device=self.device)
        self._figure8_altitude = torch.zeros(self.num_envs, device=self.device)
        self._figure8_time = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_position"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_airspeed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "Figure8AirspeedHeadingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTarget position enabled: True"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired command in world frame. Shape is (num_envs, 5)."""
        return torch.cat([self.sha_command, self.pos_command_w], dim=1)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        self.metrics["error_position"] += (
            torch.linalg.norm(self.pos_command_w - self.robot.data.root_pos_w, dim=-1)
            / max_command_step
        )
        self.metrics["error_airspeed"] += (
            self.sha_command[:, 0] - self.robot.data.root_lin_vel_b[:, 0]
        ) / max_command_step
        self.metrics["error_heading"] += (
            torch.abs(
                math_utils.wrap_to_pi(
                    self.sha_command[:, 1] - self.robot.data.heading_w
                )
            )
        ) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.sha_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.airspeed)
        self._figure8_center_w[env_ids] = self._env.scene.env_origins[env_ids]
        self._figure8_size[env_ids] = r.uniform_(*self.cfg.ranges.size)
        self._figure8_phase_offset[env_ids] = 0  # r.uniform_(0.0, 2.0 * torch.pi)
        self._figure8_altitude[env_ids] = r.uniform_(*self.cfg.ranges.altitude)
        self._figure8_time[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        self._figure8_time += dt

        size = torch.clamp(self._figure8_size, min=1e-6)
        phase = (
            self._figure8_phase_offset
            + (self.sha_command[:, 0] / size) * self._figure8_time
        )

        self.pos_command_w[:, 0] = self._figure8_center_w[:, 0] + size * torch.sin(
            phase
        )
        self.pos_command_w[:, 1] = self._figure8_center_w[
            :, 1
        ] + 0.5 * size * torch.sin(2.0 * phase)
        self.pos_command_w[:, 2] = self._figure8_altitude

        tangent_x = torch.cos(phase)
        tangent_y = torch.cos(2.0 * phase)
        self.sha_command[:, 1] = math_utils.wrap_to_pi(
            torch.atan2(tangent_y, tangent_x)
        )
        self.sha_command[:, 2] = self._figure8_altitude

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(
                    self.cfg.goal_pos_visualizer_cfg
                )
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        robot_pos_w = self.robot.data.root_pos_w
        robot_pos_w[:, 2] += 0.1

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_ah_velocity_to_arrow(
            self.sha_command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_w[:, :2]
        )

        self.goal_pos_visualizer.visualize(self.pos_command_w, None)
        self.goal_vel_visualizer.visualize(
            robot_pos_w,
            vel_des_arrow_quat,
            vel_des_arrow_scale,
        )
        self.current_vel_visualizer.visualize(
            robot_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1)
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat

    def _resolve_ah_velocity_to_arrow(
        self, ah_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            ah_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= ah_velocity[:, 0]
        heading_angle = ah_velocity[:, 1]
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat
