# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for drone ARL environments.

This sub-module contains the functions that can be used to enable different events
for drone training, such as applying disturbances to the multirotor vehicle.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_disturbance(
    env: ManagerBasedEnv,
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
    env: ManagerBasedEnv,
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
    
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
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
        self._disturbance_start_time = torch.full(
            (self.num_envs,), -1.0, device=self.device, dtype=torch.float32
        )
        
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
        env: ManagerBasedEnv,
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