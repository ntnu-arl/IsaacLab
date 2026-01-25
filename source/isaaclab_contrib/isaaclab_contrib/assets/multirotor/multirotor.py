# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import quat_apply_inverse, quat_rotate

from isaaclab_contrib.actuators import Thruster
from isaaclab_contrib.utils.types import MultiRotorActions

from .multirotor_data import MultirotorData

if TYPE_CHECKING:
    from .multirotor_cfg import MultirotorCfg

# import logger
logger = logging.getLogger(__name__)


class Multirotor(Articulation):
    """A multirotor articulation asset class.

    This class extends the base articulation class to support multirotor vehicles
    with thruster actuators that can be applied at specific body locations.
    """

    cfg: MultirotorCfg
    """Configuration instance for the multirotor."""

    actuators: dict[str, Thruster]
    """Dictionary of thruster actuator instances for the multirotor.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`MultirotorCfg.actuators`
    attribute. They are used to compute the thruster commands during the :meth:`write_data_to_sim` function.
    """

    def __init__(self, cfg: MultirotorCfg):
        """Initialize the multirotor articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def thruster_names(self) -> list[str]:
        """Ordered names of thrusters in the multirotor."""
        if not hasattr(self, "actuators") or not self.actuators:
            return []

        thruster_names = []
        for actuator in self.actuators.values():
            if hasattr(actuator, "thruster_names"):
                thruster_names.extend(actuator.thruster_names)
            else:
                raise ValueError("Non thruster actuator found in multirotor actuators. Not supported at the moment.")

        return thruster_names

    @property
    def num_thrusters(self) -> int:
        """Number of thrusters in the multirotor."""
        return len(self.thruster_names)

    @property
    def allocation_matrix(self) -> torch.Tensor:
        """Allocation matrix for control allocation."""
        if self.cfg.allocation_matrix is None:
            raise RuntimeError(
                "Allocation matrix is None. This should have been computed during initialization. "
                "Please check that _compute_allocation_matrix() was called."
            )
        return torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)

    """
    Operations
    """

    def set_thrust_target(
        self,
        target: torch.Tensor,
        thruster_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set target thrust values for thrusters.

        Args:
            target: Target thrust values. Shape is (num_envs, num_thrusters) or (num_envs,).
            thruster_ids: Indices of thrusters to set. Defaults to None (all thrusters).
            env_ids: Environment indices to set. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if thruster_ids is None:
            thruster_ids = slice(None)

        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and thruster_ids != slice(None):
            env_ids = env_ids[:, None]

        # set targets
        self._data.thrust_target[env_ids, thruster_ids] = target

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the multirotor to default state.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
        """
        # call parent reset
        super().reset(env_ids)

        # reset multirotor-specific data
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # reset thruster targets to default values
        if self._data.thrust_target is not None and self._data.default_thruster_rps is not None:
            self._data.thrust_target[env_ids] = self._data.default_thruster_rps[env_ids]

    def write_data_to_sim(self):
        """Write thrust and torque commands to the simulation.

        This method applies thruster forces and external disturbances (both impulse and continuous)
        using the wrench composer system. Thruster forces are added to the instantaneous composer
        each step, while external disturbances can be either instantaneous (impulses) or permanent
        (continuous forces like wind).

        The net wrench from the allocation matrix is computed at the full articulation COM.
        We apply it to the base_link, but at the full articulation COM position (in base_link frame).
        However, we need to subtract the extra torque that the wrench composer would add from the
        position offset, since the allocation matrix already accounts for all lever arms.
        """
        # Apply thruster model
        self._apply_actuator_model()
        # Combine individual thrusts into a wrench vector
        self._combine_thrusts()

        # The allocation matrix computes the net wrench at the FULL ARTICULATION COM.
        # Get full articulation COM position in base_link frame
        com_pos_b = self._compute_articulation_com()  # COM in base_link frame (3,)

        # Get base_link COM position in base_link frame (body index 0)
        # body_com_pos_b is the COM position of each body in its own body frame
        base_com_pos_b = self.data.body_com_pos_b[0, 0, :]  # (3,) - base_link COM in base_link frame

        # Compute offset from base_link COM to full articulation COM
        r_com_offset = com_pos_b - base_com_pos_b  # (3,)

        # The wrench from allocation matrix is [Fx, Fy, Fz, Tx, Ty, Tz] at full articulation COM
        # When we apply it at the full articulation COM position, the wrench composer will add:
        #   extra_torque = r_com_offset × F
        # But the allocation matrix already includes all lever arm effects, so we need to subtract this
        forces = self._internal_force_target_sim[:, 0, :]  # (num_instances, 3)
        torques = self._internal_torque_target_sim[:, 0, :]  # (num_instances, 3)

        # Compute extra torque that would be added by position offset
        # r_com_offset × F for each environment
        extra_torque = torch.cross(
            r_com_offset.unsqueeze(0).expand(self.num_instances, -1), forces, dim=1
        )  # (num_instances, 3)

        # Subtract the extra torque to avoid double-counting
        torques_corrected = torques - extra_torque  # (num_instances, 3)

        # Expand to (num_instances, 1, 3) for wrench composer
        positions_wp = wp.from_torch(com_pos_b.unsqueeze(0).expand(self.num_instances, 1, -1), dtype=wp.vec3f)

        # Apply the net wrench to base_link, at the full articulation COM position.
        # We've corrected the torque to account for the position offset.
        self._instantaneous_wrench_composer.add_forces_and_torques(
            env_ids=self._ALL_INDICES_WP,
            body_ids=wp.from_torch(torch.tensor([0], dtype=torch.int32, device=self.device), dtype=wp.int32),
            forces=wp.from_torch(forces.unsqueeze(1), dtype=wp.vec3f),
            torques=wp.from_torch(torques_corrected.unsqueeze(1), dtype=wp.vec3f),
            positions=positions_wp,  # Apply at full articulation COM position (in base_link frame)
            is_global=False,  # Positions are in base_link frame, not world frame
        )

        # Apply drag forces and torques
        self._apply_drag()

        # Apply all wrenches together
        if self.instantaneous_wrench_composer.active or self.permanent_wrench_composer.active:
            if self.instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self.instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self.permanent_wrench_composer.composed_force,
                    torques=self.permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES_WP,
                    env_ids=self._ALL_INDICES_WP,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self.instantaneous_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self.instantaneous_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self.permanent_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self.permanent_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
        # Reset instantaneous composer (thruster forces + impulses) permanent composer persists
        self.instantaneous_wrench_composer.reset()

    def _initialize_impl(self):
        """Initialize the multirotor implementation."""
        # call parent initialization
        super()._initialize_impl()

        # Replace data container with MultirotorData
        self._data = MultirotorData(self.root_physx_view, self.device)

        # Create thruster buffers with correct size (SINGLE PHASE)
        self._create_thruster_buffers()

        # Process thruster configuration
        self._process_thruster_cfg()

        # Compute allocation matrix if not provided
        if self.cfg.allocation_matrix is None:
            self._compute_allocation_matrix()

        # Process configuration
        self._process_cfg()

        # Update the robot data
        self.update(0.0)

        # Log multirotor information
        self._log_multirotor_info()

    def _count_thrusters_from_config(self) -> int:
        """Count total number of thrusters from actuator configuration.

        Returns:
            Total number of thrusters across all actuator groups.
        """
        total_thrusters = 0

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            if not hasattr(actuator_cfg, "thruster_names_expr"):
                continue

            # Use find_bodies to count thrusters for this actuator
            body_indices, thruster_names = self.find_bodies(actuator_cfg.thruster_names_expr)
            total_thrusters += len(body_indices)

        if total_thrusters == 0:
            raise ValueError(
                "No thrusters found in actuator configuration. "
                "Please check thruster_names_expr in your MultirotorCfg.actuators."
            )

        return total_thrusters

    def _create_thruster_buffers(self):
        """Create thruster buffers with correct size."""
        num_instances = self.num_instances
        num_thrusters = self._count_thrusters_from_config()

        # Create thruster data tensors with correct size
        self._data.default_thruster_rps = torch.zeros(num_instances, num_thrusters, device=self.device)
        # thrust after controller and allocation is applied
        self._data.thrust_target = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.computed_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.applied_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)

        # Combined wrench buffers
        self._thrust_target_sim = torch.zeros_like(self._data.thrust_target)  # thrust after actuator model is applied
        # wrench target for combined mode
        self._internal_wrench_target_sim = torch.zeros(num_instances, 6, device=self.device)
        # internal force/torque targets per body for combined mode
        self._internal_force_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)
        self._internal_torque_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)

        # Placeholder thruster names (will be filled during actuator creation)
        self._data.thruster_names = [f"thruster_{i}" for i in range(num_thrusters)]

    def _process_actuators_cfg(self):
        """Override parent method to do nothing - we handle thrusters separately."""
        # Do nothing - we handle thruster processing in _process_thruster_cfg() otherwise this
        # gives issues with joint name expressions
        pass

    def _process_cfg(self):
        """Post processing of multirotor configuration parameters."""
        # Handle root state (like parent does)
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

        # Handle thruster-specific initial state
        if hasattr(self._data, "default_thruster_rps") and hasattr(self.cfg.init_state, "rps"):
            # Match against thruster names
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.init_state.rps, self.thruster_names
            )
            if indices_list:
                rps_values = torch.tensor(values_list, device=self.device)
                self._data.default_thruster_rps[:, indices_list] = rps_values
                self._data.thrust_target[:, indices_list] = rps_values

    def _process_thruster_cfg(self):
        """Process and apply multirotor thruster properties."""
        # create actuators
        self.actuators = dict()
        self._has_implicit_actuators = False

        # Check for mixed configurations (same as before)
        has_thrusters = False
        has_joints = False

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            if hasattr(actuator_cfg, "thruster_names_expr"):
                has_thrusters = True
            elif hasattr(actuator_cfg, "joint_names_expr"):
                has_joints = True

        if has_thrusters and has_joints:
            raise ValueError("Mixed configurations with both thrusters and regular joints are not supported.")

        if has_joints:
            raise ValueError("Regular joint actuators are not supported in Multirotor class.")

        # Store the body-to-thruster mapping
        self._thruster_body_mapping = {}

        # Track thruster names as we create actuators
        all_thruster_names = []

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            body_indices, thruster_names = self.find_bodies(actuator_cfg.thruster_names_expr)

            # Create 0-based thruster array indices starting from current count
            start_idx = len(all_thruster_names)
            thruster_array_indices = list(range(start_idx, start_idx + len(body_indices)))

            # Track all thruster names
            all_thruster_names.extend(thruster_names)

            # Store the mapping
            self._thruster_body_mapping[actuator_name] = {
                "body_indices": body_indices,
                "array_indices": thruster_array_indices,
                "thruster_names": thruster_names,
            }

            # Create thruster actuator
            actuator: Thruster = actuator_cfg.class_type(
                cfg=actuator_cfg,
                thruster_names=thruster_names,
                thruster_ids=thruster_array_indices,
                num_envs=self.num_instances,
                device=self.device,
                init_thruster_rps=self._data.default_thruster_rps[:, thruster_array_indices],
            )

            # Store actuator
            self.actuators[actuator_name] = actuator

            # Log information
            logger.info(
                f"Thruster actuator: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (thruster names: {thruster_names} [{body_indices}])."
            )

        # Update thruster names in data container
        self._data.thruster_names = all_thruster_names

        # Log summary
        logger.info(f"Initialized {len(self.actuators)} thruster actuator(s) for multirotor.")

    def _compute_allocation_matrix(self):
        """Compute allocation matrix from USD file and robot configuration.

        The allocation matrix maps thruster forces to body forces and torques.
        It's a 6xN matrix where:
        - Rows 0-2: Force contributions (Fx, Fy, Fz)
        - Rows 3-5: Torque contributions (Tx, Ty, Tz)
        - Columns: One per thruster

        This method computes the matrix based on:
        - Thruster positions relative to CENTER OF MASS (computed from all bodies)
        - Thrust direction (assumed to be +Z axis of thruster body for upward thrust)
        - Rotor directions (CW/CCW) for torque contribution

        The thrust direction is defined as upward (+Z) because:
        - Airflow direction points downwards
        - The reacting force (thrust) on the system points upwards
        """
        # Get configuration
        if self.cfg.rotor_directions is None:
            raise ValueError(
                "Cannot compute allocation matrix: rotor_directions must be provided "
                "in config when allocation_matrix is not specified."
            )

        rotor_directions = self.cfg.rotor_directions

        # Get torque constant (cq) from thruster config
        # This is the ratio of rotor drag torque to thrust
        cq = 0.0
        if "thrusters" in self.cfg.actuators:
            cq = self.cfg.actuators["thrusters"].torque_to_thrust_ratio

        # Get thruster names in order
        thruster_names = self.thruster_names
        num_thrusters = len(thruster_names)

        if len(rotor_directions) != num_thrusters:
            raise ValueError(
                f"Cannot compute allocation matrix: number of rotor_directions ({len(rotor_directions)}) "
                f"must match number of thrusters ({num_thrusters})"
            )

        # Get base_link body index
        base_link_ids, _ = self.find_bodies("base_link", preserve_order=True)
        if len(base_link_ids) == 0:
            raise ValueError("Cannot compute allocation matrix: could not find 'base_link' in articulation")
        base_link_id = base_link_ids[0]

        # Get body poses in world frame (after initialization)
        # Update to get initial poses
        self.update(dt=0.0)

        # Get base_link pose in world frame
        base_pos_w = self.data.body_link_pos_w[0, base_link_id, :].clone()
        base_quat_w = self.data.body_link_quat_w[0, base_link_id, :].clone()

        # Compute full articulation COM
        com_pos_b = self._compute_articulation_com()
        logger.info(f"Computed full articulation COM in base_link frame: {com_pos_b}")

        # Initialize allocation matrix
        allocation_matrix = torch.zeros(6, num_thrusters, device=self.device)

        # For each thruster, find its body and compute contribution
        for i, thruster_name in enumerate(thruster_names):
            # Find the body that contains this thruster
            # Thruster names typically match body names (e.g., "back_left_prop")
            thruster_body_ids, thruster_body_names = self.find_bodies(thruster_name, preserve_order=True)

            if len(thruster_body_ids) == 0:
                # Try to find by pattern matching
                thruster_body_ids, thruster_body_names = self.find_bodies(f".*{thruster_name}.*", preserve_order=True)

            if len(thruster_body_ids) == 0:
                raise ValueError(
                    f"Cannot compute allocation matrix: could not find body for thruster '{thruster_name}'"
                )

            thruster_body_id = thruster_body_ids[0]

            # Get thruster body position and orientation in world frame
            # Use body COM position (not link position) for consistency
            thruster_com_pos_w = self.data.body_com_pos_w[0, thruster_body_id, :3].clone()
            thruster_quat_w = self.data.body_link_quat_w[0, thruster_body_id, :].clone()

            # Transform to body frame (relative to base_link first, then adjust for COM)
            thruster_com_pos_rel_w = thruster_com_pos_w - base_pos_w
            thruster_com_pos_b = quat_apply_inverse(base_quat_w, thruster_com_pos_rel_w)

            # CRITICAL: Position relative to full articulation COM (not base_link origin)
            thruster_pos_com_b = thruster_com_pos_b - com_pos_b

            # Thrust direction: +Z axis in thruster body frame (upward thrust)
            # Airflow points down, but reaction force (thrust) on system points up
            thrust_dir_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # +Z in thruster frame
            thrust_dir_w = quat_rotate(thruster_quat_w, thrust_dir_local)
            thrust_dir_b = quat_apply_inverse(base_quat_w, thrust_dir_w)

            # Normalize thrust direction
            thrust_dir_b = thrust_dir_b / torch.norm(thrust_dir_b)

            # Force contribution (rows 0-2): thrust direction in body frame
            allocation_matrix[0:3, i] = thrust_dir_b

            # Torque contribution (rows 3-5): r × F - alpha * cq * F
            # r is position vector from COM to thruster (in body frame)
            r = thruster_pos_com_b  # Position relative to COM

            # Torque from force: r × F
            # Use explicit dim argument to avoid deprecation warning
            torque_from_force = torch.linalg.cross(r, thrust_dir_b)

            # Rotor drag torque: -alpha * cq * F
            # Negative sign because drag opposes motion
            alpha_i = float(rotor_directions[i])  # 1 for CCW, -1 for CW
            rotor_torque = -alpha_i * cq * thrust_dir_b

            # Total torque
            allocation_matrix[3:6, i] = torque_from_force + rotor_torque

        # Convert to list format and set in config
        allocation_matrix_list = allocation_matrix.cpu().numpy().tolist()
        self.cfg.allocation_matrix = allocation_matrix_list

        logger.info(f"Computed allocation matrix from USD file: {num_thrusters} thrusters")

    def _compute_articulation_com(self) -> torch.Tensor:
        """Compute the center of mass of the entire articulation.

        Computes the weighted average COM from all bodies in the articulation,
        where each body's contribution is weighted by its mass.

        Returns:
            torch.Tensor: COM position in base_link frame. Shape is (3,).
        """
        # Get base_link body index
        base_link_ids, _ = self.find_bodies("base_link", preserve_order=True)
        if len(base_link_ids) == 0:
            raise ValueError("Cannot compute articulation COM: could not find 'base_link' in articulation")
        base_link_id = base_link_ids[0]

        # Get base_link pose in world frame
        base_pos_w = self.data.body_link_pos_w[0, base_link_id, :].clone()
        base_quat_w = self.data.body_link_quat_w[0, base_link_id, :].clone()

        # Get all body COM positions in world frame
        body_com_pos_w = self.data.body_com_pos_w[0, :, :3].clone()  # Shape: (num_bodies, 3)

        # Get all body masses and ensure they're on the correct device
        body_masses = self.root_physx_view.get_masses()[0, :].clone().to(self.device)  # Shape: (num_bodies,)

        # Compute weighted average COM: COM_total = sum(mass_i * COM_i) / sum(mass_i)
        total_mass = torch.sum(body_masses)
        if total_mass < 1e-6:
            raise ValueError("Cannot compute articulation COM: total mass is too small or zero")

        com_pos_w = torch.sum(body_masses[:, None] * body_com_pos_w, dim=0) / total_mass  # Shape: (3,)

        # Transform COM to base_link frame
        com_pos_rel_w = com_pos_w - base_pos_w
        com_pos_b = quat_apply_inverse(base_quat_w, com_pos_rel_w)

        return com_pos_b

    def _apply_actuator_model(self):
        """Processes thruster commands for the multirotor by forwarding them to the actuators.

        The actions are first processed using actuator models. The thruster actuator models
        compute the thruster level simulation commands and sets them into the PhysX buffers.
        """

        # process thruster actions per group
        for actuator in self.actuators.values():
            if not isinstance(actuator, Thruster):
                continue

            # prepare input for actuator model based on cached data
            control_action = MultiRotorActions(
                thrusts=self._data.thrust_target[:, actuator.thruster_indices],
                thruster_indices=actuator.thruster_indices,
            )

            # compute thruster command from the actuator model
            control_action = actuator.compute(control_action)

            # update targets (these are set into the simulation)
            if control_action.thrusts is not None:
                self._thrust_target_sim[:, actuator.thruster_indices] = control_action.thrusts

            # update state of the actuator model
            self._data.computed_thrust[:, actuator.thruster_indices] = actuator.computed_thrust
            self._data.applied_thrust[:, actuator.thruster_indices] = actuator.applied_thrust

    def _apply_drag(self):
        """Apply aerodynamic drag forces and torques to the base link at center of mass.

        The drag model follows:
        - Linear drag: F = -C_lin * v - C_quad * |v| * v (world-frame)
        - Angular drag: T = -C_lin * w - C_quad * |w| * w (body-frame)
        """
        # Get world-frame velocities for base link (body index 0)
        v = self.data.body_lin_vel_w[:, 0, :]
        w_world = self.data.body_ang_vel_w[:, 0, :]

        # Convert angular velocity from world to body frame
        quat_w = self.data.body_quat_w[:, 0, :]
        w = quat_apply_inverse(quat_w, w_world)

        # Compute linear drag in world frame
        lin_drag = -self.cfg.lin_drag_linear_coef * v
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        lin_drag = lin_drag - self.cfg.lin_drag_quadratic_coef * v_norm * v

        # Compute angular drag in body frame
        ang_drag = -self.cfg.ang_drag_linear_coef * w
        w_norm = torch.linalg.vector_norm(w, dim=-1, keepdim=True)
        ang_drag = ang_drag - self.cfg.ang_drag_quadratic_coef * w_norm * w

        # Reshape for wrench composer
        forces = lin_drag.unsqueeze(1)  # (num_envs, 1, 3) - world-frame
        torques = ang_drag.unsqueeze(1)  # (num_envs, 1, 3) - body-frame

        # Get COM positions for base_link (body index 0)
        base_com_positions = self.data.body_com_pos_w[:, 0:1, :3].clone()  # (num_envs, 1, 3)
        positions_wp = wp.from_torch(base_com_positions, dtype=wp.vec3f)

        # Add drag to instantaneous wrench composer (applied at COM)
        self.instantaneous_wrench_composer.add_forces_and_torques(
            env_ids=self._ALL_INDICES_WP,
            body_ids=wp.from_torch(torch.tensor([0], dtype=torch.int32, device=self.device), dtype=wp.int32),
            forces=wp.from_torch(forces, dtype=wp.vec3f),
            torques=wp.from_torch(torques, dtype=wp.vec3f),
            positions=positions_wp,
            is_global=True,  # Forces are in world frame
        )

    def _prepare_wrench_arrays(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int],
        env_ids: torch.Tensor | None = None,
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
        """Prepare Warp arrays for wrench application at center of mass.

        Args:
            forces: External forces. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            torques: External torques. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            body_ids: Body indices to apply wrench to (list of ints).
            env_ids: Environment indices to apply wrench to. Defaults to None (all environments).

        Returns:
            Tuple of (env_ids_wp, body_ids_wp, forces_wp, torques_wp, positions_wp) as Warp arrays.
            positions_wp contains the COM positions of the bodies in world frame.
        """
        # Resolve env_ids
        if env_ids is None:
            env_ids_wp = self._ALL_INDICES_WP
            num_envs = self.num_instances
        else:
            env_ids_wp = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            num_envs = len(env_ids)

        # Convert body_ids to 1D tensor (not expanded)
        body_ids_tensor = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
        num_bodies = len(body_ids)
        body_ids_wp = wp.from_torch(body_ids_tensor, dtype=wp.int32)  # 1D array

        # Expand forces and torques to (num_envs, num_bodies, 3) if needed
        if forces.dim() == 2:  # (num_envs, 3)
            forces = forces.unsqueeze(1).expand(-1, num_bodies, -1)
        elif forces.shape[1] == 1:  # (num_envs, 1, 3)
            forces = forces.expand(-1, num_bodies, -1)

        if torques.dim() == 2:  # (num_envs, 3)
            torques = torques.unsqueeze(1).expand(-1, num_bodies, -1)
        elif torques.shape[1] == 1:  # (num_envs, 1, 3)
            torques = torques.expand(-1, num_bodies, -1)

        forces_wp = wp.from_torch(forces, dtype=wp.vec3f)
        torques_wp = wp.from_torch(torques, dtype=wp.vec3f)

        # Get COM positions for the specified bodies in world frame
        # body_com_pos_w shape: (num_instances, num_bodies, 3)
        if env_ids is None:
            body_com_positions = self.data.body_com_pos_w[:, body_ids, :3].clone()  # (num_instances, num_bodies, 3)
        else:
            body_com_positions = self.data.body_com_pos_w[env_ids, :, :3][
                :, body_ids, :
            ].clone()  # (num_envs, num_bodies, 3)

        positions_wp = wp.from_torch(body_com_positions, dtype=wp.vec3f)

        return env_ids_wp, body_ids_wp, forces_wp, torques_wp, positions_wp

    def add_instantaneous_disturbance(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int],
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Add instantaneous external disturbance (impulse) to the multirotor at center of mass.

        This convenience method handles Warp array conversion internally. The wrench
        is applied for one simulation step and then automatically cleared.
        Forces are applied at the center of mass of each body.

        Args:
            forces: External forces. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            torques: External torques. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            body_ids: Body indices to apply wrench to (list of ints).
            env_ids: Environment indices to apply wrench to. Defaults to None (all environments).
            is_global: Whether forces/torques are in global frame. Defaults to False (body frame).
        """
        env_ids_wp, body_ids_wp, forces_wp, torques_wp, positions_wp = self._prepare_wrench_arrays(
            forces, torques, body_ids, env_ids
        )

        # Add to instantaneous wrench composer (forces applied at COM positions)
        self.instantaneous_wrench_composer.add_forces_and_torques(
            env_ids=env_ids_wp,
            body_ids=body_ids_wp,
            forces=forces_wp,
            torques=torques_wp,
            positions=positions_wp,
            is_global=is_global,
        )

    def set_permanent_disturbance(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int],
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Set permanent external disturbance (continuous) on the multirotor at center of mass.

        This convenience method handles Warp array conversion internally. The wrench
        persists until explicitly cleared via the permanent_wrench_composer.reset() method.
        Forces are applied at the center of mass of each body.

        Args:
            forces: External forces. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            torques: External torques. Shape is (len(env_ids), len(body_ids), 3) or (len(env_ids), 3).
            body_ids: Body indices to apply wrench to (list of ints).
            env_ids: Environment indices to apply wrench to. Defaults to None (all environments).
            is_global: Whether forces/torques are in global frame. Defaults to False (body frame).
        """
        env_ids_wp, body_ids_wp, forces_wp, torques_wp, positions_wp = self._prepare_wrench_arrays(
            forces, torques, body_ids, env_ids
        )

        # Set to permanent wrench composer (forces applied at COM positions)
        self.permanent_wrench_composer.set_forces_and_torques(
            env_ids=env_ids_wp,
            body_ids=body_ids_wp,
            forces=forces_wp,
            torques=torques_wp,
            positions=positions_wp,
            is_global=is_global,
        )

    def _combine_thrusts(self):
        """Combine individual thrusts into a wrench vector."""
        thrusts = self._thrust_target_sim
        self._internal_wrench_target_sim = (self.allocation_matrix @ thrusts.T).T
        # Apply forces to base link (body index 0) only
        self._internal_force_target_sim[:, 0, :] = self._internal_wrench_target_sim[:, :3]
        self._internal_torque_target_sim[:, 0, :] = self._internal_wrench_target_sim[:, 3:]

    def _validate_cfg(self):
        """Validate the multirotor configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
        """
        # Only validate if actuators have been created
        if hasattr(self, "actuators") and self.actuators:
            # Validate thruster-specific configuration
            for actuator_name in self.actuators:
                if isinstance(self.actuators[actuator_name], Thruster):
                    initial_thrust = self.actuators[actuator_name].curr_thrust
                    # check that the initial thrust is within the limits
                    thrust_limits = self.actuators[actuator_name].cfg.thrust_range
                    if torch.any(initial_thrust < thrust_limits[0]) or torch.any(initial_thrust > thrust_limits[1]):
                        raise ValueError(
                            f"Initial thrust for actuator '{actuator_name}' is out of bounds: "
                            f"{initial_thrust} not in {thrust_limits}"
                        )

    def _log_multirotor_info(self):
        """Log multirotor-specific information."""
        logger.info(f"Multirotor initialized with {self.num_thrusters} thrusters")
        logger.info(f"Thruster names: {self.thruster_names}")
        logger.info(f"Thruster force direction: {self.cfg.thruster_force_direction}")
