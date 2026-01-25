# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import numpy as np
from isaaclab_contrib.assets import Multirotor

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim import build_simulation_context

# Pre-defined configs
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


@pytest.fixture
def sim():
    """Create simulation context."""
    with build_simulation_context(dt=1.0 / 120.0, device="cuda:0") as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.isaacsim_ci
def test_allocation_matrix_computation(sim):
    """Test that allocation matrix is computed correctly from USD file.
    
    This test verifies that the automatically computed allocation matrix
    matches the expected matrix from ARL_ROBOT_1_CFG.
    """
    # Create environment prims
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5
    
    # Create Top-level Xforms, one for each articulation
    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    
    # Create config with allocation_matrix=None to trigger automatic computation
    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    multirotor_cfg.allocation_matrix = None  # Force computation
    
    # Set dt for thrusters (required by ThrusterCfg)
    if "thrusters" in multirotor_cfg.actuators:
        multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)
    
    # Create Multirotor asset - this should compute the allocation matrix
    multirotor = Multirotor(multirotor_cfg)
    
    # Reset sim and asset
    sim.reset()
    multirotor.reset()
    
    # Get the computed allocation matrix
    computed_matrix = multirotor.allocation_matrix.cpu().numpy()
    
    # Get the expected allocation matrix from the original config
    expected_matrix = np.array(ARL_ROBOT_1_CFG.allocation_matrix, dtype=np.float32)
    
    # Verify shape
    assert computed_matrix.shape == expected_matrix.shape, (
        f"Computed matrix shape {computed_matrix.shape} does not match "
        f"expected shape {expected_matrix.shape}"
    )
    assert computed_matrix.shape == (6, 4), (
        f"Allocation matrix should be 6x4 (6 DOF, 4 thrusters), "
        f"but got shape {computed_matrix.shape}"
    )
    
    # Compare matrices with tolerance
    # Note: Small differences are expected due to floating point precision
    atol = 1e-5  # Absolute tolerance
    rtol = 1e-4  # Relative tolerance (1%)
    
    np.testing.assert_allclose(
        computed_matrix,
        expected_matrix,
        atol=atol,
        rtol=rtol,
        err_msg=(
            f"Computed allocation matrix does not match expected matrix.\n"
            f"Computed:\n{computed_matrix}\n"
            f"Expected:\n{expected_matrix}\n"
            f"Difference:\n{np.abs(computed_matrix - expected_matrix)}"
        ),
    )
    
    # Additional checks for specific values
    # Force contributions (rows 0-2) should be [0, 0, 1] for all thrusters
    # (all thrusters point upward in Z direction)
    assert np.allclose(computed_matrix[0, :], 0.0, atol=atol), "Fx should be 0 for all thrusters"
    assert np.allclose(computed_matrix[1, :], 0.0, atol=atol), "Fy should be 0 for all thrusters"
    assert np.allclose(computed_matrix[2, :], 1.0, atol=atol), "Fz should be 1.0 for all thrusters (upward)"
    
    # Torque contributions (rows 3-5) should match expected values
    # These depend on thruster positions relative to COM
    expected_torques = expected_matrix[3:, :]
    computed_torques = computed_matrix[3:, :]
    
    np.testing.assert_allclose(
        computed_torques,
        expected_torques,
        atol=atol,
        rtol=rtol,
        err_msg=(
            f"Computed torque contributions do not match expected values.\n"
            f"Computed torques:\n{computed_torques}\n"
            f"Expected torques:\n{expected_torques}"
        ),
    )

@pytest.mark.isaacsim_ci
def test_allocation_matrix_computation_with_explicit_config(sim):
    """Test that explicitly provided allocation matrix is not recomputed.
    
    This test verifies that if allocation_matrix is provided in config,
    it is not recomputed and is used as-is.
    """
    # Create environment prims
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5
    
    # Create Top-level Xforms
    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    
    # Create config with explicit allocation_matrix
    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    # Keep the original allocation_matrix (don't set to None)
    
    # Set dt for thrusters (required by ThrusterCfg)
    if "thrusters" in multirotor_cfg.actuators:
        multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)
    
    # Create Multirotor asset
    multirotor = Multirotor(multirotor_cfg)
    
    # Reset sim and asset
    sim.reset()
    multirotor.reset()
    
    # Get the allocation matrix
    computed_matrix = multirotor.allocation_matrix.cpu().numpy()
    expected_matrix = np.array(ARL_ROBOT_1_CFG.allocation_matrix, dtype=np.float32)
    
    # Should match exactly since it was provided explicitly
    np.testing.assert_array_equal(
        computed_matrix,
        expected_matrix,
        err_msg="Explicitly provided allocation matrix should be used as-is",
    )


@pytest.mark.isaacsim_ci
def test_allocation_matrix_computation_requires_rotor_directions(sim):
    """Test that allocation matrix computation requires rotor_directions.
    
    This test verifies that an error is raised if allocation_matrix is None
    but rotor_directions is also None.
    """
    # Create environment prims
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5
    
    # Create Top-level Xforms
    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    
    # Create config without allocation_matrix and without rotor_directions
    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    multirotor_cfg.allocation_matrix = None
    multirotor_cfg.rotor_directions = None  # This should cause an error
    
    # Set dt for thrusters (required by ThrusterCfg, even for error test)
    if "thrusters" in multirotor_cfg.actuators:
        multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)
    
    # Create Multirotor asset - should raise ValueError during _initialize_impl()
    # The error is raised in _compute_allocation_matrix() which is called in _initialize_impl()
    with pytest.raises(ValueError, match="rotor_directions must be provided"):
        multirotor = Multirotor(multirotor_cfg)
        # Force initialization to trigger the error
        sim.reset()
        multirotor.reset()