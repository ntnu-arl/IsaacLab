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

import numpy as np
import pytest
import torch
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
    has the correct shape and properties (generic test for any multirotor).
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

    # Get number of thrusters dynamically
    num_thrusters = multirotor.num_thrusters

    # Verify shape - allocation matrix should be (6, num_thrusters)
    assert computed_matrix.shape == (6, num_thrusters), (
        f"Allocation matrix should be 6x{num_thrusters} (6 DOF, {num_thrusters} thrusters), "
        f"but got shape {computed_matrix.shape}"
    )

    # Tolerance for floating point comparisons
    atol = 1e-5  # Absolute tolerance

    # Verify force contributions (rows 0-2) are not all zero and are finite
    # For non-planar configurations, thrusters may not all point straight up
    computed_forces = computed_matrix[0:3, :]
    assert not np.allclose(computed_forces, 0.0, atol=atol), "Force contributions should not all be zero"
    assert np.all(np.isfinite(computed_forces)), "All force values should be finite"

    # Verify torque contributions (rows 3-5) are non-zero and have reasonable values
    # Torque values depend on thruster positions relative to COM
    computed_torques = computed_matrix[3:, :]
    assert not np.allclose(computed_torques, 0.0, atol=atol), "Torque contributions should not all be zero"
    assert np.all(np.isfinite(computed_torques)), "All torque values should be finite"


@pytest.mark.isaacsim_ci
def test_allocation_matrix_computation_with_explicit_config(sim):
    """Test that automatically computed allocation matrix matches expected for ARL_ROBOT_1.

    This test verifies that the automatically computed allocation matrix for ARL_ROBOT_1
    matches the expected specific values.
    """
    # Create environment prims
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5

    # Create Top-level Xforms
    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])

    # Expected allocation matrix for ARL_ROBOT_1
    expected_allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.1, -0.1, -0.1, 0.1],
        [-0.1, -0.1, 0.1, 0.1],
        [-0.07, 0.07, -0.07, 0.07],
    ]

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
    expected_matrix = np.array(expected_allocation_matrix, dtype=np.float32)

    # Get number of thrusters dynamically
    num_thrusters = multirotor.num_thrusters

    # Verify shape
    assert computed_matrix.shape == expected_matrix.shape, (
        f"Computed matrix shape {computed_matrix.shape} does not match expected shape {expected_matrix.shape}"
    )
    assert computed_matrix.shape == (6, num_thrusters), (
        f"Allocation matrix should be 6x{num_thrusters} (6 DOF, {num_thrusters} thrusters), "
        f"but got shape {computed_matrix.shape}"
    )

    # Tolerance for floating point comparisons
    atol = 1e-5  # Absolute tolerance
    rtol = 1e-4  # Relative tolerance

    # Compare computed matrix with expected matrix
    np.testing.assert_allclose(
        computed_matrix,
        expected_matrix,
        atol=atol,
        rtol=rtol,
        err_msg=(
            f"Computed allocation matrix does not match expected matrix for ARL_ROBOT_1.\n"
            f"Computed:\n{computed_matrix}\n"
            f"Expected:\n{expected_matrix}\n"
            f"Difference:\n{np.abs(computed_matrix - expected_matrix)}"
        ),
    )

    # Additional checks for specific values for ARL_ROBOT_1
    # Force contributions (rows 0-2) should be [0, 0, 1] for all thrusters
    # (all thrusters point upward in Z direction for ARL_ROBOT_1)
    assert np.allclose(computed_matrix[0, :], 0.0, atol=atol), "Fx should be 0 for all thrusters"
    assert np.allclose(computed_matrix[1, :], 0.0, atol=atol), "Fy should be 0 for all thrusters"
    assert np.allclose(computed_matrix[2, :], 1.0, atol=atol), "Fz should be 1.0 for all thrusters (upward)"

    # Verify torque contributions (rows 3-5) match expected values
    expected_torques = expected_matrix[3:, :]
    computed_torques = computed_matrix[3:, :]

    np.testing.assert_allclose(
        computed_torques,
        expected_torques,
        atol=atol,
        rtol=rtol,
        err_msg=(
            f"Computed torque contributions do not match expected values for ARL_ROBOT_1.\n"
            f"Computed torques:\n{computed_torques}\n"
            f"Expected torques:\n{expected_torques}"
        ),
    )
