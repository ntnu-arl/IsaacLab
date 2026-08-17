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

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim import build_simulation_context

from isaaclab_contrib.assets import Multirotor

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
    """Test that allocation matrix is computed correctly from USD file."""
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5

    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])

    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    multirotor_cfg.allocation_matrix = None

    if "thrusters" in multirotor_cfg.actuators:
        multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)

    multirotor = Multirotor(multirotor_cfg)

    sim.reset()
    multirotor.reset()

    computed_matrix = multirotor.allocation_matrix.cpu().numpy()
    num_thrusters = multirotor.num_thrusters

    assert computed_matrix.shape == (6, num_thrusters), (
        f"Allocation matrix should be 6x{num_thrusters} (6 DOF, {num_thrusters} thrusters), "
        f"but got shape {computed_matrix.shape}"
    )

    atol = 1e-5
    computed_forces = computed_matrix[0:3, :]
    assert not np.allclose(computed_forces, 0.0, atol=atol), "Force contributions should not all be zero"
    assert np.all(np.isfinite(computed_forces)), "All force values should be finite"

    computed_torques = computed_matrix[3:, :]
    assert not np.allclose(computed_torques, 0.0, atol=atol), "Torque contributions should not all be zero"
    assert np.all(np.isfinite(computed_torques)), "All torque values should be finite"


@pytest.mark.isaacsim_ci
def test_allocation_matrix_computation_with_explicit_config(sim):
    """Test auto-computed allocation matrix against ARL_ROBOT_1 geometry and rotor directions.

    Thruster order is ``back_left``, ``back_right``, ``front_left``, ``front_right``.
    Forces are +Z; roll/pitch come from 0.1 m moment arms about the articulation COM;
    yaw reaction torque is ``-rotor_direction * torque_to_thrust_ratio``.
    """
    num_envs = 1
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5

    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])

    cq = float(ARL_ROBOT_1_CFG.actuators["thrusters"].torque_to_thrust_ratio)
    rotor_directions = np.array(ARL_ROBOT_1_CFG.rotor_directions, dtype=np.float32)
    expected_allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.1, -0.1, 0.1, -0.1],
        [0.1, 0.1, -0.1, -0.1],
        (-rotor_directions * cq).tolist(),
    ]

    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    multirotor_cfg.allocation_matrix = None

    if "thrusters" in multirotor_cfg.actuators:
        multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)

    multirotor = Multirotor(multirotor_cfg)

    sim.reset()
    multirotor.reset()

    computed_matrix = multirotor.allocation_matrix.cpu().numpy()
    expected_matrix = np.array(expected_allocation_matrix, dtype=np.float32)
    num_thrusters = multirotor.num_thrusters

    assert computed_matrix.shape == expected_matrix.shape, (
        f"Computed matrix shape {computed_matrix.shape} does not match expected shape {expected_matrix.shape}"
    )
    assert computed_matrix.shape == (6, num_thrusters), (
        f"Allocation matrix should be 6x{num_thrusters} (6 DOF, {num_thrusters} thrusters), "
        f"but got shape {computed_matrix.shape}"
    )

    atol = 1e-5
    rtol = 1e-4

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

    assert np.allclose(computed_matrix[0, :], 0.0, atol=atol), "Fx should be 0 for all thrusters"
    assert np.allclose(computed_matrix[1, :], 0.0, atol=atol), "Fy should be 0 for all thrusters"
    assert np.allclose(computed_matrix[2, :], 1.0, atol=atol), "Fz should be 1.0 for all thrusters (upward)"
    np.testing.assert_allclose(
        computed_matrix[5, :],
        -rotor_directions * cq,
        atol=atol,
        rtol=rtol,
        err_msg="Yaw row should be -rotor_direction * torque_to_thrust_ratio",
    )
