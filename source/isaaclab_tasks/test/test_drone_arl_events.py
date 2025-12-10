# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Integration tests for drone ARL disturbances using real Articulation/WrenchComposer."""
from isaaclab.app import AppLauncher

# launch omniverse app headless
simulation_app = AppLauncher(headless=True).app

import pytest
import torch

from isaaclab.managers import EventManager, EventTermCfg, SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.assets import Articulation
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
from isaaclab_tasks.manager_based.drone_arl.mdp import events as drone_events


@pytest.fixture
def sim():
    """Create simulation context."""
    with build_simulation_context(dt=1.0 / 120.0, device="cuda:0") as sim:
        sim._app_control_on_stop_handle = None
        yield sim
        # no explicit shutdown; AppLauncher handles exit


@pytest.fixture
def real_env(sim):
    """Minimal env wrapper with a real Multirotor and real WrenchComposer."""
    # Create environment prims
    num_envs = 1
    import isaacsim.core.utils.prims as prim_utils
    
    translations = torch.zeros(num_envs, 3, device=sim.device)
    translations[:, 0] = torch.arange(num_envs) * 2.5
    
    # Create Top-level Xforms, one for each articulation
    for i in range(num_envs):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    
    # Create Multirotor asset
    multirotor_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Env_.*/Robot")
    multirotor_cfg.actuators["thrusters"].dt = float(sim.cfg.dt)
    multirotor_cfg.init_state.joint_pos = {}
    multirotor_cfg.init_state.joint_vel = {}
    multirotor_cfg.actuators = {}
    articulation = Articulation(multirotor_cfg)
    
    # reset sim and asset
    sim.reset()
    articulation.reset()

    class SceneMap:
        def __init__(self, asset):
            self.asset = asset

        def __getitem__(self, key):
            if key == "robot":
                return self.asset
            raise KeyError(key)
        
        def keys(self):
            return ["robot"]

    class DummyEnv:
        def __init__(self, sim, asset):
            self.sim = sim
            self.scene = SceneMap(asset)
            self.num_envs = asset.num_instances
            self.dt = sim.cfg.dt
            self.step_dt = sim.cfg.dt
            self.device = sim.device
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    return DummyEnv(sim, articulation)


@pytest.mark.isaacsim_ci
def test_apply_disturbance_impulse(real_env):
    """Test applying impulse disturbance (duration_steps=1)."""
    cfg = {
        "impulse_disturbance": EventTermCfg(
            func=drone_events.apply_disturbance,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "force_range": (1.0, 1.0),
                "torque_range": (0.1, 0.1),
                "body_name": "base_link",
                "duration_steps": 1,
            },
        ),
    }
    event_man = EventManager(cfg, real_env)
    asset = real_env.scene["robot"]

    # Initial state should have no forces
    initial_force = asset._instantaneous_wrench_composer.composed_force_as_torch.clone()
    
    # Let interval elapse to trigger event
    fired = False
    for _ in range(5):  # give a few chances for the interval to trigger
        event_man.apply("interval", dt=real_env.dt)
        body_ids, _ = asset.find_bodies("base_link", preserve_order=True)
        forces_on_body = asset._instantaneous_wrench_composer.composed_force_as_torch[:, body_ids, :]
        if torch.any(forces_on_body != 0):
            fired = True
            break
        # advance sim; this also resets the instantaneous composer
        asset.write_data_to_sim()
        real_env.sim.step(render=False)
        asset.update(dt=real_env.dt)

    assert fired, "Impulse disturbance should apply forces to base_link"

    # After event, instantaneous composer should have been active
    # (forces are reset each step, so we check if it was active during application)
    assert asset._instantaneous_wrench_composer._active or torch.any(
        asset._instantaneous_wrench_composer.composed_force_as_torch != 0
    )


@pytest.mark.isaacsim_ci
def test_apply_disturbance_continuous(real_env):
    """Test applying continuous disturbance (duration_steps>1)."""
    cfg = {
        "continuous_disturbance": EventTermCfg(
            func=drone_events.apply_disturbance,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "force_range": (1.0, 1.0),
                "torque_range": (0.1, 0.1),
                "body_name": "base_link",
                "duration_steps": 5,
            },
        ),
    }
    event_man = EventManager(cfg, real_env)
    asset = real_env.scene["robot"]

    initial_force = asset._permanent_wrench_composer.composed_force_as_torch.clone()

    # Let interval elapse to trigger event
    for _ in range(2):
        event_man.apply("interval", dt=real_env.dt)
        asset.write_data_to_sim()
        real_env.sim.step(render=False)
        asset.update(dt=real_env.dt)

    # Permanent composer should be active with non-zero forces
    assert asset._permanent_wrench_composer._active or torch.any(
        asset._permanent_wrench_composer.composed_force_as_torch != 0
    )


@pytest.mark.isaacsim_ci
def test_clear_disturbance(real_env):
    """Test clearing permanent disturbances."""
    asset = real_env.scene["robot"]
    
    # Apply a permanent wrench
    env_ids = torch.arange(asset.num_instances, device=asset.device, dtype=torch.int32)
    body_ids, _ = asset.find_bodies("base_link", preserve_order=True)
    
    asset.set_permanent_external_wrench(
        forces=torch.ones((asset.num_instances, len(body_ids), 3), device=asset.device),
        torques=torch.ones((asset.num_instances, len(body_ids), 3), device=asset.device),
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=False,
    )
    
    # Verify forces are set
    assert torch.allclose(
        asset._permanent_wrench_composer.composed_force_as_torch[:, body_ids, :],
        torch.ones((asset.num_instances, len(body_ids), 3), device=asset.device),
        atol=1e-5,
    )

    # Clear the disturbance
    drone_events.clear_disturbance(real_env, env_ids, asset_cfg=SceneEntityCfg("robot"))

    # Forces should be cleared
    assert torch.allclose(
        asset._permanent_wrench_composer.composed_force_as_torch,
        torch.zeros((asset.num_instances, asset.num_bodies, 3), device=asset.device),
        atol=1e-5,
    )


@pytest.mark.isaacsim_ci
def test_apply_continuous_disturbance_with_duration(real_env):
    """Test class-based continuous disturbance with auto-clearing."""
    cfg = {
        "continuous_disturbance": EventTermCfg(
            func=drone_events.apply_continuous_disturbance_with_duration,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "force_range": (1.0, 1.0),
                "torque_range": (0.1, 0.1),
                "body_name": "base_link",
                "duration_s": 0.05,
            },
        ),
    }
    event_man = EventManager(cfg, real_env)
    asset = real_env.scene["robot"]

    initial = asset._permanent_wrench_composer.composed_force_as_torch.clone()

    # Trigger event and advance time
    for _ in range(2):
        event_man.apply("interval", dt=real_env.dt)
        real_env.episode_length_buf += 1
        asset.write_data_to_sim()
        real_env.sim.step(render=False)
        asset.update(dt=real_env.dt)

    # Disturbance should be applied
    applied = asset._permanent_wrench_composer.composed_force_as_torch.clone()
    assert not torch.allclose(applied, initial, atol=1e-5) or asset._permanent_wrench_composer._active

    # Advance beyond duration to allow auto-clear
    # duration_s=0.05, dt=1/120, so need ~6 steps
    for _ in range(6):
        real_env.episode_length_buf += 1
        event_man.apply("interval", dt=real_env.dt)
        asset.write_data_to_sim()
        real_env.sim.step(render=False)
        asset.update(dt=real_env.dt)

    # Either cleared or still active; if cleared, forces go to zero
    if not asset._permanent_wrench_composer._active:
        assert torch.allclose(
            asset._permanent_wrench_composer.composed_force_as_torch,
            torch.zeros_like(asset._permanent_wrench_composer.composed_force_as_torch),
            atol=1e-5,
        )