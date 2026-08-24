# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary PhysX-to-Newton multirotor comparison tests for MR-04."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Everything below follows simulator launch."""

import inspect
import math
import types

import pytest
import torch
import warp as wp
from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import (
    Multirotor,
    MultirotorBase,
    MultirotorCfg,
    MultirotorData,
    MultirotorDataBase,
    MultirotorDataPhysx,
    MultirotorPhysx,
)


def import_newton_types():
    """Import Newton comparison types when the optional Newton package is installed."""
    pytest.importorskip("newton", reason="Newton is required for the temporary MR-04 backend comparison tests.")
    from isaaclab_newton.assets.articulation import Articulation as NewtonArticulation  # noqa: PLC0415

    from isaaclab_contrib.assets import MultirotorDataNewton, MultirotorNewton  # noqa: PLC0415

    return NewtonArticulation, MultirotorNewton, MultirotorDataNewton


def _make_uninitialized_newton_multirotor(multirotor_type):
    """Create a callback-safe Newton instance for isolated wrench-write tests."""
    multirotor = object.__new__(multirotor_type)
    multirotor._initialize_handle = None
    multirotor._invalidate_initialize_handle = None
    multirotor._prim_deletion_handle = None
    multirotor._debug_vis_handle = None
    multirotor._physics_ready_handle = None
    return multirotor


def test_default_multirotor_remains_physx_during_validation():
    """The compatibility API keeps existing configurations on PhysX during the comparison period."""
    assert Multirotor is MultirotorPhysx
    assert MultirotorData is MultirotorDataPhysx


def test_physx_type_implements_the_common_api():
    """The PhysX reference exposes the common multirotor method signatures."""
    assert issubclass(MultirotorPhysx, (MultirotorBase, PhysxArticulation))
    assert MultirotorPhysx.__backend_name__ == "physx"

    common_methods = (
        "set_thrust_target",
        "reset",
        "write_data_to_sim",
        "_apply_actuator_model",
        "_apply_combined_wrench_to_composer",
        "_apply_motor_wrenches_to_composer",
        "_apply_drag",
        "_combine_thrusts",
    )
    for method_name in common_methods:
        common_signature = inspect.signature(getattr(MultirotorBase, method_name))
        assert inspect.signature(getattr(MultirotorPhysx, method_name)) == common_signature


def test_newton_type_implements_the_common_api():
    """The Newton implementation exposes the same common multirotor method signatures."""
    NewtonArticulation, MultirotorNewton, _ = import_newton_types()
    assert issubclass(MultirotorNewton, (MultirotorBase, NewtonArticulation))
    assert MultirotorNewton.__backend_name__ == "newton"

    common_methods = (
        "set_thrust_target",
        "reset",
        "write_data_to_sim",
        "_apply_actuator_model",
        "_apply_combined_wrench_to_composer",
        "_apply_motor_wrenches_to_composer",
        "_apply_drag",
        "_combine_thrusts",
    )
    for method_name in common_methods:
        common_signature = inspect.signature(getattr(MultirotorBase, method_name))
        assert inspect.signature(getattr(MultirotorNewton, method_name)) == common_signature


def test_physx_data_type_implements_the_common_api():
    """The PhysX data container provides the common multirotor state fields."""
    assert issubclass(MultirotorDataPhysx, MultirotorDataBase)

    for field_name in MultirotorDataBase.__annotations__:
        assert hasattr(MultirotorDataPhysx, field_name)


def test_newton_data_type_implements_the_common_api():
    """The Newton data container provides the common multirotor state fields."""
    _, _, MultirotorDataNewton = import_newton_types()
    assert issubclass(MultirotorDataNewton, MultirotorDataBase)

    for field_name in MultirotorDataBase.__annotations__:
        assert hasattr(MultirotorDataNewton, field_name)


def test_newton_wrench_write_uses_bound_external_wrench_array(monkeypatch):
    """Newton rotates composed body-frame loads into its world-frame external-wrench array."""
    _, MultirotorNewton, _ = import_newton_types()
    import isaaclab_contrib.assets.multirotor.multirotor_newton as multirotor_newton_module  # noqa: PLC0415

    force_buffer = object()
    torque_buffer = object()
    link_quaternion_buffer = object()
    wrench_buffer = object()
    environment_mask = object()
    body_mask = object()
    calls = []

    class RecordingComposer:
        def __init__(self, active):
            self.active = active
            self.out_force_b = force_buffer
            self.out_torque_b = torque_buffer

        def add_raw_buffers_from(self, other):
            calls.append(("add", other))

        def compose_to_body_frame(self):
            calls.append(("compose",))

    instantaneous_composer = RecordingComposer(active=True)
    permanent_composer = RecordingComposer(active=False)
    multirotor = _make_uninitialized_newton_multirotor(MultirotorNewton)
    multirotor._instantaneous_wrench_composer = instantaneous_composer
    multirotor._permanent_wrench_composer = permanent_composer
    multirotor._data = types.SimpleNamespace(
        body_link_quat_w=types.SimpleNamespace(warp=link_quaternion_buffer),
        _sim_bind_body_external_wrench=wrench_buffer,
    )
    multirotor._ALL_ENV_MASK = environment_mask
    multirotor._ALL_BODY_MASK = body_mask
    multirotor._root_view = types.SimpleNamespace(count=2, link_count=5)
    multirotor._device = "test-device"

    def record_launch(kernel, *, dim, device, inputs):
        calls.append(("launch", kernel, dim, device, inputs))

    monkeypatch.setattr(multirotor_newton_module.wp, "launch", record_launch)

    multirotor._write_external_wrenches_to_sim()

    assert calls == [
        ("add", permanent_composer),
        ("compose",),
        (
            "launch",
            multirotor_newton_module._write_body_frame_wrench_to_newton,
            (2, 5),
            "test-device",
            [
                force_buffer,
                torque_buffer,
                link_quaternion_buffer,
                wrench_buffer,
                environment_mask,
                body_mask,
            ],
        ),
    ]


def test_newton_wrench_kernel_rotates_body_loads_to_world():
    """Newton receives rotated world-frame loads and leaves masked bodies untouched."""
    import isaaclab_contrib.assets.multirotor.multirotor_newton as multirotor_newton_module  # noqa: PLC0415

    device = wp.get_device()
    forces_b = wp.array([[(1.0, 0.0, 0.0), (2.0, 3.0, 4.0)]], dtype=wp.vec3f, device=device)
    torques_b = wp.array([[(0.0, 1.0, 0.0), (5.0, 6.0, 7.0)]], dtype=wp.vec3f, device=device)
    body_link_quat_w = wp.array(
        [[(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), (0.0, 0.0, 0.0, 1.0)]],
        dtype=wp.quatf,
        device=device,
    )
    wrench_w = wp.array(
        [[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (8.0, 9.0, 10.0, 11.0, 12.0, 13.0)]],
        dtype=wp.spatial_vectorf,
        device=device,
    )
    env_mask = wp.array([True], dtype=wp.bool, device=device)
    body_mask = wp.array([True, False], dtype=wp.bool, device=device)

    wp.launch(
        multirotor_newton_module._write_body_frame_wrench_to_newton,
        dim=(1, 2),
        inputs=[forces_b, torques_b, body_link_quat_w, wrench_w, env_mask, body_mask],
        device=device,
    )

    expected = torch.tensor([[[-0.0, 1.0, 0.0, -1.0, -0.0, 0.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]]])
    torch.testing.assert_close(torch.from_numpy(wrench_w.numpy()), expected, rtol=0.0, atol=1.0e-6)


def _create_rigid_multirotor_prim(prim_path: str, translation: tuple[float, float, float]) -> None:
    """Create a self-contained rigid multirotor articulation for backend comparisons."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(prim_path, "Xform", translation=translation)
    robot_prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    base_path = f"{prim_path}/base_link"
    base = UsdGeom.Cube.Define(stage, base_path)
    base.CreateSizeAttr(0.2)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _set_cube_mass_properties(base.GetPrim(), mass=1.0, size=0.2)

    motor_offsets = [(-0.1, -0.1, 0.0), (-0.1, 0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0)]
    UsdGeom.Scope.Define(stage, f"{prim_path}/joints")
    for index, offset in enumerate(motor_offsets):
        motor_path = f"{prim_path}/motor_{index}"
        motor = UsdGeom.Cube.Define(stage, motor_path)
        motor.CreateSizeAttr(0.04)
        motor.AddTranslateOp().Set(Gf.Vec3d(*offset))
        UsdPhysics.RigidBodyAPI.Apply(motor.GetPrim())
        _set_cube_mass_properties(motor.GetPrim(), mass=0.05, size=0.04)

        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/joints/motor_{index}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(motor_path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*offset))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))


def _set_cube_mass_properties(prim, mass: float, size: float) -> None:
    """Author solver-independent mass and inertia for a uniform cube."""
    from pxr import Gf, UsdPhysics  # noqa: PLC0415

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    diagonal_inertia = mass * size**2 / 6.0
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(diagonal_inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))


def _generate_rigid_multirotor_cfg(
    prim_path: str,
    dt: float,
    force_application_level: str = "root_link",
    initial_thrust: float = 0.0,
    motor_time_constant: float = 0.05,
    linear_drag: tuple[float, float] = (0.0, 0.0),
    angular_drag: tuple[float, float] = (0.0, 0.0),
) -> MultirotorCfg:
    """Create a deterministic multirotor configuration shared by both backends."""
    return MultirotorCfg(
        prim_path=prim_path,
        init_state=MultirotorCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rps={".*": math.sqrt(initial_thrust)}),
        actuators={
            "thrusters": ThrusterCfg(
                dt=dt,
                thrust_range=(0.0, 20.0),
                thrust_const_range=(1.0, 1.0),
                tau_inc_range=(motor_time_constant, motor_time_constant),
                tau_dec_range=(motor_time_constant, motor_time_constant),
                torque_to_thrust_ratio=0.02,
                use_rps=False,
                integration_scheme="euler",
                thruster_names_expr=["motor_0", "motor_1", "motor_2", "motor_3"],
            )
        },
        allocation_matrix=None,
        rotor_directions=[1, -1, -1, 1],
        force_application_level=force_application_level,
        lin_drag_linear_coef=linear_drag[0],
        lin_drag_quadratic_coef=linear_drag[1],
        ang_drag_linear_coef=angular_drag[0],
        ang_drag_quadratic_coef=angular_drag[1],
    )


def _run_rigid_trajectory(
    multirotor_type,
    sim_cfg: SimulationCfg,
    thrust_target: list[float],
    command_steps: int,
    coast_steps: int = 0,
    force_application_level: str = "root_link",
    initial_thrust: float = 0.0,
    motor_time_constant: float = 0.05,
    initial_velocity: tuple[float, float, float, float, float, float] | None = None,
    linear_drag: tuple[float, float] = (0.0, 0.0),
    angular_drag: tuple[float, float] = (0.0, 0.0),
) -> dict[str, torch.Tensor]:
    """Run a deterministic rigid-airframe trajectory for one backend."""
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        _create_rigid_multirotor_prim("/World/Robot", (0.0, 0.0, 1.0))
        sim_utils.update_stage()
        multirotor = multirotor_type(
            _generate_rigid_multirotor_cfg(
                "/World/Robot",
                float(sim.cfg.dt),
                force_application_level=force_application_level,
                initial_thrust=initial_thrust,
                motor_time_constant=motor_time_constant,
                linear_drag=linear_drag,
                angular_drag=angular_drag,
            )
        )
        sim.reset()

        root_velocity = torch.zeros((1, 6), device=sim.device)
        if initial_velocity is not None:
            root_velocity[:] = torch.tensor(initial_velocity, device=sim.device)
        multirotor.write_root_link_velocity_to_sim_index(root_velocity=root_velocity)
        multirotor.update(0.0)

        initial_position = multirotor.data.root_link_pos_w.torch.clone()
        initial_quaternion = multirotor.data.root_link_quat_w.torch.clone()
        initial_root_velocity = multirotor.data.root_link_vel_w.torch.clone()
        multirotor.set_thrust_target(torch.tensor([thrust_target], device=sim.device))
        for _ in range(command_steps):
            multirotor.write_data_to_sim()
            sim.step(render=False)
            multirotor.update(sim.cfg.dt)

        if coast_steps > 0:
            multirotor.set_thrust_target(torch.zeros((1, 4), device=sim.device))
            for _ in range(coast_steps):
                multirotor.write_data_to_sim()
                sim.step(render=False)
                multirotor.update(sim.cfg.dt)

        return {
            "displacement": (multirotor.data.root_link_pos_w.torch - initial_position).detach().cpu().clone(),
            "initial_quaternion": initial_quaternion.detach().cpu().clone(),
            "initial_velocity": initial_root_velocity.detach().cpu().clone(),
            "quaternion": multirotor.data.root_link_quat_w.torch.detach().cpu().clone(),
            "velocity": multirotor.data.root_link_vel_w.torch.detach().cpu().clone(),
            "applied_thrust": multirotor.data.applied_thrust.detach().cpu().clone(),
            "body_mass": multirotor.data.body_mass.torch.detach().cpu().clone(),
            "body_inertia": multirotor.data.body_inertia.torch.detach().cpu().clone(),
        }


def _create_jointed_multirotor_prim(prim_path: str) -> None:
    """Create a floating base with one motor body connected by an unactuated revolute joint."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(prim_path, "Xform", translation=(0.0, 0.0, 1.0))
    robot_prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    base_path = f"{prim_path}/base_link"
    base = UsdGeom.Cube.Define(stage, base_path)
    base.CreateSizeAttr(0.2)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _set_cube_mass_properties(base.GetPrim(), mass=1.0, size=0.2)

    motor_path = f"{prim_path}/motor_0"
    motor = UsdGeom.Cube.Define(stage, motor_path)
    motor.CreateSizeAttr(0.08)
    motor.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(motor.GetPrim())
    _set_cube_mass_properties(motor.GetPrim(), mass=0.1, size=0.08)

    UsdGeom.Scope.Define(stage, f"{prim_path}/joints")
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joints/motor_hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(motor_path)])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.25, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)


def _generate_jointed_multirotor_cfg(prim_path: str, dt: float, force_application_level: str) -> MultirotorCfg:
    """Create the deterministic configuration for the jointed-airframe comparison."""
    return MultirotorCfg(
        prim_path=prim_path,
        init_state=MultirotorCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rps={"motor_0": 0.0}),
        actuators={
            "thrusters": ThrusterCfg(
                dt=dt,
                thrust_range=(0.0, 5.0),
                thrust_const_range=(1.0, 1.0),
                tau_inc_range=(0.0, 0.0),
                tau_dec_range=(0.0, 0.0),
                torque_to_thrust_ratio=0.0,
                use_rps=False,
                integration_scheme="euler",
                thruster_names_expr=["motor_0"],
            )
        },
        allocation_matrix=None,
        rotor_directions=[1],
        force_application_level=force_application_level,
    )


def _run_jointed_trajectory(
    multirotor_type, sim_cfg: SimulationCfg, force_application_level: str
) -> dict[str, torch.Tensor]:
    """Run the jointed-airframe load-placement scenario for one backend."""
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        _create_jointed_multirotor_prim("/World/Robot")
        sim_utils.update_stage()
        multirotor = multirotor_type(
            _generate_jointed_multirotor_cfg("/World/Robot", float(sim.cfg.dt), force_application_level)
        )
        sim.reset()

        multirotor.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=sim.device))
        multirotor.write_joint_velocity_to_sim_index(velocity=torch.zeros((1, 1), device=sim.device))
        multirotor.update(0.0)

        initial_joint_position = multirotor.data.joint_pos.torch.clone()
        initial_root_position = multirotor.data.root_link_pos_w.torch.clone()
        multirotor.set_thrust_target(torch.tensor([[0.5]], device=sim.device))
        for _ in range(20):
            multirotor.write_data_to_sim()
            sim.step(render=False)
            multirotor.update(sim.cfg.dt)

        return {
            "joint_displacement": (multirotor.data.joint_pos.torch - initial_joint_position).detach().cpu().clone(),
            "joint_velocity": multirotor.data.joint_vel.torch.detach().cpu().clone(),
            "root_displacement": (multirotor.data.root_link_pos_w.torch - initial_root_position).detach().cpu().clone(),
            "root_velocity": multirotor.data.root_link_vel_w.torch.detach().cpu().clone(),
        }


def _run_reset_lifecycle(multirotor_type, sim_cfg: SimulationCfg) -> dict[str, torch.Tensor]:
    """Exercise construction, wrench writing, and partial-environment reset for one backend."""
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        for env_index in range(2):
            sim_utils.create_prim(f"/World/Env_{env_index}", "Xform")
            _create_rigid_multirotor_prim(f"/World/Env_{env_index}/Robot", (2.0 * env_index, 0.0, 1.0))
        sim_utils.update_stage()
        multirotor = multirotor_type(
            _generate_rigid_multirotor_cfg(
                "/World/Env_.*/Robot", float(sim.cfg.dt), initial_thrust=1.0, motor_time_constant=0.0
            )
        )
        sim.reset()

        default_target = multirotor.data.thrust_target.clone()
        multirotor.set_thrust_target(torch.tensor([[4.0] * 4, [5.0] * 4], device=sim.device))
        multirotor.write_data_to_sim()
        commanded_thrust = multirotor.data.applied_thrust.clone()

        multirotor.reset(torch.tensor([1], device=sim.device))
        multirotor.write_data_to_sim()
        return {
            "default_target": default_target.detach().cpu().clone(),
            "commanded_thrust": commanded_thrust.detach().cpu().clone(),
            "reset_target": multirotor.data.thrust_target.detach().cpu().clone(),
            "reset_actuator_state": multirotor.actuators["thrusters"].curr_thrust.detach().cpu().clone(),
            "reset_applied_thrust": multirotor.data.applied_thrust.detach().cpu().clone(),
        }


def _backend_cases(gravity: tuple[float, float, float]) -> dict[str, tuple[type, SimulationCfg]]:
    """Create equivalent PhysX and Newton backend cases using the default device."""
    _, MultirotorNewton, _ = import_newton_types()
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: PLC0415

    dt = 1.0 / 120.0
    return {
        "physx": (MultirotorPhysx, SimulationCfg(dt=dt, gravity=gravity)),
        "newton": (
            MultirotorNewton,
            SimulationCfg(dt=dt, gravity=gravity, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        ),
    }


def _assert_rigid_trajectory_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    orientation_atol: float,
) -> None:
    """Compare backend-independent rigid-body trajectory observables."""
    torch.testing.assert_close(actual["displacement"], expected["displacement"], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual["velocity"], expected["velocity"], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual["body_mass"], expected["body_mass"])
    torch.testing.assert_close(actual["body_inertia"], expected["body_inertia"])
    torch.testing.assert_close(actual["applied_thrust"], expected["applied_thrust"])
    orientation_error = math_utils.quat_error_magnitude(actual["quaternion"], expected["quaternion"])
    assert torch.max(orientation_error) < orientation_atol


@pytest.mark.isaacsim_ci
def test_newton_matches_physx_deterministic_hover():
    """Both backends hold a rigid vehicle near hover from an initialized steady motor state."""
    cases = _backend_cases(gravity=(0.0, 0.0, -9.81))
    hover_thrust = 1.2 * 9.81 / 4.0
    results = {
        name: _run_rigid_trajectory(
            multirotor_type,
            sim_cfg,
            thrust_target=[hover_thrust] * 4,
            command_steps=120,
            initial_thrust=hover_thrust,
        )
        for name, (multirotor_type, sim_cfg) in cases.items()
    }

    for result in results.values():
        assert torch.linalg.vector_norm(result["displacement"]) < 1.0e-1
        assert torch.linalg.vector_norm(result["velocity"]) < 1.0e-1
        orientation_error = math_utils.quat_error_magnitude(result["quaternion"], result["initial_quaternion"])
        assert torch.max(orientation_error) < 1.0e-2
    _assert_rigid_trajectory_close(results["newton"], results["physx"], 5.0e-2, 5.0e-3, 1.0e-2)


@pytest.mark.isaacsim_ci
def test_newton_matches_physx_single_motor_impulse():
    """Both backends produce equivalent linear and angular response to one motor impulse."""
    cases = _backend_cases(gravity=(0.0, 0.0, 0.0))
    results = {
        name: _run_rigid_trajectory(
            multirotor_type,
            sim_cfg,
            thrust_target=[8.0, 0.0, 0.0, 0.0],
            command_steps=1,
            coast_steps=29,
            motor_time_constant=0.0,
        )
        for name, (multirotor_type, sim_cfg) in cases.items()
    }

    for result in results.values():
        # An 8 N impulse applied for one 1/120 s step to the 1.2 kg fixture should create a readily measurable
        # translation and both linear and angular velocity. These bounds reject an effectively disconnected wrench.
        assert torch.linalg.vector_norm(result["displacement"]) > 5.0e-3
        assert torch.linalg.vector_norm(result["velocity"][:, :3]) > 3.0e-2
        assert torch.linalg.vector_norm(result["velocity"][:, 3:]) > 1.0e-1
    _assert_rigid_trajectory_close(results["newton"], results["physx"], 7.5e-2, 1.0e-2, 1.5e-2)


@pytest.mark.isaacsim_ci
def test_newton_root_and_motor_link_modes_match_physx():
    """Rigid-airframe root and motor load placement are equivalent in both backends."""
    cases = _backend_cases(gravity=(0.0, 0.0, 0.0))
    results = {}
    for backend_name, (multirotor_type, sim_cfg) in cases.items():
        results[backend_name] = {
            force_application_level: _run_rigid_trajectory(
                multirotor_type,
                sim_cfg.replace(),
                thrust_target=[1.0, 2.0, 3.0, 4.0],
                command_steps=20,
                force_application_level=force_application_level,
                motor_time_constant=0.0,
            )
            for force_application_level in ("root_link", "motor_link")
        }

    for backend_results in results.values():
        for result in backend_results.values():
            assert torch.linalg.vector_norm(result["displacement"]) > 5.0e-2
            orientation_change = math_utils.quat_error_magnitude(result["quaternion"], result["initial_quaternion"])
            assert torch.max(orientation_change) > 1.0e-1
        # Applying loads to fixed child links requires constraint projection, while the equivalent root wrench does
        # not. Allow the small backend-local integration difference without relaxing the Newton-to-PhysX checks.
        _assert_rigid_trajectory_close(
            backend_results["motor_link"], backend_results["root_link"], 7.5e-2, 2.0e-3, 2.0e-2
        )
    for force_application_level in ("root_link", "motor_link"):
        _assert_rigid_trajectory_close(
            results["newton"][force_application_level],
            results["physx"][force_application_level],
            7.5e-2,
            4.0e-3,
            1.5e-2,
        )


@pytest.mark.isaacsim_ci
def test_newton_matches_physx_aerodynamic_drag():
    """Both backends produce equivalent linear and angular aerodynamic damping."""
    cases = _backend_cases(gravity=(0.0, 0.0, 0.0))
    results = {
        name: _run_rigid_trajectory(
            multirotor_type,
            sim_cfg,
            thrust_target=[0.0] * 4,
            command_steps=60,
            initial_velocity=(1.0, -0.5, 0.25, 0.4, -0.2, 0.3),
            linear_drag=(0.2, 0.05),
            angular_drag=(0.02, 0.01),
        )
        for name, (multirotor_type, sim_cfg) in cases.items()
    }

    for result in results.values():
        initial_linear_speed = torch.linalg.vector_norm(result["initial_velocity"][:, :3])
        initial_angular_speed = torch.linalg.vector_norm(result["initial_velocity"][:, 3:])
        assert torch.linalg.vector_norm(result["velocity"][:, :3]) < 0.95 * initial_linear_speed
        assert torch.linalg.vector_norm(result["velocity"][:, 3:]) < 0.90 * initial_angular_speed
    _assert_rigid_trajectory_close(results["newton"], results["physx"], 7.5e-2, 1.0e-2, 1.5e-2)


@pytest.mark.isaacsim_ci
def test_newton_jointed_airframe_matches_physx_internal_response():
    """Both backends preserve the internal-response distinction between root and motor loads."""
    cases = _backend_cases(gravity=(0.0, 0.0, 0.0))
    results = {}
    for backend_name, (multirotor_type, sim_cfg) in cases.items():
        results[backend_name] = {
            force_application_level: _run_jointed_trajectory(
                multirotor_type, sim_cfg.replace(), force_application_level
            )
            for force_application_level in ("root_link", "motor_link")
        }

    for backend_results in results.values():
        mode_difference = torch.abs(
            backend_results["motor_link"]["joint_displacement"] - backend_results["root_link"]["joint_displacement"]
        )
        assert torch.max(mode_difference) > 1.0e-2

    for force_application_level in ("root_link", "motor_link"):
        newton_result = results["newton"][force_application_level]
        physx_result = results["physx"][force_application_level]
        torch.testing.assert_close(
            newton_result["joint_displacement"], physx_result["joint_displacement"], rtol=1.25e-1, atol=2.0e-2
        )
        torch.testing.assert_close(
            newton_result["joint_velocity"], physx_result["joint_velocity"], rtol=1.25e-1, atol=2.0e-2
        )
        torch.testing.assert_close(
            newton_result["root_displacement"], physx_result["root_displacement"], rtol=1.25e-1, atol=7.5e-3
        )
        torch.testing.assert_close(
            newton_result["root_velocity"], physx_result["root_velocity"], rtol=1.25e-1, atol=2.0e-2
        )


@pytest.mark.isaacsim_ci
def test_newton_matches_physx_reset_lifecycle():
    """Newton and PhysX restore only the selected environment after a partial reset."""
    cases = _backend_cases(gravity=(0.0, 0.0, 0.0))
    results = {
        name: _run_reset_lifecycle(multirotor_type, sim_cfg) for name, (multirotor_type, sim_cfg) in cases.items()
    }

    for result in results.values():
        torch.testing.assert_close(result["default_target"], torch.ones_like(result["default_target"]))
        expected_commanded_thrust = torch.tensor([[4.0] * 4, [5.0] * 4])
        expected_reset_thrust = torch.tensor([[4.0] * 4, [1.0] * 4])
        torch.testing.assert_close(result["commanded_thrust"], expected_commanded_thrust)
        torch.testing.assert_close(result["reset_target"], expected_reset_thrust)
        torch.testing.assert_close(result["reset_actuator_state"], expected_reset_thrust)
        torch.testing.assert_close(result["reset_applied_thrust"], expected_reset_thrust)
    for key in results["physx"]:
        torch.testing.assert_close(results["newton"][key], results["physx"][key])


def test_newton_wrench_write_ignores_empty_composers(monkeypatch):
    """Newton leaves the bound wrench array untouched when neither composer has pending loads."""
    _, MultirotorNewton, _ = import_newton_types()
    import isaaclab_contrib.assets.multirotor.multirotor_newton as multirotor_newton_module  # noqa: PLC0415

    multirotor = _make_uninitialized_newton_multirotor(MultirotorNewton)
    multirotor._instantaneous_wrench_composer = types.SimpleNamespace(active=False)
    multirotor._permanent_wrench_composer = types.SimpleNamespace(active=False)
    monkeypatch.setattr(
        multirotor_newton_module.wp,
        "launch",
        lambda *args, **kwargs: pytest.fail("Empty wrench composers must not launch the Newton update kernel."),
    )

    multirotor._write_external_wrenches_to_sim()
