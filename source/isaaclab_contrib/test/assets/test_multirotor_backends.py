# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary PhysX-to-Newton multirotor comparison tests for MR-04."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cpu").app

"""Everything below follows simulator launch."""

import inspect

import pytest
from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

from isaaclab_contrib.assets import (
    Multirotor,
    MultirotorBase,
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
    """The Newton scaffold exposes the same common multirotor method signatures."""
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


def test_newton_wrench_write_is_an_explicit_scaffold():
    """Newton fails clearly until its backend-specific wrench path is implemented."""
    _, MultirotorNewton, _ = import_newton_types()
    multirotor = object.__new__(MultirotorNewton)
    with pytest.raises(NotImplementedError, match="Newton multirotor wrench application"):
        multirotor._write_external_wrenches_to_sim()
