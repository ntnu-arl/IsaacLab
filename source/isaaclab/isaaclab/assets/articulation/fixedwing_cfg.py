# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING


from isaaclab.utils import configclass

from .articulation_cfg import ArticulationCfg
from .fixedwing import FixedWing


@configclass
class FloaterCfg:
    """Configuration parameters for a multirotor articulation.

    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    def __ini__(
        self,
        C_d: float = 2.25,
        C_lt: float = 1.2,
        C_m: float = 0.06,
        wing_area_projected: float = 1,
        has_controlsurface: bool = False,
        connected_actuator: str = "",
        q_drag: float = 0.1,
        q_lift: float = 0.3,
        q_torque: float = 0.1,
    ):
        self.C_d = C_d
        self.C_lt = C_lt
        self.C_m = C_m
        self.wing_area_projected = wing_area_projected
        self.has_controlsurface = has_controlsurface
        self.connected_actuator = connected_actuator
        self.q_drag = q_drag
        self.q_lift = q_lift
        self.q_torque = q_torque


@configclass
class FixedWingCfg(ArticulationCfg):
    """Configuration parameters for a multirotor articulation.

    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    class_type: type = FixedWing

    aero_link_names: str | list[str] = MISSING
    """Name or list of names of the links where aerodynamic forces are applied."""
    controlsurface_mapping: dict[str, str] = MISSING
    """Mapping from control surface joint names to aerodynamic link names."""

    controlactuator_mapping: dict[str, str] = MISSING
    """Mapping from aerodynamic link names to actuator joint names."""

    # Drag coefficients (per-axis scalars)
    lin_drag_linear_coef: float = 1.0
    lin_drag_quadratic_coef: float = 1.0
    """Drag coefficients for linear velocity."""

    ang_drag_linear_coef: float = 1.0
    ang_drag_quadratic_coef: float = 1.0
    """Drag coefficients for angular velocity."""

    C_d: float = 2.25
    """Overall drag coefficient for the fixed-wing body."""
    C_lt: float = 1.2
    """Lift coefficient for the fixed-wing body in turbulent flow."""
    C_m: float = 0.06
    """Moment coefficient for the fixed-wing body."""

    rho: float = 1.225
    """Air density in kg/m^3."""
    wing_area_projected: float = 0.25
    """Projected wing area in m^2."""

    q_drag: float = 0.05
    q_lift: float = 0.5
    q_torque: float = 0.1
    """Control surface effectiveness coefficients."""
