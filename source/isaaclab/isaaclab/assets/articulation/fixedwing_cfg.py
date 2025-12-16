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

    def __init__(
        self,
        C_d: float = 2.25,
        C_lt: float = 1.2,
        C_m: float = 0.06,
        wing_area_projected: float = 1,
        has_controlsurface: bool = False,
        connected_actuator: str = "",
        q_drag: float = 0.0,
        q_lift: float = 0.0,
        q_torque: float = 0.0,
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


class EngineCfg:
    """Configuration parameters for a multirotor articulation.

    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    def __init__(
        self,
        max_thrust: float = 15.0,
        min_rpm: float = 0.0,
        max_rpm: float = 1000.0,
        thrust_coefficient: float = 1.0,
        torque_coefficient: float = 0.1,
        connected_actuator: str = "",
        spin_direction: int = 1,
    ):
        self.max_thrust = max_thrust
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self.thrust_coefficient = thrust_coefficient
        self.torque_coefficient = torque_coefficient
        self.connected_actuator = connected_actuator
        self.spin_direction = spin_direction


@configclass
class FixedWingCfg(ArticulationCfg):
    """Configuration parameters for a multirotor articulation.

    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    class_type: type = FixedWing

    wings: dict[str, FloaterCfg] = MISSING
    """Configuration for each floater."""

    engines: dict[str, EngineCfg] = MISSING
    """Configuration for each engine."""

    rho: float = 1.225
    """Air density in kg/m^3."""
