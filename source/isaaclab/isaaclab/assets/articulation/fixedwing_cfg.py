# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch

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
        C_ds: float = 0.05,
        C_ll: float = 2.6,
        C_lt: float = 1.2,
        C_m: float = 0.06,
        stallable: bool = True,
        stall_angle: float = 8.0,
        stall_range: float = 12.0,
        width: float = 0.0,
        depth: float = 0.0,
        has_controlsurface: bool = False,
        connected_actuator: str = "",
        q_reduced_effectiveness: float = 0.4,
        q_drag: float = 0.0,
        q_lift: float = 0.0,
        q_torque: float = 0.0,
        mixed_airflow: bool = False,
        influenced_by: str = "",
        mixed_airflow_coefficient: float = 0.3,
    ):
        self.C_d = C_d
        self.C_ds = C_ds
        self.C_ll = C_ll
        self.C_lt = C_lt
        self.C_m = C_m
        self.C_rdp = 1.17 * ((depth * 0.5) ** 3) / 2 * width
        self.C_rdr = 1.17 * ((width * 0.5) ** 3) / 2 * depth
        self.wing_area_projected = width * depth
        self.stallable = stallable
        self.stall_angle = stall_angle / 180.0 * 3.141592653589793
        self.stall_range = stall_range / 180.0 * 3.141592653589793
        self.has_controlsurface = has_controlsurface
        self.connected_actuator = connected_actuator
        self.q_reduced_effectiveness = q_reduced_effectiveness
        self.q_drag = q_drag
        self.q_lift = q_lift
        self.q_torque = q_torque
        self.mixed_airflow = mixed_airflow
        self.influenced_by = influenced_by
        self.mixed_airflow_coefficient = mixed_airflow_coefficient


class EngineCfg:
    """Configuration parameters for a multirotor articulation.

    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    def __init__(
        self,
        max_thrust: float = 15.0,
        max_rpm: float = 1000.0,
        thrust_coefficient: float = 1.0,
        torque_coefficient: float = 0.1,
        connected_actuator: str = "",
        spin_direction: int = 1,
        effectiveness: float = 0.0,
    ):
        self.max_thrust = max_thrust
        self.max_rpm = max_rpm
        self.thrust_coefficient = thrust_coefficient
        self.torque_coefficient = torque_coefficient
        self.connected_actuator = connected_actuator
        self.spin_direction = spin_direction
        self.effectiveness = effectiveness


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
