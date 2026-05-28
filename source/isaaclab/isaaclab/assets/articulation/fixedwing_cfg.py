# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch
import numpy as np

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
        Cd_max=2.1,
        Cd_s=0.0,
        Cd_da=0.0,
        Cl_da=0.0,
        Cl_max=1.1,
        Cm_max=0.0,
        Cm_da=0.0,
        Cm_s=0.0,
        dCl_dq=0.0,
        dCm_dq=0.0,
        dCd_dq=0.0,
        stallable: bool = True,
        offset_angle: float = 0.0,
        stall_angle: float = 12.0,
        stall_range: float = 8.0,
        width: float = 0.0,
        chord: float = 0.0,
        has_controlsurface: bool = False,
        connected_actuator: str = "",
        q_reduced_effectiveness: float = 0.4,
        mixed_airflow: bool = False,
        influenced_by: str = "",
        mixed_airflow_coefficient: float = 0.3,
    ):

        self.Cd_max = Cd_max
        self.Cd_s = Cd_s
        self.Cd_da = float(np.rad2deg(Cd_da))
        self.Cl_da = float(np.rad2deg(Cl_da))
        self.Cl_max = Cl_max
        self.Cm_max = Cm_max
        self.Cm_da = float(np.rad2deg(Cm_da))
        self.Cm_s = Cm_s
        self.dCl_dq = float(np.rad2deg(dCl_dq))
        self.dCm_dq = float(np.rad2deg(dCm_dq))
        self.dCd_dq = float(np.rad2deg(dCd_dq))
        self.G_rdp = 1.17 * ((chord * 0.5) ** 3) / 2 * width
        self.G_rdr = 1.17 * ((width * 0.5) ** 3) / 2 * chord
        self.chord = chord
        self.width = width
        self.wing_area_projected = width * chord
        self.stallable = stallable
        self.offset_angle = float(np.deg2rad(offset_angle))
        self.stall_angle = float(np.deg2rad(stall_angle))
        self.stall_range = float(np.deg2rad(stall_range))
        self.has_controlsurface = has_controlsurface
        self.connected_actuator = connected_actuator
        self.q_reduced_effectiveness = q_reduced_effectiveness
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
    """Configuration parameters for a fixed wing articulation.

    This extends the base articulation configuration to support fixed wing-specific
    settings.
    """

    class_type: type = FixedWing

    wings: dict[str, FloaterCfg] = MISSING
    """Configuration for each floater."""

    engines: dict[str, EngineCfg] = MISSING
    """Configuration for each engine."""

    rho: float = 1.225
    """Air density in kg/m^3."""
