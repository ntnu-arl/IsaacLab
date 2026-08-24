# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility exports for the multirotor implementation.

For now, :class:`Multirotor` remains the PhysX implementation. New code that needs to name
the backend explicitly should import :class:`MultirotorPhysx` or :class:`MultirotorNewton` from this package.
"""

from .multirotor_physx import MultirotorDataPhysx, MultirotorPhysx

Multirotor = MultirotorPhysx
MultirotorData = MultirotorDataPhysx

__all__ = ["Multirotor", "MultirotorData"]
