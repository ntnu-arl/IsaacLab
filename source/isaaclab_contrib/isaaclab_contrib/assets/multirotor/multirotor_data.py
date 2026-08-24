# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility export for multirotor data during the MR-04 backend migration."""

from .multirotor_physx import MultirotorDataPhysx

MultirotorData = MultirotorDataPhysx

__all__ = ["MultirotorData"]
