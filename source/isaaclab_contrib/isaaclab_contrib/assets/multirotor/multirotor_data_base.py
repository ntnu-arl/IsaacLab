# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class MultirotorDataBase:
    """Backend-independent data API for a multirotor articulation.

    Backend data containers combine this interface with their articulation data implementation. All tensor attributes
    have shape ``(num_instances, num_thrusters)``.
    """

    thruster_names: list[str] | None = None
    """Ordered thruster names matching the thrust tensor columns."""

    default_thruster_rps: torch.Tensor | None = None
    """Default thruster speeds [revolutions/s]."""

    thrust_target: torch.Tensor | None = None
    """Target thrusts [N]."""

    computed_thrust: torch.Tensor | None = None
    """Computed thrusts before clipping [N]."""

    applied_thrust: torch.Tensor | None = None
    """Applied thrusts after clipping [N]."""
