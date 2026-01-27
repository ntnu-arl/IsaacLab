# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch


class FixedWingData:
    """Data container for a multirotor articulation.

    This class extends the base articulation data container to include multirotor-specific
    data such as thruster states and forces.
    """

    aero_link_mapping: dict[str, int] = {}
    """Dictionary mapping aerodynamic link names to their indices."""

    aero_actuator_link_mapping: dict[str, int] = {}
    """Dictionary mapping aerodynamic actuator link names to their indices."""

    engine_link_mapping: dict[str, int] = {}
    """Dictionary mapping engine link names to their indices."""

    engine_actuator_idx_mapping: dict[str, int] = {}
    """Dictionary mapping engine actuator names to their indices."""

    wing_drag_tensor: dict[str, torch.Tensor] = {}
    """Dictionary mapping wing link names to their drag tensors."""

    # --- IGNORE ---
