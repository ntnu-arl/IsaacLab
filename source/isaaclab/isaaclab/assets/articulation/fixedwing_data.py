# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets.articulation.articulation_data import ArticulationData


class FixedWingData(ArticulationData):
    """Data container for a multirotor articulation.

    This class extends the base articulation data container to include multirotor-specific
    data such as thruster states and forces.
    """

    aero_links: dict[str, int] = None
    """Dictionary mapping aerodynamic link names to their body indices."""

    # --- IGNORE ---
