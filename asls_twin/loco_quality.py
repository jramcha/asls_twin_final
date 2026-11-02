"""Common acceptance rule for measured LOCO history across all stages."""

from __future__ import annotations

import numpy as np


MIN_HISTORY_CURRENT_A = 0.170


def accepted_history_current(current_A: float | None) -> bool:
    """Only a recorded, finite current of at least 170 mA enters model history."""
    return (current_A is not None and np.isfinite(current_A)
            and MIN_HISTORY_CURRENT_A <= current_A <= 1.0)
