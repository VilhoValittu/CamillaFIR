from __future__ import annotations

import numpy as np


def apply_manual_target_preview_shift(target_curve, shift_db: float):
    """Apply manual level offset to the displayed target curve only."""
    target_arr = np.asarray(target_curve, dtype=float)
    try:
        shift = float(shift_db)
    except Exception:
        shift = 0.0

    if not np.isfinite(shift) or abs(shift) <= 1e-9:
        return target_arr
    return target_arr + shift

