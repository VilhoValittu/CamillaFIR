from __future__ import annotations

import numpy as np

from ..application.house_curve_service import apply_manual_target_curve_tilt


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


def apply_manual_target_preview_adjustments(
    freq_axis,
    target_curve,
    shift_db: float,
    tilt_db_per_oct: float = 0.0,
):
    """Apply manual preview-only target adjustments on top of the base curve."""
    tilted = apply_manual_target_curve_tilt(freq_axis, target_curve, tilt_db_per_oct)
    return apply_manual_target_preview_shift(tilted, shift_db)
