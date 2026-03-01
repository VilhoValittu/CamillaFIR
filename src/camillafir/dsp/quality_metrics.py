from __future__ import annotations

import numpy as np


def _mag_error_db(target_db, measured_db, gain_db, offset_db):
    """Return magnitude error in dB: target - (measured + gain + offset)."""
    target = np.asarray(target_db, dtype=float)
    measured = np.asarray(measured_db, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    try:
        offset = float(offset_db)
    except Exception:
        offset = 0.0
    predicted = measured + gain + offset
    return target - predicted


def _rms(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))
