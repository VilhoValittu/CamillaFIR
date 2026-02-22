from __future__ import annotations

import numpy as np


def cfg_float_allow_zero(cfg, key: str, default: float) -> float:
    """Funktio: cfg float allow zero."""
    try:
        v = getattr(cfg, key, default)
    except Exception:
        v = default
    if v is None:
        return float(default)
    if isinstance(v, str) and v.strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def safe_range(x, default_min=200.0, default_max=3000.0):
    try:
        a = float(x[0])
        b = float(x[1])
        if np.isfinite(a) and np.isfinite(b) and b > a:
            return [a, b]
    except Exception:
        pass
    return [float(default_min), float(default_max)]
