from __future__ import annotations

from .dsp_config import CfgReader, coerce_range2


def cfg_float_allow_zero(cfg, key: str, default: float) -> float:
    """Funktio: cfg float allow zero."""
    return CfgReader(cfg).float_allow_zero(key, default)


def safe_range(x, default_min=200.0, default_max=3000.0):
    return coerce_range2(x, default_min, default_max)
