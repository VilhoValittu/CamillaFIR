"""Compatibility shim — policy logic lives in config.mode_policy."""
from ..config.mode_policy import (  # noqa: F401
    MODE_CLAMPS,
    MODE_DEFAULTS,
    _apply_clamps,
    _apply_defaults,
    _clamp_float,
    apply_mode_to_cfg,
)
