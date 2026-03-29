"""Public auto-mode package surface."""

from .api import (
    AUTO_MODE_COMPAT_VERSION,
    AUTO_MODE_EXC_FROM_F6_ADD_HZ,
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_REFINE_TRIALS,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    AUTO_MODE_TRIALS,
    _auto_select_target_curve_with_trials,
    _run_auto_mode_search,
    get_auto_mode_cache_path,
)
from .protection_seed import (
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
)
from .target_preselection import _auto_select_builtin_target_curve

__all__ = [
    "AUTO_MODE_COMPAT_VERSION",
    "AUTO_MODE_EXC_FROM_F6_ADD_HZ",
    "AUTO_MODE_EXC_MAX_HZ",
    "AUTO_MODE_EXC_MIN_HZ",
    "AUTO_MODE_LOCAL_REFINE_ENABLED",
    "AUTO_MODE_LOCAL_REFINE_TOP_K",
    "AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP",
    "AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ",
    "AUTO_MODE_LOW_BASS_MAX_HZ",
    "AUTO_MODE_LOW_BASS_MIN_HZ",
    "AUTO_MODE_REFINE_TRIALS",
    "AUTO_MODE_TARGET_TOP_N",
    "AUTO_MODE_TARGET_TRIALS_PER_CURVE",
    "AUTO_MODE_TRIALS",
    "_auto_select_builtin_target_curve",
    "_auto_select_target_curve_with_trials",
    "_estimate_auto_hpf_from_response",
    "_estimate_auto_mag_c_min_hz",
    "_resolve_auto_hpf_application",
    "_run_auto_mode_search",
    "get_auto_mode_cache_path",
]
