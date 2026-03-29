"""Public auto-mode package surface.

Keep package exports lazy so submodule imports such as
``camillafir.auto_mode.filter_priors`` do not eagerly import the full auto-mode
stack during package initialization.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "AUTO_MODE_COMPAT_VERSION": ("api", "AUTO_MODE_COMPAT_VERSION"),
    "AUTO_MODE_EXC_FROM_F6_ADD_HZ": ("api", "AUTO_MODE_EXC_FROM_F6_ADD_HZ"),
    "AUTO_MODE_EXC_MAX_HZ": ("api", "AUTO_MODE_EXC_MAX_HZ"),
    "AUTO_MODE_EXC_MIN_HZ": ("api", "AUTO_MODE_EXC_MIN_HZ"),
    "AUTO_MODE_LOCAL_REFINE_ENABLED": ("api", "AUTO_MODE_LOCAL_REFINE_ENABLED"),
    "AUTO_MODE_LOCAL_REFINE_TOP_K": ("api", "AUTO_MODE_LOCAL_REFINE_TOP_K"),
    "AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP": ("api", "AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP"),
    "AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ": ("api", "AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ"),
    "AUTO_MODE_LOW_BASS_MAX_HZ": ("api", "AUTO_MODE_LOW_BASS_MAX_HZ"),
    "AUTO_MODE_LOW_BASS_MIN_HZ": ("api", "AUTO_MODE_LOW_BASS_MIN_HZ"),
    "AUTO_MODE_REFINE_TRIALS": ("api", "AUTO_MODE_REFINE_TRIALS"),
    "AUTO_MODE_TARGET_TOP_N": ("api", "AUTO_MODE_TARGET_TOP_N"),
    "AUTO_MODE_TARGET_TRIALS_PER_CURVE": ("api", "AUTO_MODE_TARGET_TRIALS_PER_CURVE"),
    "AUTO_MODE_TRIALS": ("api", "AUTO_MODE_TRIALS"),
    "_auto_select_target_curve_with_trials": ("api", "_auto_select_target_curve_with_trials"),
    "_run_auto_mode_search": ("api", "_run_auto_mode_search"),
    "get_auto_mode_cache_path": ("api", "get_auto_mode_cache_path"),
    "_estimate_auto_hpf_from_response": ("protection_seed", "_estimate_auto_hpf_from_response"),
    "_estimate_auto_mag_c_min_hz": ("protection_seed", "_estimate_auto_mag_c_min_hz"),
    "_resolve_auto_hpf_application": ("protection_seed", "_resolve_auto_hpf_application"),
    "_auto_select_builtin_target_curve": ("target_preselection", "_auto_select_builtin_target_curve"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name == "api":
        module = import_module(f"{__name__}.api")
        globals()[name] = module
        return module

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"api"})
