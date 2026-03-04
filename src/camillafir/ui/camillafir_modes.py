from __future__ import annotations

from typing import Dict, Any, Tuple
from camillafir.config.models import FilterConfig


def _clamp_float(v, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except Exception:
        return float(lo)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _apply_defaults(cfg: FilterConfig, d: Dict[str, Any]) -> None:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def _apply_clamps(cfg: FilterConfig, clamps: Dict[str, Tuple[Any, Any]]) -> None:
    for k, lim in clamps.items():
        if not hasattr(cfg, k):
            continue

        lo, hi = lim

        if isinstance(lo, bool) and isinstance(hi, bool):
            setattr(cfg, k, bool(lo))
            continue

        if lo == hi:
            setattr(cfg, k, lo)
            continue

        try:
            cur = getattr(cfg, k)
            setattr(cfg, k, _clamp_float(cur, float(lo), float(hi)))
        except Exception:
            pass



MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "BASIC": {
        "filter_type_str": "Linear Phase",
        "global_gain_db": 0.0,

        "enable_mag_correction": True,
        "unsafe_raw_dsp": False,
        "mag_c_min": 25.0,
        "mag_c_max": 250.0,
        "max_boost_db": 3.0,
        "max_cut_db": 15.0,

        "phase_safe_2058": False,
        "phase_limit": 400.0,

        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 12,
        "fdw_cycles": 10.0,
        "reg_strength": 30.0,

        "max_slope_db_per_oct": 12.0,
        "max_slope_boost_db_per_oct": 6.0,
        "max_slope_cut_db_per_oct": 24.0,
        "df_smoothing": True,

        "enable_tdc": True,
        "tdc_strength": 50.0,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "enable_afdw": True,
        


        "ir_export_window_mode": "auto",
        "ir_window_right": 500.0,
        "ir_window_left": 80.0,
        "mixed_split_freq": 180.0,
        "trans_width": 100.0,

        "bass_first_ai": True,
        "bass_first_mode_max_hz": 180.0,

        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",

        "do_normalize": False,
        "exc_prot": True,
        "low_bass_cut_hz": 50.0,
        "low_bass_cut_enable": True,
    },

    "ADVANCED": {
        "filter_type_str": "Mixed Phase",
        "global_gain_db": 0.0,

        "enable_mag_correction": True,
        "unsafe_raw_dsp": False,
        "mag_c_min": 18.0,
        "mag_c_max": 230.0,
        "max_boost_db": 5.0,
        "max_cut_db": 24.0,

        "phase_safe_2058": False,
        "phase_limit": 320.0,

        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 48,
        "fdw_cycles": 10.0,
        "reg_strength": 18.0,

        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 36.0,
        "max_slope_cut_db_per_oct": 0.0,
        "df_smoothing": False,

        "enable_tdc": True,
        "tdc_strength": 15.0,
        "tdc_max_reduction_db": 6.0,
        "tdc_slope_db_per_oct": 12.0,
        "enable_afdw": True,
        "ir_window_right": 500.0,
        "ir_window_left": 85.0,
        "ir_export_window_mode": "auto",
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,

        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "lvl_min": 200.0,
        "lvl_max": 3000.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "mixed_split_freq": 180.0,
        "trans_width": 100.0,
        "do_normalize": False,
        "exc_prot": True,
        "low_bass_cut_hz": 40.0,
        "low_bass_cut_enable": True,
        "comparison_mode": True,
        "bass_adaptive_isolation_mode": True,
        "conf_pull_floor": 0.05,
        "conf_pull_max_hz": 180.0,
        "conf_pull_gamma_cut": 0.554,
        "conf_pull_gamma_boost": 0.75,
        "low_bass_cut_strength": 0.0,
        "ir_anchor_mode": "min_causal",
    },
}


MODE_CLAMPS: Dict[str, Dict[str, Tuple[Any, Any]]] = {
    "BASIC": {
        "max_boost_db": (0.0, 4.0),
        "max_cut_db": (0.0, 15.0),

        "filter_smooth": (1, 24),
        "reg_strength": (10.0, 60.0),
        "ir_export_window_mode": ("auto", "auto"),
        "enable_tdc": (True, True),
        "tdc_strength": (0.0, 70.0),
        "tdc_max_reduction_db": (0.0, 12.0),
        "tdc_slope_db_per_oct": (0.0, 12.0),
        "mixed_split_freq": (100.0, 200.0),
        "enable_afdw": (True, True),
        "fdw_cycles": (10.0, 15.0),
        "mag_c_min": (18.0, 300.0),
        "mag_c_max": (18.0, 300.0),
        "phase_limit": (200.0, 450.0),
        "low_bass_cut_hz": (20.0, 100.0),
        "low_bass_cut_enable": (True, True),
        "stereo_link": (True, True),
        "unsafe_raw_dsp": (False, False),
    },

    "ADVANCED": {},
}

# AUTO uses BASIC defaults/guardrails but enables automatic preset search.
MODE_DEFAULTS["AUTO"] = dict(MODE_DEFAULTS.get("ADVANCED", {}))
MODE_CLAMPS["AUTO"] = dict(MODE_CLAMPS.get("ADVANCED", {}))
MODE_DEFAULTS["AUTO"]["stereo_link_strategy"] = "auto"


def apply_mode_to_cfg(cfg: FilterConfig, mode: str | None, *, apply_defaults: bool = True) -> FilterConfig:
    """Soveltaa tai paivittaa: apply mode to cfg."""
    m = (mode or "BASIC").upper().strip()
    if m not in MODE_DEFAULTS:
        m = "BASIC"

    if apply_defaults:
        _apply_defaults(cfg, MODE_DEFAULTS[m])
    _apply_clamps(cfg, MODE_CLAMPS.get(m, {}))
    return cfg
