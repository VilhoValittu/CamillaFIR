# camillafir_modes.py
from __future__ import annotations

from typing import Dict, Any, Tuple
from models import FilterConfig  # uses your existing dataclass fields :contentReference[oaicite:1]{index=1}

#Version: v0.1.1

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

        # bool clamp (forced True/False)
        if isinstance(lo, bool) and isinstance(hi, bool):
            setattr(cfg, k, bool(lo))
            continue

        # numeric clamp
        try:
            cur = getattr(cfg, k)
            setattr(cfg, k, _clamp_float(cur, float(lo), float(hi)))
        except Exception:
            # never fail mode application
            pass


# ---------------------------------------------------------------------
# Mode policy
# ---------------------------------------------------------------------

MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # BASIC
    "BASIC": {
        # --- BASIC ---
        "filter_type_str": "Linear Phase",
        "global_gain_db": 0.0,

        # --- CORRECTION LIMITS ---
        "enable_mag_correction": True,
        "mag_c_min": 25.0,
        "mag_c_max": 250.0,
        "max_boost_db": 3.0,
        "max_cut_db": 30.0,

        # Phase: allow, but still bounded by other rails
        "phase_safe_2058": False,
        "phase_limit": 600.0,

        # --- SMOOTHING / FDW / REG ---
        "smoothing_type": "Psychoacoustic",
        "smoothing_level": 12,     # 1/12 oct default
        "fdw_cycles": 10.0,
        "reg_strength": 30.0,

        # Asymmetric slope rails
        "max_slope_db_per_oct": 12.0,          # fallback/back-compat
        "max_slope_boost_db_per_oct": 6.0,
        "max_slope_cut_db_per_oct": 24.0,
        "df_smoothing": True,

        # --- ADVANCED FEATURES (ON, guarded) ---
        "enable_tdc": True,
        "tdc_strength": 50.0,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "enable_afdw": True,

        # --- IR WINDOW / MIXED ---
        "ir_window_ms": 500.0,
        "ir_window_ms_left": 100.0,
        "mixed_split_freq": 300.0,
        "trans_width": 100.0,

        # --- BASS-FIRST ---
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 180.0,

        # --- LEVELING ---
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 75.0,
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "stereo_link": True,

        # --- PROTECTIONS ---
        "do_normalize": False,
        "exc_prot": True,
        "low_bass_cut_hz": 50.0,
    },

    # ADVANCED = your “expert” profile: minimal policy, no clamps
    "ADVANCED": {
        "filter_type_str": "Linear Phase",
        "global_gain_db": 0.0,

        "enable_mag_correction": True,
        "mag_c_min": 18.0,
        "mag_c_max": 200.0,
        "max_boost_db": 3.0,
        "max_cut_db": 30.0,

        "phase_safe_2058": False,
        "phase_limit": 400.0,

        "smoothing_type": "Psychoacoustic",
        "smoothing_level": 24,
        "fdw_cycles": 10.0,
        "reg_strength": 30.0,

        # Off by default: user decides
        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        # Auto by default: internal normalization (no UI control)
        "df_smoothing": False,

        "enable_tdc": True,
        "tdc_strength": 50.0,
        "tdc_max_reduction_db": 12.0,
        "tdc_slope_db_per_oct": 0.0,
        "enable_afdw": True,

        "ir_window_ms": 500.0,
        "ir_window_ms_left": 50.0,

        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,

        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 75.0,
        "lvl_min": 200.0,
        "lvl_max": 3000.0,
        "stereo_link": True,

        "do_normalize": False,
        "exc_prot": False,
        "low_bass_cut_hz": 15.0,
    },
}


MODE_CLAMPS: Dict[str, Dict[str, Tuple[Any, Any]]] = {
    # BASIC: guard rails (hard clamps)
    "BASIC": {
        "max_boost_db": (0.0, 6.0),
        "max_cut_db": (0.0, 15.0),

        "smoothing_level": (1, 24),
        "reg_strength": (10.0, 60.0),

        "enable_tdc": (True, True),
        "tdc_strength": (0.0, 70.0),
        "tdc_max_reduction_db": (0.0, 12.0),
        "tdc_slope_db_per_oct": (0.0, 12.0),

        "enable_afdw": (True, True),
        "fdw_cycles": (6.0, 16.0),
        # Correction band guard rails (BASIC)
        "mag_c_min": (18.0, 300.0),
        "mag_c_max": (18.0, 300.0),
        "phase_limit": (200.0, 1000.0),
        "low_bass_cut_hz": (20.0, 100.0),
    },

    # ADVANCED: no clamps
    "ADVANCED": {},
}


def apply_mode_to_cfg(cfg: FilterConfig, mode: str | None, *, apply_defaults: bool = False) -> FilterConfig:
    """
    Apply mode defaults + clamps to an existing FilterConfig.

    Modes:
      - BASIC: guarded "studio"
      - ADVANCED: freer "expert" (no clamps)

    Safe behavior:
      - Unknown mode -> BASIC
      - Never raises (best-effort)
    """
    m = (mode or "BASIC").upper().strip()
    if m not in MODE_DEFAULTS:
        m = "BASIC"

    if apply_defaults:
        _apply_defaults(cfg, MODE_DEFAULTS[m])
    _apply_clamps(cfg, MODE_CLAMPS.get(m, {}))
    return cfg
