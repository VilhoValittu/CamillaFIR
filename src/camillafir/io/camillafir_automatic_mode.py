import logging
import math
import json
import os
import hashlib
import time
import random
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..dsp import camillafir_dsp as dsp
from ..dsp.target_match import target_match_from_stats
from ..engine import build_config, run_pipeline, summarize_run
from ..app_paths import camillafir_data_dir
from ..ui import camillafir_plot as plots
from ..ui.camillafir_housecurve import get_house_curve_by_name

logger = logging.getLogger("CamillaFIR")
MAX_SAFE_BOOST = 8.0
AUTO_MODE_TRIALS = 100
AUTO_MODE_REFINE_TRIALS = 50
AUTO_MODE_GOAL_DEFAULT = "balanced"
AUTO_MODE_GOAL_ROOM_SAFE = "room-safe"
AUTO_MODE_GOAL_LOW_RIPPLE = "low-ripple"
AUTO_MODE_GOAL_FLAT = "flat"
# Legacy names kept for backward compatibility (cache/config from older versions).
AUTO_MODE_GOAL_ACOUSTIC = "acoustic"
AUTO_MODE_GOAL_HYBRID = "hybrid"

# --- Auto-mode preset cache ---
# Stores best preset per (measurement + key settings) signature, so next run can start from it.
AUTO_MODE_CACHE_ENABLED = True
AUTO_MODE_CACHE_MAX_ITEMS = 64
AUTO_MODE_CACHE_FILENAME = "camillafir_auto_mode_cache.json"
AUTO_MODE_CACHE_FILTER_KEYS = ("linear", "mixed", "minimum", "asym")
_AUTO_CACHE_VERSION_MISMATCH_LOGGED = False


def _auto_cache_path() -> str:
    # Preferred: platform-correct app data dir.
    preferred_base = os.fspath(camillafir_data_dir())
    preferred_path = os.path.join(preferred_base, AUTO_MODE_CACHE_FILENAME)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, AUTO_MODE_CACHE_FILENAME)

    try:
        os.makedirs(preferred_base, exist_ok=True)
    except Exception:
        # Keep legacy fallback writable if preferred base is not available.
        try:
            os.makedirs(legacy_base, exist_ok=True)
        except Exception:
            pass
        return legacy_path

    # One-time migration from legacy location to new platform path.
    try:
        if (not os.path.isfile(preferred_path)) and os.path.isfile(legacy_path):
            with open(legacy_path, "rb") as src_f:
                payload = src_f.read()
            with open(preferred_path, "wb") as dst_f:
                dst_f.write(payload)
            logger.info(f"Automatic mode cache migrated to: {preferred_path}")
    except Exception:
        # If migration fails, continue using legacy path to preserve behavior.
        return legacy_path

    return preferred_path


def get_auto_mode_cache_path() -> str:
    """Return active auto-mode cache file path."""
    return _auto_cache_path()


def _auto_program_version(base_data: dict | None) -> str:
    try:
        return str((base_data or {}).get("program_version", "") or "").strip()
    except Exception:
        return ""


def _auto_filter_cache_key(base_data: dict | None = None, *, filter_type: str | None = None) -> str:
    ft = str(
        filter_type
        if filter_type is not None
        else (base_data or {}).get("filter_type", "")
        or ""
    ).strip().lower()
    if ft in AUTO_MODE_CACHE_FILTER_KEYS:
        return str(ft)
    if "asym" in ft:
        return "asym"
    if "mixed" in ft:
        return "mixed"
    if "minimum" in ft or "minphase" in ft or ("min" in ft and "phase" in ft):
        return "minimum"
    if "linear" in ft:
        return "linear"
    return "mixed"


def _auto_cache_bucket_template() -> dict:
    return {
        "items": {},
        "target_by_measurement": {},
        "last_used_best": {},
    }


def _auto_cache_empty(*, program_version: str | None = None) -> dict:
    out = {
        "v": 3,
        "items": {},
        "target_by_measurement": {},
        "by_filter": {},
    }
    ver = str(program_version or "").strip()
    if ver:
        out["program_version"] = str(ver)
    for k in AUTO_MODE_CACHE_FILTER_KEYS:
        out["by_filter"][str(k)] = _auto_cache_bucket_template()
    return out


def _auto_cache_bucket(
    cache: dict,
    *,
    filter_key: str | None,
    create: bool = False,
) -> dict | None:
    if not isinstance(cache, dict):
        return None
    by_filter = cache.get("by_filter", {})
    if not isinstance(by_filter, dict):
        if not bool(create):
            return None
        by_filter = {}
        cache["by_filter"] = by_filter
    fk = _auto_filter_cache_key(filter_type=str(filter_key or ""))
    bucket = by_filter.get(fk)
    if not isinstance(bucket, dict):
        if not bool(create):
            return None
        bucket = _auto_cache_bucket_template()
        by_filter[fk] = bucket
    if bool(create):
        if not isinstance(bucket.get("items", {}), dict):
            bucket["items"] = {}
        if not isinstance(bucket.get("target_by_measurement", {}), dict):
            bucket["target_by_measurement"] = {}
        if not isinstance(bucket.get("last_used_best", {}), dict):
            bucket["last_used_best"] = {}
    return bucket


def _auto_goal_norm(goal: str | None) -> str:
    goal_norm = str(goal or AUTO_MODE_GOAL_DEFAULT).strip().lower()
    goal_aliases = {
        "c": AUTO_MODE_GOAL_FLAT,
        "acoustic": AUTO_MODE_GOAL_FLAT,
        "hybrid": AUTO_MODE_GOAL_LOW_RIPPLE,
        "room_safe": AUTO_MODE_GOAL_ROOM_SAFE,
        "roomsafe": AUTO_MODE_GOAL_ROOM_SAFE,
        "low_ripple": AUTO_MODE_GOAL_LOW_RIPPLE,
        "lowripple": AUTO_MODE_GOAL_LOW_RIPPLE,
    }
    goal_norm = str(goal_aliases.get(goal_norm, goal_norm))
    if goal_norm not in (
        AUTO_MODE_GOAL_DEFAULT,
        AUTO_MODE_GOAL_ROOM_SAFE,
        AUTO_MODE_GOAL_LOW_RIPPLE,
        AUTO_MODE_GOAL_FLAT,
    ):
        goal_norm = AUTO_MODE_GOAL_DEFAULT
    return str(goal_norm)


AUTO_MODE_PHASE1_PLATEAU_ROUNDS = 5
AUTO_MODE_PHASE2_PLATEAU_ROUNDS = 8
AUTO_MODE_TARGET_TOP_N = 3
AUTO_MODE_TARGET_TRIALS_PER_CURVE = 10
AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT = 0.75
AUTO_MODE_TARGET_TOP_N_MIN = 3
AUTO_MODE_TARGET_TOP_N_MAX = 6
AUTO_MODE_TARGET_TOP_N_SPREAD_DB = 0.35
AUTO_MODE_TARGET_PRESELECT_BOOST_W = 0.22
AUTO_MODE_TARGET_PRESELECT_SLOPE_W = 0.18
AUTO_MODE_TARGET_PRESELECT_ASYM_W = 0.30
AUTO_MODE_TARGET_PRESELECT_MODE_W = 0.16
AUTO_MODE_TARGET_PRESELECT_MAX_BASS_BOOST_REF_DB = 8.0
AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MIN_HZ = 25.0
AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MAX_HZ = 160.0
AUTO_MODE_TARGET_CACHE_AS_WILDCARD = True
AUTO_MODE_TARGET_PREFER_MILDER_STEP = True
AUTO_MODE_TARGET_MILDER_MAX_RANK_DROP = 1.50
AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB = 0.25
AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD = 0.20
AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD = 0.15
AUTO_MODE_TARGET_MILDER_MAX_AVG_SCORE_DROP_ACOUSTIC = 0.8
AUTO_MODE_TARGET_BEST_RANK_TIE_EPS = 0.05
AUTO_MODE_LOCAL_REFINE_ENABLED = True
AUTO_MODE_LOCAL_REFINE_TOP_K = 2
AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP = 12
AUTO_MODE_LOCAL_REFINE_SHRINK = 0.35
AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1 = True
AUTO_MODE_LOCAL_REFINEMENT_TOP_N = 3
AUTO_MODE_LOCAL_REFINEMENT_PER_ANCHOR = 12
AUTO_MODE_LOCAL_REFINEMENT_SHRINK = 0.35
AUTO_MODE_REFINE_TIEBREAK_ENABLE = True
AUTO_MODE_REFINE_TIEBREAK_RANK_EPS = 0.20
AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS = 0.02
AUTO_MODE_REFINE_MODE_SOFT_K = 0.25
AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB = 0.06

# --- Trial parallelism ---
# 0 workers in UI/config => auto (cpu_count), 1 => sequential.
AUTO_MODE_PARALLEL_ENABLED = True
AUTO_MODE_PARALLEL_MIN_TRIALS = 6
AUTO_MODE_PARALLEL_MAX_WORKERS = 0
AUTO_MODE_PARALLEL_BATCH_MULTIPLIER = 2
AUTO_MODE_OPTUNA_PILOT_ENABLED = True
AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS = 24
AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS = 12

# --- Dual-mode detection + mode-aware scoring ---
# Some rooms have two dominant LF resonances (e.g. 40-60 Hz + 90-130 Hz).
# Penalizing only the single worst mode can let the 2nd mode slip through.
AUTO_MODE_DUAL_MODE_ENABLED = True
AUTO_MODE_DUAL_MODE_TOP_N = 2

# How strongly mode-band ripple influences rank_score.
# mode_ripple_db values are typically small (e.g. 0.02 .. 0.15 dB).
AUTO_MODE_MODE_RIPPLE_OK_DB = 0.030
AUTO_MODE_MODE_RIPPLE_PENALTY_W = 30.0
AUTO_MODE_MODE_RIPPLE_SECONDARY_W = 0.85

AUTO_MODE_PHASE2_PARETO_POOL_MIN = 6
AUTO_MODE_PHASE2_PARETO_POOL_MAX = 15
AUTO_MODE_PHASE2_PARETO_RANK_WINDOW = 2.0
AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP = 0.35
AUTO_MODE_PHASE2_PARETO_PREPOST_EPS = 0.002
AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS = 0.005
AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS = 0.003
AUTO_MODE_PHASE2_PARETO_BOOST_EPS = 0.02

# --- Phase2 hard-gate (pre-Pareto) ---
# Removes obvious "bad actors" (event severity / ripple) from the phase2 kept pool
# before building a Pareto front. This improves consistency on hard data without
# adding extra trials.
AUTO_MODE_PHASE2_HARD_GATE_ENABLED = True
AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP = 8
AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION = 0.70
AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION = 0.75
AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK = True

# --- Adaptive search-space shrinking (phase2 local + phase3 micro) ---
# Smaller => tighter search around anchor(s). This is applied on top of fixed
# shrink constants and is derived from phase1 stability.
AUTO_MODE_ADAPTIVE_SHRINK_ENABLED = True
AUTO_MODE_ADAPTIVE_SHRINK_MIN = 0.20
AUTO_MODE_ADAPTIVE_SHRINK_MAX = 0.55
AUTO_MODE_PHASE3_MICRO_ENABLED = True
AUTO_MODE_PHASE3_MICRO_TRIALS = 12
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ = 110.0
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ = 40.0
AUTO_MODE_HYBRID_LOCAL_TOP_N = 2
AUTO_MODE_HYBRID_LOCAL_PER_ANCHOR = 10
# Rank score calibration (UI/report scale only).
# Example: raw 66.3 -> calibrated ~85.0 with bias 18.7.
AUTO_MODE_RANK_SCORE_GAIN = 1.0
AUTO_MODE_RANK_SCORE_BIAS = 18.7
# Event penalty confidence gate: lower confidence => softer event penalty.
AUTO_MODE_EVENT_PEN_CONF_GATE_ENABLE = True
AUTO_MODE_EVENT_PEN_CONF_GATE_MIN_SCALE = 0.45
AUTO_MODE_EVENT_PEN_CONF_GATE_FULL_CONF = 0.80
AUTO_MODE_EVENT_PEN_BASE_PER_EVENT = 0.5
AUTO_MODE_EVENT_PEN_DT_WEIGHT = 0.02
AUTO_MODE_EVENT_PEN_DT_POWER = 2.0
AUTO_MODE_EVENT_PEN_DT_REF_MS = 100.0
AUTO_MODE_MAG_C_MIN_MIN_HZ = 15.0
AUTO_MODE_MAG_C_MIN_MAX_HZ = 70.0
AUTO_MODE_MAG_C_MIN_REF_MIN_HZ = 80.0
AUTO_MODE_MAG_C_MIN_REF_MAX_HZ = 200.0
AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ = 80.0
AUTO_MODE_MAG_C_MIN_SMOOTH_OCT = 1.0
AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ = 2.0
AUTO_MODE_LOW_BASS_MIN_HZ = 18.0
AUTO_MODE_LOW_BASS_MAX_HZ = 55.0
AUTO_MODE_EXC_FROM_F6_ADD_HZ = 8.0
AUTO_MODE_EXC_MIN_HZ = 20.0
AUTO_MODE_EXC_MAX_HZ = 80.0
AUTO_MODE_PHASE_LIMIT_MIN_HZ = 150.0
AUTO_MODE_PHASE_LIMIT_MAX_HZ = 500.0
AUTO_MODE_PHASE_LIMIT_SIGMA_HZ = 20.0
AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ = 30.0
AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ = 320.0
AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ = 300.0
AUTO_MODE_PHASE_LIMIT_PRIOR_TOL_HZ = 90.0
AUTO_MODE_PHASE_LIMIT_PRIOR_SPAN_HZ = 70.0
AUTO_MODE_PHASE_LIMIT_PRIOR_WEIGHT = 1.2
AUTO_MODE_PHASE_LIMIT_PRIOR_MAX_PEN = 4.0
AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_FRAC = 0.35
AUTO_MODE_PHASE_LIMIT_EXPLORE_UNIFORM_FRAC = 0.20
AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ = 90.0
AUTO_MODE_HPF_MIN_HZ = 14.0
AUTO_MODE_HPF_MAX_HZ = 140.0
AUTO_MODE_HPF_REF_MIN_HZ = 90.0
AUTO_MODE_HPF_REF_MAX_HZ = 260.0
AUTO_MODE_HPF_SEARCH_MAX_HZ = 180.0
AUTO_MODE_HPF_SMOOTH_OCT = 1.00 #0.67
AUTO_MODE_HPF_AUTO_ENABLE_MIN_CONF = 0.45
AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT = (6, 12, 18, 24, 30, 36, 42, 48, 54)
AUTO_MODE_BUILTIN_TARGETS = (
    "Harman6",
    "Harman8",
    "Harman4",
    "Harman10",
    "Harman12",
    "Studio",
    "Nearfield",
    "HiFi",
    "Speech",
    "Toole",
    "BK_Light",
    "BK_Medium",
    "BK_Strong",
    "Flat",
    "Cinema",
)
AUTO_MODE_BUILTIN_TARGET_LOOKUP = {
    str(name).strip().lower(): str(name).strip()
    for name in AUTO_MODE_BUILTIN_TARGETS
}


def _auto_builtin_target_name(hc_mode: str | None) -> str | None:
    """Return canonical built-in target name or None for non built-ins."""
    key = str(hc_mode or "").strip().lower()
    if not key:
        return None
    return AUTO_MODE_BUILTIN_TARGET_LOOKUP.get(key)


def _auto_hash_array(a: np.ndarray, *, decimals: int = 4, max_len: int = 1200) -> str:
    """
    Stable-ish hash for numeric arrays:
      - flatten
      - drop non-finite
      - downsample to max_len points
      - round
      - sha256
    """
    try:
        x = np.asarray(a, dtype=float).reshape(-1)
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    m = np.isfinite(x)
    x = x[m]
    if x.size <= 0:
        return ""
    if x.size > int(max_len):
        idx = np.linspace(0, x.size - 1, int(max_len)).astype(int)
        x = x[idx]
    x = np.round(x, int(decimals))
    b = x.astype(np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()


def _auto_measurement_signature(measurements: dict) -> str:
    fL = measurements.get("f_l")
    mL = measurements.get("m_l")
    fR = measurements.get("f_r")
    mR = measurements.get("m_r")
    h = hashlib.sha256()
    h.update(_auto_hash_array(np.asarray(fL) if fL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(mL) if mL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(fR) if fR is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(mR) if mR is not None else np.asarray([])).encode("ascii", "ignore"))
    return h.hexdigest()


def _auto_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> str:
    """
    Signature for caching:
      - measurement response (f/m arrays L+R)
      - key settings that affect search space and result
    """
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    h = hashlib.sha256()
    h.update(_auto_measurement_signature(measurements).encode("ascii", "ignore"))
    keys = {
        "fs": int(fs_v),
        "taps": int(taps_v),
        "filter_type": ft,
        "auto_goal": str(_auto_goal(base_data)),
        "enable_tdc": bool(base_data.get("enable_tdc", True)),
        "enable_afdw": bool(base_data.get("enable_afdw", True)),
        "bass_first_ai": bool(base_data.get("bass_first_ai", True)),
        "mag_c_max": float(_auto_safe_float(base_data.get("mag_c_max", 250.0), 250.0)),
        "_auto_mag_c_min_hz": float(_auto_safe_float(base_data.get("_auto_mag_c_min_hz", float("nan")), float("nan"))),
        "_auto_low_bass_cut_hz": float(_auto_safe_float(base_data.get("_auto_low_bass_cut_hz", float("nan")), float("nan"))),
        "_auto_exc_freq_hz": float(_auto_safe_float(base_data.get("_auto_exc_freq_hz", float("nan")), float("nan"))),
        "xos": xos if isinstance(xos, list) else [],
        "hpf": hpf if isinstance(hpf, dict) or hpf is None else str(hpf),
    }
    if bool(include_hc_mode):
        keys["hc_mode"] = str(hc_mode or base_data.get("hc_mode", "") or "").strip()
    try:
        h.update(json.dumps(keys, sort_keys=True, default=str).encode("utf-8"))
    except Exception:
        h.update(str(keys).encode("utf-8", "ignore"))
    return h.hexdigest()


def _auto_seed_from_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> int:
    """
    Deterministic RNG seed for auto-mode trials.
    Same measurements + same key settings => same seed => reproducible trials/results.
    """
    try:
        sig = _auto_signature(
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=hc_mode,
            include_hc_mode=bool(include_hc_mode),
        )
        if not sig:
            raise ValueError("empty signature")
        # Use first 8 hex chars => 32-bit seed.
        return int(str(sig)[:8], 16) & 0xFFFFFFFF
    except Exception:
        # Fallback: measurement signature only.
        try:
            msig = _auto_measurement_signature(measurements or {})
            return int(str(msig)[:8], 16) & 0xFFFFFFFF if msig else 0
        except Exception:
            return 0


def _auto_apply_seed(seed: int) -> None:
    """
    Apply deterministic seeding for both numpy Generator usage and any legacy
    random/np.random usage elsewhere.
    """
    try:
        s = int(seed) & 0xFFFFFFFF
    except Exception:
        s = 0
    try:
        random.seed(s)
    except Exception:
        pass
    try:
        np.random.seed(s)
    except Exception:
        pass


def _auto_cache_load(*, program_version: str | None = None) -> dict:
    global _AUTO_CACHE_VERSION_MISMATCH_LOGGED
    path = _auto_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            obj = {}
        expected_ver = str(program_version or "").strip()
        if not expected_ver:
            return obj
        cached_ver = str(obj.get("program_version", "") or "").strip()
        if cached_ver == expected_ver:
            return obj
        if not bool(_AUTO_CACHE_VERSION_MISMATCH_LOGGED):
            logger.info(
                "Automatic mode cache version mismatch: "
                f"cached='{cached_ver or 'n/a'}', current='{expected_ver}'. "
                "Ignoring cache and running fresh trials."
            )
            _AUTO_CACHE_VERSION_MISMATCH_LOGGED = True
        return _auto_cache_empty(program_version=expected_ver)
    except Exception:
        expected_ver = str(program_version or "").strip()
        if expected_ver:
            return _auto_cache_empty(program_version=expected_ver)
        return {}


def _auto_cache_save(cache: dict, *, program_version: str | None = None) -> None:
    path = _auto_cache_path()
    try:
        cache_obj = dict(cache or {})
        try:
            cache_obj["v"] = int(max(3, int(cache_obj.get("v", 0) or 0)))
        except Exception:
            cache_obj["v"] = 3
        by_filter = cache_obj.get("by_filter", {})
        if not isinstance(by_filter, dict):
            by_filter = {}
        for k in AUTO_MODE_CACHE_FILTER_KEYS:
            if not isinstance(by_filter.get(k), dict):
                by_filter[k] = _auto_cache_bucket_template()
            else:
                if not isinstance(by_filter[k].get("items", {}), dict):
                    by_filter[k]["items"] = {}
                if not isinstance(by_filter[k].get("target_by_measurement", {}), dict):
                    by_filter[k]["target_by_measurement"] = {}
                if not isinstance(by_filter[k].get("last_used_best", {}), dict):
                    by_filter[k]["last_used_best"] = {}
        cache_obj["by_filter"] = by_filter
        ver = str(program_version or cache_obj.get("program_version", "") or "").strip()
        if ver:
            cache_obj["program_version"] = str(ver)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache_obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        # best-effort only
        return


def _auto_cache_get_entry(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    if not sig:
        return None
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    bucket_items = {}
    if isinstance(bucket, dict):
        raw_bucket_items = bucket.get("items", {})
        if isinstance(raw_bucket_items, dict):
            bucket_items = raw_bucket_items
    if isinstance(bucket_items, dict):
        entry = bucket_items.get(sig)
        if isinstance(entry, dict):
            return dict(entry)
        if len(bucket_items) > 0:
            return None
    # Legacy fallback: old cache file without filter buckets.
    items = cache.get("items", {})
    if isinstance(items, dict):
        entry = items.get(sig)
        return dict(entry) if isinstance(entry, dict) else None
    return None


def _auto_cache_get_best(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    entry = _auto_cache_get_entry(
        sig,
        filter_key=filter_key,
        program_version=program_version,
    )
    if not isinstance(entry, dict):
        return None
    preset = entry.get("best_preset")
    return dict(preset) if isinstance(preset, dict) else None


def _auto_cache_get_best_target(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> str | None:
    entry = _auto_cache_get_entry(
        sig,
        filter_key=filter_key,
        program_version=program_version,
    )
    if not isinstance(entry, dict):
        return None
    hc = str(entry.get("best_target_curve", entry.get("best_hc_mode", "")) or "").strip()
    return _auto_builtin_target_name(hc)


def _auto_cache_get_target_for_measurements(
    measurements: dict,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return None
    cache = _auto_cache_load(program_version=program_version)

    # Preferred map: target selected by measurement signature.
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    target_map = {}
    if isinstance(bucket, dict):
        raw_target_map = bucket.get("target_by_measurement", {})
        if isinstance(raw_target_map, dict):
            target_map = raw_target_map
    if isinstance(target_map, dict):
        direct = target_map.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)
        if len(target_map) > 0:
            return None
    # Legacy fallback: old cache file without filter buckets.
    target_map_legacy = cache.get("target_by_measurement", {})
    if isinstance(target_map_legacy, dict):
        direct = target_map_legacy.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map_legacy.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)

    # Backward compatibility: check legacy item entries if they carry measurement_sig.
    items = {}
    if isinstance(bucket, dict):
        raw_items = bucket.get("items", {})
        if isinstance(raw_items, dict):
            items = raw_items
    if not items:
        items = cache.get("items", {})
    if not isinstance(items, dict):
        return None
    best = None
    best_t = -1
    for entry in items.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("measurement_sig", "") or "") != str(msig):
            continue
        entry_goal = _auto_goal_norm(str(entry.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
        if entry_goal != goal_norm:
            continue
        try:
            t = int(entry.get("t", 0) or 0)
        except Exception:
            t = 0
        if t >= best_t:
            best_t = int(t)
            best = dict(entry)
    return dict(best) if isinstance(best, dict) else None


def _auto_cache_put_target_for_measurements(
    *,
    measurements: dict,
    best_hc_mode: str | None,
    best_preset: dict,
    best_metrics: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    hc_val = str(best_hc_mode or "").strip()
    if not hc_val:
        return
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    target_map = bucket.get("target_by_measurement", {})
    if not isinstance(target_map, dict):
        target_map = {}
    goal_norm = _auto_goal_norm(goal)
    scoped_key = f"{msig}|{goal_norm}"
    target_map[str(scoped_key)] = {
        "t": int(time.time()),
        "measurement_sig": str(msig),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_target_curve": hc_val,
        "best_hc_mode": hc_val,
        "best_preset": dict(best_preset or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    try:
        if len(target_map) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                target_map.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            target_map = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        pass
    bucket["target_by_measurement"] = target_map
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)


def _auto_cache_put_best(
    sig: str,
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    if not sig or not isinstance(best_preset, dict):
        return
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    items = bucket.get("items", {})
    if not isinstance(items, dict):
        items = {}
    entry = {
        "t": int(time.time()),
        "auto_goal": str(_auto_goal_norm(goal)),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_preset": dict(best_preset),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
    items[str(sig)] = entry
    try:
        if len(items) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                items.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            items = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        pass
    bucket["items"] = items
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)


def _auto_cache_get_last_used_best(
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    last_map = {}
    if isinstance(bucket, dict):
        raw_last_map = bucket.get("last_used_best", {})
        if isinstance(raw_last_map, dict):
            last_map = raw_last_map
    if isinstance(last_map, dict):
        direct = last_map.get(str(goal_norm))
        if isinstance(direct, dict):
            return dict(direct)
        if len(last_map) > 0:
            return None
    # Legacy fallback: global map if older code used it.
    legacy_map = cache.get("last_used_best", {})
    if not isinstance(legacy_map, dict):
        return None
    direct = legacy_map.get(str(goal_norm))
    return dict(direct) if isinstance(direct, dict) else None


def _auto_cache_put_last_used_best(
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    if not isinstance(best_preset, dict) or not best_preset:
        return
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    last_map = bucket.get("last_used_best", {})
    if not isinstance(last_map, dict):
        last_map = {}
    entry = {
        "t": int(time.time()),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_preset": dict(best_preset or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
    last_map[str(goal_norm)] = entry
    bucket["last_used_best"] = last_map
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)


def _auto_safe_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _auto_safe_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        s = str(value or "").strip().lower()
    except Exception:
        return bool(default)
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _auto_safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _auto_optimizer_backend(base_data: dict | None, *, default_optuna_enabled: bool = False) -> str:
    env_raw = str(os.environ.get("CAMILLAFIR_AUTO_MODE_OPTIMIZER", "") or "").strip().lower()
    if env_raw in ("builtin", "optuna"):
        return str(env_raw)

    data = dict(base_data or {})
    raw = str(data.get("auto_mode_optimizer", "") or "").strip().lower()
    if raw in ("builtin", "optuna"):
        return str(raw)

    if _auto_safe_bool(data.get("auto_mode_optuna", default_optuna_enabled), default_optuna_enabled):
        return "optuna"
    return "builtin"


@dataclass(frozen=True)
class AutoModeConfig:
    trials: int = AUTO_MODE_TRIALS
    refine_trials: int = AUTO_MODE_REFINE_TRIALS
    phase1_plateau_rounds: int = AUTO_MODE_PHASE1_PLATEAU_ROUNDS
    local_refine_enabled: bool = AUTO_MODE_LOCAL_REFINE_ENABLED
    local_refine_top_k: int = AUTO_MODE_LOCAL_REFINE_TOP_K
    local_refine_trials_per_top: int = AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP
    local_refine_shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK
    local_refine_keep_best_phase1: bool = AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1
    phase3_micro_enabled: bool = AUTO_MODE_PHASE3_MICRO_ENABLED
    phase3_micro_trials: int = AUTO_MODE_PHASE3_MICRO_TRIALS
    adaptive_shrink_max: float = AUTO_MODE_ADAPTIVE_SHRINK_MAX
    phase2_pareto_pool_min: int = AUTO_MODE_PHASE2_PARETO_POOL_MIN
    phase2_pareto_pool_max: int = AUTO_MODE_PHASE2_PARETO_POOL_MAX
    phase2_pareto_rank_window: float = AUTO_MODE_PHASE2_PARETO_RANK_WINDOW
    phase2_pareto_acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP
    phase2_hard_gate_enabled: bool = AUTO_MODE_PHASE2_HARD_GATE_ENABLED
    phase2_hard_gate_min_keep: int = AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP
    phase2_hard_gate_keep_event_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION
    phase2_hard_gate_keep_ripple_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION
    phase2_hard_gate_fallback_to_rank: bool = AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK
    refine_mode_soft_k: float = AUTO_MODE_REFINE_MODE_SOFT_K
    refine_tiebreak_rank_eps: float = AUTO_MODE_REFINE_TIEBREAK_RANK_EPS
    exc_min_hz: float = AUTO_MODE_EXC_MIN_HZ
    exc_max_hz: float = AUTO_MODE_EXC_MAX_HZ
    cache_enabled: bool = AUTO_MODE_CACHE_ENABLED
    optuna_pilot_enabled: bool = AUTO_MODE_OPTUNA_PILOT_ENABLED
    optuna_pilot_min_trials: int = AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS
    optuna_pilot_startup_trials: int = AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS

    @classmethod
    def from_base_data(cls, base_data: dict | None) -> "AutoModeConfig":
        data = dict(base_data or {})
        return cls(
            trials=max(1, _auto_safe_int(data.get("auto_mode_trials", AUTO_MODE_TRIALS), AUTO_MODE_TRIALS)),
            refine_trials=max(1, _auto_safe_int(data.get("auto_mode_refine_trials", AUTO_MODE_REFINE_TRIALS), AUTO_MODE_REFINE_TRIALS)),
            phase1_plateau_rounds=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_phase1_plateau_rounds", AUTO_MODE_PHASE1_PLATEAU_ROUNDS),
                    AUTO_MODE_PHASE1_PLATEAU_ROUNDS,
                ),
            ),
            local_refine_enabled=_auto_safe_bool(
                data.get("auto_mode_local_refine_enabled", AUTO_MODE_LOCAL_REFINE_ENABLED),
                AUTO_MODE_LOCAL_REFINE_ENABLED,
            ),
            local_refine_top_k=max(
                1,
                _auto_safe_int(data.get("auto_mode_local_refine_top_k", AUTO_MODE_LOCAL_REFINE_TOP_K), AUTO_MODE_LOCAL_REFINE_TOP_K),
            ),
            local_refine_trials_per_top=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_local_refine_trials_per_top", AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP),
                    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
                ),
            ),
            local_refine_shrink=float(
                np.clip(
                    _auto_safe_float(data.get("auto_mode_local_refine_shrink", AUTO_MODE_LOCAL_REFINE_SHRINK), AUTO_MODE_LOCAL_REFINE_SHRINK),
                    0.05,
                    1.50,
                )
            ),
            local_refine_keep_best_phase1=_auto_safe_bool(
                data.get("auto_mode_local_refine_keep_phase1", AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1),
                AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1,
            ),
            phase3_micro_enabled=_auto_safe_bool(
                data.get("auto_mode_phase3_micro_enabled", AUTO_MODE_PHASE3_MICRO_ENABLED),
                AUTO_MODE_PHASE3_MICRO_ENABLED,
            ),
            phase3_micro_trials=max(
                1,
                _auto_safe_int(data.get("auto_mode_phase3_micro_trials", AUTO_MODE_PHASE3_MICRO_TRIALS), AUTO_MODE_PHASE3_MICRO_TRIALS),
            ),
            adaptive_shrink_max=float(
                np.clip(
                    _auto_safe_float(data.get("auto_mode_adaptive_shrink_max", AUTO_MODE_ADAPTIVE_SHRINK_MAX), AUTO_MODE_ADAPTIVE_SHRINK_MAX),
                    0.05,
                    1.0,
                )
            ),
            phase2_pareto_pool_min=max(
                1,
                _auto_safe_int(data.get("auto_mode_phase2_pareto_pool_min", AUTO_MODE_PHASE2_PARETO_POOL_MIN), AUTO_MODE_PHASE2_PARETO_POOL_MIN),
            ),
            phase2_pareto_pool_max=max(
                1,
                _auto_safe_int(data.get("auto_mode_phase2_pareto_pool_max", AUTO_MODE_PHASE2_PARETO_POOL_MAX), AUTO_MODE_PHASE2_PARETO_POOL_MAX),
            ),
            phase2_pareto_rank_window=max(
                0.0,
                _auto_safe_float(
                    data.get("auto_mode_phase2_pareto_rank_window", AUTO_MODE_PHASE2_PARETO_RANK_WINDOW),
                    AUTO_MODE_PHASE2_PARETO_RANK_WINDOW,
                ),
            ),
            phase2_pareto_acoustic_drop=max(
                0.0,
                _auto_safe_float(
                    data.get("auto_mode_phase2_pareto_acoustic_drop", AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP),
                    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
                ),
            ),
            phase2_hard_gate_enabled=_auto_safe_bool(
                data.get("auto_mode_phase2_hard_gate_enabled", AUTO_MODE_PHASE2_HARD_GATE_ENABLED),
                AUTO_MODE_PHASE2_HARD_GATE_ENABLED,
            ),
            phase2_hard_gate_min_keep=max(
                1,
                _auto_safe_int(data.get("auto_mode_phase2_hard_gate_min_keep", AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP), AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP),
            ),
            phase2_hard_gate_keep_event_fraction=float(
                np.clip(
                    _auto_safe_float(
                        data.get("auto_mode_phase2_hard_gate_keep_event_fraction", AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION),
                        AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_hard_gate_keep_ripple_fraction=float(
                np.clip(
                    _auto_safe_float(
                        data.get("auto_mode_phase2_hard_gate_keep_ripple_fraction", AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION),
                        AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_hard_gate_fallback_to_rank=_auto_safe_bool(
                data.get("auto_mode_phase2_hard_gate_fallback_to_rank", AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK),
                AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
            ),
            refine_mode_soft_k=max(
                0.0,
                _auto_safe_float(data.get("auto_mode_refine_mode_soft_k", AUTO_MODE_REFINE_MODE_SOFT_K), AUTO_MODE_REFINE_MODE_SOFT_K),
            ),
            refine_tiebreak_rank_eps=max(
                0.0,
                _auto_safe_float(
                    data.get("auto_mode_refine_tiebreak_rank_eps", AUTO_MODE_REFINE_TIEBREAK_RANK_EPS),
                    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
                ),
            ),
            exc_min_hz=max(
                1.0,
                _auto_safe_float(data.get("auto_mode_exc_min_hz", AUTO_MODE_EXC_MIN_HZ), AUTO_MODE_EXC_MIN_HZ),
            ),
            exc_max_hz=max(
                1.0,
                _auto_safe_float(data.get("auto_mode_exc_max_hz", AUTO_MODE_EXC_MAX_HZ), AUTO_MODE_EXC_MAX_HZ),
            ),
            cache_enabled=_auto_safe_bool(
                data.get("auto_mode_cache_enabled", AUTO_MODE_CACHE_ENABLED),
                AUTO_MODE_CACHE_ENABLED,
            ),
            optuna_pilot_enabled=_auto_safe_bool(
                data.get("auto_mode_optuna", AUTO_MODE_OPTUNA_PILOT_ENABLED),
                AUTO_MODE_OPTUNA_PILOT_ENABLED,
            ),
            optuna_pilot_min_trials=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_optuna_min_trials", AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS),
                    AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS,
                ),
            ),
            optuna_pilot_startup_trials=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_optuna_startup_trials", AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS),
                    AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
                ),
            ),
        )

    def refine_trial_hint(self, goal: str | None) -> int:
        goal_norm = _auto_goal_norm(goal)
        hint = int(max(1, self.refine_trials))
        if bool(self.local_refine_enabled) and goal_norm in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
            AUTO_MODE_GOAL_ACOUSTIC,
            AUTO_MODE_GOAL_HYBRID,
        ):
            hint = int(max(1, self.local_refine_top_k) * max(1, self.local_refine_trials_per_top))
        return int(max(1, hint))

def _auto_trial_workers(base_data: dict | None, n_trials: int) -> int:
    if (not bool(AUTO_MODE_PARALLEL_ENABLED)) or int(n_trials) < int(AUTO_MODE_PARALLEL_MIN_TRIALS):
        return 1
    cpu_n = int(max(1, _auto_safe_int(os.cpu_count(), 1)))
    env_raw = os.environ.get("CAMILLAFIR_AUTO_MODE_WORKERS", "").strip()
    req = _auto_safe_int((base_data or {}).get("auto_mode_workers", 0), 0)
    if env_raw:
        req = _auto_safe_int(env_raw, req)
    if req <= 0:
        req = int(cpu_n)
    hard_max = int(max(0, _auto_safe_int(AUTO_MODE_PARALLEL_MAX_WORKERS, 0)))
    if hard_max > 0:
        req = min(req, hard_max)
    req = int(max(1, min(int(req), int(cpu_n), int(max(1, n_trials)))))
    return int(req)


def _auto_trial_chunk_size(workers: int) -> int:
    w = int(max(1, _auto_safe_int(workers, 1)))
    mul = int(max(1, _auto_safe_int(AUTO_MODE_PARALLEL_BATCH_MULTIPLIER, 2)))
    return int(max(w, w * mul))


def _clip(v, lo, hi):
    vlo = _auto_safe_float(lo, 0.0)
    vhi = _auto_safe_float(hi, vlo)
    if vhi < vlo:
        vlo, vhi = vhi, vlo
    return float(np.clip(_auto_safe_float(v, vlo), vlo, vhi))


def _auto_is_phase_search_filter(filter_type: str | None) -> bool:
    fk = str(_auto_filter_cache_key(filter_type=filter_type))
    return fk in ("linear", "asym")


def _auto_phase_limit_clip(value, *, default: float = 400.0) -> float:
    v = _auto_safe_float(value, float("nan"))
    if not np.isfinite(v):
        v = _auto_safe_float(default, 400.0)
    return _clip(v, float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ))


def _auto_phase_limit_center(value, *, default: float | None = None) -> float:
    v = _auto_safe_float(value, float("nan"))
    lo = float(AUTO_MODE_PHASE_LIMIT_MIN_HZ)
    hi = float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)
    if np.isfinite(v) and (lo <= float(v) <= hi):
        return float(v)
    d = _auto_safe_float(
        AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ if default is None else default,
        AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ,
    )
    return float(_clip(d, lo, hi))


def _auto_phase_limit_prior_penalty(phase_limit_hz: float, *, filter_key: str | None) -> float:
    if not _auto_is_phase_search_filter(filter_key):
        return 0.0
    pl = _auto_safe_float(phase_limit_hz, float("nan"))
    if not np.isfinite(pl):
        return 0.0
    center = float(
        _clip(
            AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ,
            AUTO_MODE_PHASE_LIMIT_MIN_HZ,
            AUTO_MODE_PHASE_LIMIT_MAX_HZ,
        )
    )
    tol = float(max(1.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_TOL_HZ, 90.0)))
    span = float(max(1.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_SPAN_HZ, 70.0)))
    w = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_WEIGHT, 1.2)))
    max_pen = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_MAX_PEN, 4.0)))
    excess = max(0.0, abs(float(pl) - center) - tol)
    pen = float(w) * ((float(excess) / float(span)) ** 2.0)
    return float(min(max_pen, max(0.0, pen)))


def _jitter(rng, v, sigma, lo, hi, *, base_data: dict | None = None, key: str | None = None, default=None):
    center = _auto_safe_float(v, float("nan"))
    if not np.isfinite(center):
        if key and isinstance(base_data, dict):
            center = _auto_safe_float(base_data.get(key, default), float("nan"))
        if not np.isfinite(center):
            if default is not None:
                center = _auto_safe_float(default, float("nan"))
            if not np.isfinite(center):
                center = 0.5 * (_auto_safe_float(lo, 0.0) + _auto_safe_float(hi, 0.0))
    sig = max(0.0, _auto_safe_float(sigma, 0.0))
    if sig <= 0.0:
        return _clip(center, lo, hi)
    try:
        x = float(rng.normal(loc=float(center), scale=float(sig)))
    except Exception:
        x = float(center)
    return _clip(x, lo, hi)


def _auto_sample_mag_low_pair(
    rng,
    *,
    mag_center: float,
    low_center: float,
    mag_sigma: float,
    low_sigma: float,
) -> tuple[float, float]:
    mag = float(
        _jitter(
            rng,
            mag_center,
            mag_sigma,
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            default=mag_center,
        )
    )
    low = float(
        _jitter(
            rng,
            low_center,
            low_sigma,
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
            default=low_center,
        )
    )
    # Keep low-bass policy meaningful: when possible, don't place it below mag_c_min.
    if np.isfinite(mag) and np.isfinite(low) and float(mag) <= float(AUTO_MODE_LOW_BASS_MAX_HZ):
        low = float(max(float(low), float(mag)))
    mag = float(
        _clip(
            mag,
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low = float(
        _clip(
            low,
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )
    return float(round(mag, 1)), float(round(low, 1))


def _auto_goal(base_data: dict | None, default: str = AUTO_MODE_GOAL_DEFAULT) -> str:
    g = str((base_data or {}).get("auto_goal", default) or default).strip().lower()
    return str(_auto_goal_norm(g))


def _auto_goal_basis_text(goal: str) -> str:
    return "rank_score"


def _auto_metric_text(metrics: dict | None, goal: str) -> str:
    m = dict(metrics or {})
    return f"rank={_auto_safe_float(m.get('rank_score'), 0.0):.3f}"


def _auto_target_one_step_milder(hc_name: str) -> str | None:
    name = str(hc_name or "").strip()
    if not name:
        return None
    ladders = (
        ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
        ("BK_Light", "BK_Medium", "BK_Strong"),
    )
    for ladder in ladders:
        if name not in ladder:
            continue
        idx = int(ladder.index(name))
        if idx <= 0:
            return None
        return str(ladder[idx - 1])
    return None


def _auto_collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []


def _auto_get_worst_mode_hz(result) -> float | None:
    def _pick_ch_worst(st: dict | None) -> dict | None:
        refs = _auto_collect_reflections(st)
        picks = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            typ = str(r.get("type", "") or "").strip().lower()
            if typ != "resonance":
                continue
            freq = _auto_safe_float(r.get("freq", float("nan")), float("nan"))
            dt_ms = _auto_safe_float(r.get("gd_error", r.get("error_ms", float("nan"))), float("nan"))
            if not (np.isfinite(freq) and np.isfinite(dt_ms)):
                continue
            if float(dt_ms) < 180.0:
                continue
            if not (35.0 <= float(freq) <= 200.0):
                continue
            picks.append({"freq": float(freq), "dt_ms": float(dt_ms)})
        if not picks:
            return None
        return dict(max(picks, key=lambda x: float(x.get("dt_ms", 0.0))))

    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    lw = _pick_ch_worst(l_st)
    rw = _pick_ch_worst(r_st)
    if isinstance(lw, dict) and isinstance(rw, dict):
        f_l = float(_auto_safe_float(lw.get("freq", float("nan")), float("nan")))
        f_r = float(_auto_safe_float(rw.get("freq", float("nan")), float("nan")))
        dt_l = float(_auto_safe_float(lw.get("dt_ms", 0.0), 0.0))
        dt_r = float(_auto_safe_float(rw.get("dt_ms", 0.0), 0.0))
        if np.isfinite(f_l) and np.isfinite(f_r) and f_l > 0.0 and f_r > 0.0:
            # "Close enough": quarter-octave band match.
            if abs(float(np.log2(f_l / f_r))) <= 0.25:
                return float(0.5 * (f_l + f_r))
        return float(f_l if dt_l >= dt_r else f_r)
    if isinstance(lw, dict):
        return float(_auto_safe_float(lw.get("freq", float("nan")), float("nan")))
    if isinstance(rw, dict):
        return float(_auto_safe_float(rw.get("freq", float("nan")), float("nan")))
    return None


def _auto_get_top_modes_hz(result, *, top_n: int = 2) -> list[float]:
    """
    Return up to N dominant resonance frequencies (Hz) across L/R.

    Uses the same resonance constraints as _auto_get_worst_mode_hz, but returns
    multiple unique modes (merge close modes within ~quarter-octave).
    """
    n = int(max(1, top_n))

    def _collect(st: dict | None) -> list[dict]:
        refs = _auto_collect_reflections(st)
        out = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            typ = str(r.get("type", "") or "").strip().lower()
            if typ != "resonance":
                continue
            freq = _auto_safe_float(r.get("freq", float("nan")), float("nan"))
            dt_ms = _auto_safe_float(
                r.get("gd_error", r.get("error_ms", float("nan"))),
                float("nan"),
            )
            if not (np.isfinite(freq) and np.isfinite(dt_ms)):
                continue
            if float(dt_ms) < 180.0:
                continue
            if not (35.0 <= float(freq) <= 200.0):
                continue
            out.append({"freq": float(freq), "dt_ms": float(dt_ms)})
        return out

    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    picks = _collect(l_st) + _collect(r_st)
    if not picks:
        return []

    picks = sorted(picks, key=lambda x: float(x.get("dt_ms", 0.0)), reverse=True)

    # Merge close modes (quarter-octave) and keep strongest dt_ms representative.
    out: list[dict] = []
    for p in picks:
        f = float(_auto_safe_float(p.get("freq", float("nan")), float("nan")))
        dt = float(_auto_safe_float(p.get("dt_ms", 0.0), 0.0))
        if not (np.isfinite(f) and f > 0.0):
            continue
        merged = False
        for q in out:
            fq = float(_auto_safe_float(q.get("freq", float("nan")), float("nan")))
            if not (np.isfinite(fq) and fq > 0.0):
                continue
            if abs(float(np.log2(f / fq))) <= 0.25:
                # Merge by averaging freq, keep max dt.
                q["freq"] = float(0.5 * (fq + f))
                q["dt_ms"] = float(max(float(_auto_safe_float(q.get("dt_ms", 0.0), 0.0)), dt))
                merged = True
                break
        if not merged:
            out.append({"freq": float(f), "dt_ms": float(dt)})
        if len(out) >= n:
            break

    freqs = []
    for q in out:
        f = float(_auto_safe_float(q.get("freq", float("nan")), float("nan")))
        if np.isfinite(f) and f > 0.0:
            freqs.append(float(f))
    return freqs


def _auto_mode_band(f0_hz: float, base_data: dict | None = None) -> tuple[float, float] | None:
    f0 = _auto_safe_float(f0_hz, float("nan"))
    if not np.isfinite(f0) or f0 <= 0.0:
        return None
    lo = float(f0) / 1.35
    hi = float(f0) * 1.35
    lo = float(np.clip(lo, 30.0, 160.0))
    hi = float(np.clip(hi, 60.0, 220.0))
    if bool((base_data or {}).get("bass_first_ai", True)):
        bf_hi = _auto_safe_float((base_data or {}).get("bass_first_mode_max_hz", float("nan")), float("nan"))
        if np.isfinite(bf_hi):
            hi = min(float(hi), float(bf_hi))
    if hi <= lo:
        lo = float(np.clip(min(float(lo), float(hi) - 5.0), 30.0, 160.0))
    if hi <= lo:
        hi = float(np.clip(float(lo) + 5.0, 60.0, 220.0))
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _auto_event_severity(refs: list | None) -> float:
    refs = refs or []
    if not isinstance(refs, list) or not refs:
        return 0.0

    vals = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        v = _auto_safe_float(r.get("gd_error", 0.0), 0.0)
        if np.isfinite(v):
            vals.append(abs(float(v)))
    if not vals:
        return 0.0

    vals = sorted(vals, reverse=True)[:5]
    weights = (1.00, 0.75, 0.55, 0.40, 0.30)
    sev = 0.0
    for i, v in enumerate(vals):
        # Ignore very small GD irregularities; focus on meaningful events.
        sev += float(weights[i]) * max(0.0, float(v) - 2.0)
    return float(max(0.0, sev))


def _auto_event_penalty_weighted(
    events: list | None,
    *,
    base_per_event: float = AUTO_MODE_EVENT_PEN_BASE_PER_EVENT,
    dt_weight: float = AUTO_MODE_EVENT_PEN_DT_WEIGHT,
    power: float = AUTO_MODE_EVENT_PEN_DT_POWER,
    dt_ref_ms: float = AUTO_MODE_EVENT_PEN_DT_REF_MS,
) -> float:
    """
    Severity-weighted event penalty.
    Keeps base event-count behaviour and adds extra penalty for long-decay events.
    """
    ev = events or []
    if not isinstance(ev, list) or not ev:
        return 0.0

    base_per_event_f = max(0.0, float(_auto_safe_float(base_per_event, 0.5)))
    dt_weight_f = max(0.0, float(_auto_safe_float(dt_weight, 0.02)))
    power_f = max(1.0, float(_auto_safe_float(power, 2.0)))
    dt_ref_ms_f = max(1e-6, float(_auto_safe_float(dt_ref_ms, 100.0)))

    dt_list = []
    for e in ev:
        dt = None
        if isinstance(e, dict):
            dt = e.get("dt_ms", e.get("gd_error", e.get("error_ms", None)))
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            dt = e[2]
        dt_f = _auto_safe_float(dt, float("nan"))
        if np.isfinite(dt_f):
            dt_list.append(max(0.0, abs(float(dt_f))))

    base = base_per_event_f * float(len(ev))
    if not dt_list:
        return float(max(0.0, base))
    sev = sum((float(dt) / dt_ref_ms_f) ** power_f for dt in dt_list)
    return float(max(0.0, base + dt_weight_f * float(sev)))


def _auto_pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _auto_target_slope_estimate(f_hz, mag_db, *, mask=None) -> float:
    try:
        ff = np.asarray(f_hz, dtype=float).reshape(-1)
        mm = np.asarray(mag_db, dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    if ff.size <= 6 or mm.size != ff.size:
        return float("nan")
    m = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
    try:
        if mask is not None:
            mk = np.asarray(mask, dtype=bool).reshape(-1)
            if mk.size == ff.size:
                m &= mk
    except Exception:
        pass
    if int(np.count_nonzero(m)) < 6:
        return float("nan")
    x = np.log10(ff[m])
    y = mm[m]
    if x.size < 6:
        return float("nan")
    x_span = float(np.max(x) - np.min(x))
    if (not np.isfinite(x_span)) or x_span <= 1e-6:
        return float("nan")
    try:
        p = np.polyfit(x, y, 1)
        slope_db_per_dec = float(p[0])
    except Exception:
        return float("nan")
    return float(slope_db_per_dec * np.log10(2.0))


def _auto_target_preselect_score(
    *,
    fg,
    ml_g,
    mr_g,
    t_g,
    lvl_mask,
    corr_mask,
    mode_mask,
) -> dict:
    ff = np.asarray(fg, dtype=float).reshape(-1)
    ml_arr = np.asarray(ml_g, dtype=float).reshape(-1)
    mr_arr = np.asarray(mr_g, dtype=float).reshape(-1)
    tg_arr = np.asarray(t_g, dtype=float).reshape(-1)
    n = int(ff.size)
    if n <= 0 or ml_arr.size != n or mr_arr.size != n or tg_arr.size != n:
        return {}

    def _safe_mask(raw_mask, *, fallback=None, min_pts: int = 8):
        try:
            mk = np.asarray(raw_mask, dtype=bool).reshape(-1)
        except Exception:
            mk = np.asarray([], dtype=bool)
        if mk.size != n:
            if fallback is not None:
                mk = np.asarray(fallback, dtype=bool).reshape(-1)
            else:
                mk = np.ones(n, dtype=bool)
        if int(np.count_nonzero(mk)) < int(max(1, min_pts)):
            if fallback is not None:
                mk = np.asarray(fallback, dtype=bool).reshape(-1)
            if mk.size != n or int(np.count_nonzero(mk)) < int(max(1, min_pts)):
                mk = np.ones(n, dtype=bool)
        return np.asarray(mk, dtype=bool)

    def _safe_median(v, default=0.0) -> float:
        vv = np.asarray(v, dtype=float).reshape(-1)
        vv = vv[np.isfinite(vv)]
        if vv.size <= 0:
            return float(default)
        return float(np.median(vv))

    def _safe_rms(v, default=float("nan")) -> float:
        vv = np.asarray(v, dtype=float).reshape(-1)
        vv = vv[np.isfinite(vv)]
        if vv.size <= 0:
            return float(default)
        return float(np.sqrt(np.mean(np.square(vv))))

    def _safe_pos_mean(v, default=0.0) -> float:
        vv = np.asarray(v, dtype=float).reshape(-1)
        vv = vv[np.isfinite(vv)]
        if vv.size <= 0:
            return float(default)
        return float(np.mean(np.maximum(vv, 0.0)))

    lvl_m = _safe_mask(lvl_mask, fallback=np.ones(n, dtype=bool), min_pts=8)
    corr_m = _safe_mask(corr_mask, fallback=np.ones(n, dtype=bool), min_pts=8)
    mode_m = _safe_mask(mode_mask, fallback=corr_m, min_pts=6)

    off_l = _safe_median(ml_arr[lvl_m] - tg_arr[lvl_m], 0.0)
    off_r = _safe_median(mr_arr[lvl_m] - tg_arr[lvl_m], 0.0)
    off = 0.5 * (float(off_l) + float(off_r))

    err_l = ml_arr - (tg_arr + float(off_l))
    err_r = mr_arr - (tg_arr + float(off_r))
    err_avg = 0.5 * (err_l + err_r)

    fit_l = _safe_rms(err_l[corr_m], float("nan"))
    fit_r = _safe_rms(err_r[corr_m], float("nan"))
    if not np.isfinite(fit_l):
        fit_l = _safe_rms(err_l, 0.0)
    if not np.isfinite(fit_r):
        fit_r = _safe_rms(err_r, 0.0)
    fit_rms_db = 0.5 * (float(fit_l) + float(fit_r))
    asym_penalty_db = abs(float(fit_l) - float(fit_r))

    needed_l = (tg_arr + float(off_l)) - ml_arr
    needed_r = (tg_arr + float(off_r)) - mr_arr
    total_boost = 0.5 * (
        _safe_pos_mean(needed_l[corr_m], 0.0)
        + _safe_pos_mean(needed_r[corr_m], 0.0)
    )
    mode_boost = 0.5 * (
        _safe_pos_mean(needed_l[mode_m], 0.0)
        + _safe_pos_mean(needed_r[mode_m], 0.0)
    )
    boost_raw = 0.65 * float(total_boost) + 0.35 * float(mode_boost)
    boost_ref = float(max(0.1, _auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MAX_BASS_BOOST_REF_DB, 8.0)))
    boost_penalty = float(boost_ref * np.tanh(float(boost_raw) / float(boost_ref)))

    slope_meas = _auto_target_slope_estimate(ff, 0.5 * (ml_arr + mr_arr), mask=corr_m)
    slope_target = _auto_target_slope_estimate(ff, tg_arr, mask=corr_m)
    slope_penalty = 0.0
    if np.isfinite(slope_meas) and np.isfinite(slope_target):
        slope_penalty = abs(float(slope_meas) - float(slope_target))

    mode_fit_rms_db = _safe_rms(err_avg[mode_m], float("nan"))
    if not np.isfinite(mode_fit_rms_db):
        mode_fit_rms_db = _safe_rms(err_avg[corr_m], 0.0)

    preselect_score = (
        float(fit_rms_db)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_BOOST_W, 0.22)) * float(boost_penalty)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_SLOPE_W, 0.18)) * float(slope_penalty)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_ASYM_W, 0.30)) * float(asym_penalty_db)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_W, 0.16)) * float(mode_fit_rms_db)
    )
    if not np.isfinite(preselect_score):
        preselect_score = float(1e9)

    return {
        "fit_rms_db": float(fit_rms_db),
        "fit_rms_l_db": float(fit_l),
        "fit_rms_r_db": float(fit_r),
        "offset_db": float(off),
        "offset_l_db": float(off_l),
        "offset_r_db": float(off_r),
        "asym_penalty_db": float(asym_penalty_db),
        "boost_penalty": float(boost_penalty),
        "slope_penalty": float(slope_penalty),
        "mode_fit_rms_db": float(mode_fit_rms_db),
        "preselect_score": float(preselect_score),
    }


def _auto_target_adaptive_shortlist(quick_candidates: list[dict], *, top_n: int) -> tuple[list[dict], dict]:
    cands = [dict(tc or {}) for tc in list(quick_candidates or []) if isinstance(tc, dict)]
    if not cands:
        return [], {}

    def _score(tc: dict) -> float:
        return float(
            _auto_safe_float(
                tc.get("preselect_score", tc.get("fit_rms_db", float("inf"))),
                float("inf"),
            )
        )

    cands = sorted(
        cands,
        key=lambda tc: (
            _score(tc),
            _auto_safe_float(tc.get("fit_rms_db", float("inf")), float("inf")),
            str(tc.get("hc_mode", "") or "").strip(),
        ),
    )
    top_n_eff = int(max(1, _auto_safe_int(top_n, AUTO_MODE_TARGET_TOP_N)))
    n_min = int(max(1, _auto_safe_int(AUTO_MODE_TARGET_TOP_N_MIN, 3)))
    n_max = int(max(n_min, _auto_safe_int(AUTO_MODE_TARGET_TOP_N_MAX, 6)))
    spread_db = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_TOP_N_SPREAD_DB, 0.35)))
    best_score = _score(cands[0])
    spread_based_n = int(
        sum(1 for tc in cands if _score(tc) <= (float(best_score) + float(spread_db)))
    )
    shortlist_n = int(max(top_n_eff, spread_based_n))
    shortlist_n = int(min(shortlist_n, n_max))
    shortlist_n = int(max(shortlist_n, n_min))
    shortlist_n = int(min(shortlist_n, len(cands)))
    return list(cands[:shortlist_n]), {
        "best_score": float(best_score),
        "spread_db": float(spread_db),
        "spread_based_n": int(spread_based_n),
        "top_n_eff": int(top_n_eff),
        "shortlist_n": int(shortlist_n),
        "candidate_total": int(len(cands)),
    }


def _auto_target_insert_cached_wildcard(
    shortlisted: list[dict],
    quick_candidates: list[dict],
    *,
    cached_hc_mode: str | None,
) -> tuple[list[dict], dict]:
    out = [dict(tc or {}) for tc in list(shortlisted or []) if isinstance(tc, dict)]
    cached_name = _auto_builtin_target_name(cached_hc_mode)
    if not cached_name:
        return out, {"inserted": False, "reason": "no_valid_cache_target"}

    for tc in out:
        hc = str(tc.get("hc_mode", "") or "").strip()
        if hc == str(cached_name):
            tc["from_cache_wildcard"] = True
            return out, {
                "inserted": False,
                "already_present": True,
                "hc_mode": str(cached_name),
                "reason": "already_shortlisted",
            }

    matched = None
    for tc in list(quick_candidates or []):
        if not isinstance(tc, dict):
            continue
        hc = str(tc.get("hc_mode", "") or "").strip()
        if hc == str(cached_name):
            matched = dict(tc)
            break
    if not isinstance(matched, dict):
        return out, {
            "inserted": False,
            "already_present": False,
            "hc_mode": str(cached_name),
            "reason": "not_in_quick_candidates",
        }

    matched["from_cache_wildcard"] = True
    out.append(dict(matched))
    return out, {
        "inserted": True,
        "already_present": False,
        "hc_mode": str(cached_name),
        "reason": "inserted",
    }


def _auto_select_builtin_target_curve(
    data: dict,
    *,
    f_l,
    m_l,
    f_r,
    m_r,
) -> dict | None:
    try:
        fl = np.asarray(f_l, dtype=float).reshape(-1)
        ml = np.asarray(m_l, dtype=float).reshape(-1)
        fr = np.asarray(f_r, dtype=float).reshape(-1)
        mr = np.asarray(m_r, dtype=float).reshape(-1)
    except Exception:
        return None

    l_ok = bool(fl.size >= 32 and ml.size == fl.size)
    r_ok = bool(fr.size >= 32 and mr.size == fr.size)
    if (not l_ok) and (not r_ok):
        return None
    if (not l_ok) and r_ok:
        fl = np.asarray(fr, dtype=float).copy()
        ml = np.asarray(mr, dtype=float).copy()
    if (not r_ok) and l_ok:
        fr = np.asarray(fl, dtype=float).copy()
        mr = np.asarray(ml, dtype=float).copy()

    def _sorted_xy(f, y):
        idx = np.argsort(f)
        ff = np.asarray(f[idx], dtype=float)
        yy = np.asarray(y[idx], dtype=float)
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        return ff[m], yy[m]

    fl, ml = _sorted_xy(fl, ml)
    fr, mr = _sorted_xy(fr, mr)
    if fl.size < 32 and fr.size >= 32:
        fl = np.asarray(fr, dtype=float).copy()
        ml = np.asarray(mr, dtype=float).copy()
    if fr.size < 32 and fl.size >= 32:
        fr = np.asarray(fl, dtype=float).copy()
        mr = np.asarray(ml, dtype=float).copy()
    if fl.size < 32 or fr.size < 32:
        return None

    try:
        lvl_min = float(data.get("lvl_min", 500.0) or 500.0)
        lvl_max = float(data.get("lvl_max", 2000.0) or 2000.0)
    except Exception:
        lvl_min, lvl_max = 500.0, 2000.0
    if not np.isfinite(lvl_min) or not np.isfinite(lvl_max) or lvl_min <= 0.0 or lvl_max <= lvl_min:
        lvl_min, lvl_max = 500.0, 2000.0

    try:
        mag_lo = float(data.get("mag_c_min", 20.0) or 20.0)
        mag_hi = float(data.get("mag_c_max", 250.0) or 250.0)
    except Exception:
        mag_lo, mag_hi = 20.0, 250.0
    if not np.isfinite(mag_lo) or not np.isfinite(mag_hi) or mag_lo <= 0.0 or mag_hi <= mag_lo:
        mag_lo, mag_hi = 20.0, 250.0

    mode_lo = float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MIN_HZ, 25.0))
    mode_hi = float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MAX_HZ, 160.0))
    if (not np.isfinite(mode_lo)) or (not np.isfinite(mode_hi)) or mode_hi <= mode_lo:
        mode_lo, mode_hi = 25.0, 160.0

    scored = []
    for hc_name in AUTO_MODE_BUILTIN_TARGETS:
        try:
            hf, hm = get_house_curve_by_name(hc_name)
            hf = np.asarray(hf, dtype=float).reshape(-1)
            hm = np.asarray(hm, dtype=float).reshape(-1)
            if hf.size < 4 or hm.size != hf.size:
                continue
            hs = np.argsort(hf)
            hf = hf[hs]
            hm = hm[hs]
            m_h = np.isfinite(hf) & np.isfinite(hm) & (hf > 0.0)
            hf = hf[m_h]
            hm = hm[m_h]
            if hf.size < 4:
                continue

            f_lo = max(20.0, float(np.min(fl)), float(np.min(fr)), float(np.min(hf)))
            f_hi = min(20000.0, float(np.max(fl)), float(np.max(fr)), float(np.max(hf)))
            if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= (f_lo * 1.15):
                continue

            fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
            ml_g = np.interp(fg, fl, ml)
            mr_g = np.interp(fg, fr, mr)
            try:
                ml_sm, _ = dsp.apply_smoothing_std(
                    fg,
                    ml_g,
                    np.zeros_like(ml_g),
                    float(AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT),
                )
                ml_g = np.asarray(ml_sm, dtype=float)
            except Exception:
                pass
            try:
                mr_sm, _ = dsp.apply_smoothing_std(
                    fg,
                    mr_g,
                    np.zeros_like(mr_g),
                    float(AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT),
                )
                mr_g = np.asarray(mr_sm, dtype=float)
            except Exception:
                pass
            t_g = np.interp(fg, hf, hm)

            lvl_mask = (fg >= lvl_min) & (fg <= lvl_max)
            if int(np.count_nonzero(lvl_mask)) < 16:
                lvl_mask = (fg >= 300.0) & (fg <= 3000.0)
            if int(np.count_nonzero(lvl_mask)) < 16:
                lvl_mask = np.ones_like(fg, dtype=bool)

            corr_mask = (fg >= mag_lo) & (fg <= mag_hi)
            if int(np.count_nonzero(corr_mask)) < 16:
                corr_mask = np.ones_like(fg, dtype=bool)

            mode_mask = (fg >= mode_lo) & (fg <= mode_hi) & corr_mask
            if int(np.count_nonzero(mode_mask)) < 8:
                mode_mask = (fg >= mode_lo) & (fg <= mode_hi)

            tc = _auto_target_preselect_score(
                fg=fg,
                ml_g=ml_g,
                mr_g=mr_g,
                t_g=t_g,
                lvl_mask=lvl_mask,
                corr_mask=corr_mask,
                mode_mask=mode_mask,
            )
            if not isinstance(tc, dict) or not tc:
                continue
            tc["hc_mode"] = str(hc_name)
            scored.append(dict(tc))
        except Exception:
            continue

    if not scored:
        return None

    scored = sorted(
        scored,
        key=lambda d: (
            _auto_safe_float(
                d.get("preselect_score", d.get("fit_rms_db", float("inf"))),
                float("inf"),
            ),
            _auto_safe_float(d.get("fit_rms_db", float("inf")), float("inf")),
            str(d.get("hc_mode", "") or "").strip(),
        ),
    )
    best = scored[0]
    return {
        "selected_hc_mode": str(best.get("hc_mode", "Harman6")),
        "fit_rms_db": float(best.get("fit_rms_db", 0.0)),
        "offset_db": float(best.get("offset_db", 0.0)),
        "candidates": list(scored[:5]),
        "candidates_all": list(scored),
    }


def _auto_select_target_curve_with_trials(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
) -> dict | None:
    goal = _auto_goal(base_data)
    program_version = _auto_program_version(base_data)
    filter_key = _auto_filter_cache_key(base_data)
    rank_basis = _auto_goal_basis_text(goal)
    logger.info(f"Automatic mode target select: goal={goal}, basis={rank_basis}")

    # Deterministic seed for target-selection trials:
    # keeps "topNxM trials" reproducible for same dataset/settings.
    seed_target = _auto_seed_from_signature(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=None,
        include_hc_mode=False,
    )
    _auto_apply_seed(seed_target)
    logger.info(f"Automatic mode target select: seed={int(seed_target)}")

    def _cached_target_fit(hc_name: str) -> tuple[float, float]:
        fit_rms_db = float("nan")
        offset_db = 0.0
        try:
            quick_fit = _auto_select_builtin_target_curve(
                dict(base_data or {}),
                f_l=measurements.get("f_l"),
                m_l=measurements.get("m_l"),
                f_r=measurements.get("f_r"),
                m_r=measurements.get("m_r"),
            )
            cands = list(
                (quick_fit or {}).get(
                    "candidates_all",
                    (quick_fit or {}).get("candidates", []),
                )
                or []
            )
            target_c = None
            for c in cands:
                if str((c or {}).get("hc_mode", "") or "").strip() == str(hc_name):
                    target_c = dict(c or {})
                    break
            if isinstance(target_c, dict):
                fit_rms_db = float(
                    _auto_safe_float(target_c.get("fit_rms_db", float("nan")), float("nan"))
                )
                offset_db = float(_auto_safe_float(target_c.get("offset_db", 0.0), 0.0))
        except Exception:
            pass
        return float(fit_rms_db), float(offset_db)

    def _cache_target_valid(hc_name: str | None) -> bool:
        hc = _auto_builtin_target_name(hc_name)
        if not hc:
            return False
        try:
            c_f, c_m = get_house_curve_by_name(hc)
            c_f = np.asarray(c_f, dtype=float).reshape(-1)
            c_m = np.asarray(c_m, dtype=float).reshape(-1)
            return bool(c_f.size >= 4 and c_m.size == c_f.size)
        except Exception:
            return False

    def _cached_target_return(
        cached_hc_mode: str | None,
        cached_preset: dict,
        *,
        selection_method: str,
    ) -> dict | None:
        if not _cache_target_valid(cached_hc_mode):
            return None
        fit_rms_db, offset_db = _cached_target_fit(str(cached_hc_mode))
        return {
            "selected_hc_mode": str(cached_hc_mode),
            "fit_rms_db": float(fit_rms_db),
            "offset_db": float(offset_db),
            "selection_method": str(selection_method),
            "selection_basis": str(rank_basis),
            "auto_goal": str(goal),
            "top_n": 0,
            "trials_per_curve": 0,
            "candidates": [],
            "evaluated": [],
            "best_preset": dict(cached_preset or {}),
        }

    def _tc_score(tc: dict | None) -> float:
        return float(
            _auto_safe_float(
                (tc or {}).get(
                    "preselect_score",
                    (tc or {}).get("fit_rms_db", float("inf")),
                ),
                float("inf"),
            )
        )

    cached_target_hc = None
    cached_target_preset = {}
    cached_target_source = None

    # Strong cache key: measurement response only.
    cached_target_entry = None
    try:
        cached_target_entry = _auto_cache_get_target_for_measurements(
            measurements,
            goal=goal,
            filter_key=filter_key,
            program_version=program_version,
        )
    except Exception:
        cached_target_entry = None
    if isinstance(cached_target_entry, dict):
        cached_hc = str(
            cached_target_entry.get(
                "best_target_curve",
                cached_target_entry.get("best_hc_mode", ""),
            )
            or ""
        ).strip()
        cached_hc = _auto_builtin_target_name(cached_hc)
        if _cache_target_valid(cached_hc):
            cached_target_hc = str(cached_hc)
            cached_target_preset = dict(cached_target_entry.get("best_preset", {}) or {})
            cached_target_source = "cache_measurement"
            logger.info(
                "Automatic mode target select: cache seed (measurement) target=%s",
                str(cached_target_hc),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target preselect cache seed "
                    f"(measurement -> {str(cached_target_hc)})"
                )

    # Cache prior: same measurements + key settings.
    if bool(AUTO_MODE_CACHE_ENABLED):
        try:
            sig_target = _auto_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            cached_hc = _auto_cache_get_best_target(
                sig_target,
                filter_key=filter_key,
                program_version=program_version,
            )
            cached_hc = _auto_builtin_target_name(cached_hc)
            if (not cached_target_hc) and _cache_target_valid(cached_hc):
                cached_target_hc = str(cached_hc)
                cached_target_preset = _auto_cache_get_best(
                    sig_target,
                    filter_key=filter_key,
                    program_version=program_version,
                ) or {}
                cached_target_source = "cache"
                logger.info(
                    "Automatic mode target select: cache seed (signature) target=%s",
                    str(cached_target_hc),
                )
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: target preselect cache seed "
                        f"(signature -> {str(cached_target_hc)})"
                    )
        except Exception:
            pass

    f6_hz = _auto_safe_float(
        base_data.get("_auto_mag_c_min_hz", base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f" (-6 dB {f6_hz:.1f} Hz)" if np.isfinite(f6_hz) else ""

    quick = _auto_select_builtin_target_curve(
        base_data,
        f_l=measurements.get("f_l"),
        m_l=measurements.get("m_l"),
        f_r=measurements.get("f_r"),
        m_r=measurements.get("m_r"),
    )
    if not isinstance(quick, dict):
        fallback = _cached_target_return(
            cached_target_hc,
            cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: quick preselect unavailable, fallback to cached target=%s",
                str(cached_target_hc),
            )
            return dict(fallback)
        return None

    quick_candidates = list(
        quick.get("candidates_all", quick.get("candidates", []))
        or quick.get("candidates", [])
        or []
    )
    quick_candidates = [dict(tc or {}) for tc in quick_candidates if isinstance(tc, dict)]
    if not quick_candidates:
        fallback = _cached_target_return(
            cached_target_hc,
            cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: no quick candidates, fallback to cached target=%s",
                str(cached_target_hc),
            )
            return dict(fallback)
        return None

    quick_candidates = sorted(
        quick_candidates,
        key=lambda tc: (
            _tc_score(tc),
            _auto_safe_float(tc.get("fit_rms_db", float("inf")), float("inf")),
            str(tc.get("hc_mode", "") or "").strip(),
        ),
    )
    quick_rows = []
    for tc in quick_candidates:
        quick_rows.append(
            f"{str(tc.get('hc_mode', 'n/a'))}: "
            f"fit={_auto_safe_float(tc.get('fit_rms_db', float('nan')), float('nan')):.3f} dB, "
            f"pre={_tc_score(tc):.3f}, "
            f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
            f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f}"
        )
    logger.info(
        "Automatic mode target preselect candidates:\n%s",
        "\n".join(quick_rows),
    )
    if callable(status_cb):
        top3_txt = ", ".join(
            [
                (
                    f"{str(tc.get('hc_mode', 'n/a') or 'n/a')}"
                    f"(pre={_tc_score(tc):.3f}, fit={_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
                    f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f})"
                )
                for tc in quick_candidates[:3]
            ]
        )
        status_cb(
            "CamillaFIR automatic mode: target preselect top-3 "
            f"(goal {goal}) {top3_txt}"
        )

    trials_eff = max(1, int(trials_per_curve))
    shortlisted, shortlist_meta = _auto_target_adaptive_shortlist(quick_candidates, top_n=int(top_n))
    if not shortlisted:
        fallback = _cached_target_return(
            cached_target_hc,
            cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    logger.info(
        "Automatic mode target shortlist: n=%d/%d (base_top_n=%d, spread_based_n=%d, spread=%.3f dB, best_score=%.3f)",
        int(shortlist_meta.get("shortlist_n", len(shortlisted))),
        int(shortlist_meta.get("candidate_total", len(quick_candidates))),
        int(shortlist_meta.get("top_n_eff", max(1, int(top_n)))),
        int(shortlist_meta.get("spread_based_n", len(shortlisted))),
        float(_auto_safe_float(shortlist_meta.get("spread_db", AUTO_MODE_TARGET_TOP_N_SPREAD_DB), AUTO_MODE_TARGET_TOP_N_SPREAD_DB)),
        float(_auto_safe_float(shortlist_meta.get("best_score", _tc_score(shortlisted[0])), _tc_score(shortlisted[0]))),
    )
    if callable(status_cb):
        status_cb(
            "CamillaFIR automatic mode: target shortlist "
            f"(selected {int(shortlist_meta.get('shortlist_n', len(shortlisted)))}/"
            f"{int(shortlist_meta.get('candidate_total', len(quick_candidates)))} "
            f"by spread {float(_auto_safe_float(shortlist_meta.get('spread_db', AUTO_MODE_TARGET_TOP_N_SPREAD_DB), AUTO_MODE_TARGET_TOP_N_SPREAD_DB)):.2f} dB)"
        )

    cache_wildcard_participated = False
    if _auto_safe_bool(AUTO_MODE_TARGET_CACHE_AS_WILDCARD, True) and _cache_target_valid(cached_target_hc):
        shortlisted, cache_meta = _auto_target_insert_cached_wildcard(
            shortlisted,
            quick_candidates,
            cached_hc_mode=str(cached_target_hc),
        )
        cache_wildcard_participated = bool(
            cache_meta.get("inserted", False) or cache_meta.get("already_present", False)
        )
        if bool(cache_meta.get("inserted", False)):
            logger.info(
                "Automatic mode target shortlist: inserted cache wildcard target=%s",
                str(cache_meta.get("hc_mode", cached_target_hc)),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target shortlist cache wildcard inserted "
                    f"({str(cache_meta.get('hc_mode', cached_target_hc))})"
                )
        else:
            logger.info(
                "Automatic mode target shortlist: cache wildcard skipped target=%s (%s)",
                str(cache_meta.get("hc_mode", cached_target_hc)),
                str(cache_meta.get("reason", "unknown")),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target shortlist cache wildcard "
                    f"{str(cache_meta.get('reason', 'skipped'))}"
                )

    prefer_milder = _auto_safe_bool(
        base_data.get("auto_target_prefer_milder_step", AUTO_MODE_TARGET_PREFER_MILDER_STEP),
        AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    )
    if prefer_milder and shortlisted:
        leader = dict(shortlisted[0] or {})
        lead_hc = str(leader.get("hc_mode", "") or "").strip()
        lead_milder = _auto_target_one_step_milder(lead_hc)
        if lead_milder:
            milder_tc = None
            for tc in quick_candidates:
                if str(tc.get("hc_mode", "") or "").strip() == str(lead_milder):
                    milder_tc = dict(tc)
                    break
            if isinstance(milder_tc, dict):
                already = {
                    str(tc.get("hc_mode", "") or "").strip()
                    for tc in shortlisted
                    if isinstance(tc, dict)
                }
                leader_fit = _auto_safe_float(leader.get("fit_rms_db", float("inf")), float("inf"))
                milder_fit = _auto_safe_float(milder_tc.get("fit_rms_db", float("inf")), float("inf"))
                leader_pre = _tc_score(leader)
                milder_pre = _tc_score(milder_tc)
                leader_asym = _auto_safe_float(leader.get("asym_penalty_db", 0.0), 0.0)
                milder_asym = _auto_safe_float(milder_tc.get("asym_penalty_db", 0.0), 0.0)
                leader_boost = _auto_safe_float(leader.get("boost_penalty", 0.0), 0.0)
                milder_boost = _auto_safe_float(milder_tc.get("boost_penalty", 0.0), 0.0)
                cond_not_dup = str(lead_milder) not in already
                cond_fit = bool(
                    float(milder_fit)
                    <= float(leader_fit) + float(AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB)
                )
                cond_diff = bool(
                    float(milder_pre)
                    <= float(leader_pre) + float(AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD)
                )
                cond_asym = bool(
                    float(milder_asym)
                    <= float(leader_asym) + float(AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD)
                )
                if cond_not_dup and cond_fit and cond_diff and cond_asym:
                    shortlisted = list(shortlisted) + [dict(milder_tc)]
                    logger.info(
                        "Automatic mode target shortlist: included milder target %s -> %s "
                        "(fit %.3f->%.3f, pre %.3f->%.3f, boost %.3f->%.3f, asym %.3f->%.3f)",
                        str(lead_hc),
                        str(lead_milder),
                        float(leader_fit),
                        float(milder_fit),
                        float(leader_pre),
                        float(milder_pre),
                        float(leader_boost),
                        float(milder_boost),
                        float(leader_asym),
                        float(milder_asym),
                    )
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: target shortlist milder included "
                            f"({str(lead_hc)} -> {str(lead_milder)})"
                        )
                else:
                    logger.info(
                        "Automatic mode target shortlist: skipped milder target %s -> %s "
                        "(not_dup=%s, fit_ok=%s, pre_ok=%s, asym_ok=%s, boost %.3f->%.3f)",
                        str(lead_hc),
                        str(lead_milder),
                        str(cond_not_dup),
                        str(cond_fit),
                        str(cond_diff),
                        str(cond_asym),
                        float(leader_boost),
                        float(milder_boost),
                    )
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: target shortlist milder skipped "
                            f"({str(lead_hc)} -> {str(lead_milder)}, "
                            f"not_dup={str(cond_not_dup)}, fit_ok={str(cond_fit)}, "
                            f"pre_ok={str(cond_diff)}, asym_ok={str(cond_asym)})"
                        )
            else:
                logger.info(
                    "Automatic mode target shortlist: milder target for %s not found in quick candidates",
                    str(lead_hc),
                )
        else:
            logger.info(
                "Automatic mode target shortlist: no one-step milder target for %s",
                str(lead_hc),
            )

    dedup_shortlisted = []
    seen_hc = set()
    for tc in shortlisted:
        if not isinstance(tc, dict):
            continue
        hc = str(tc.get("hc_mode", "") or "").strip()
        if not hc or hc in seen_hc:
            continue
        seen_hc.add(hc)
        dedup_shortlisted.append(dict(tc))
    shortlisted = list(dedup_shortlisted)
    if not shortlisted:
        fallback = _cached_target_return(
            cached_target_hc,
            cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    for tc in shortlisted:
        tc.setdefault("preselect_score", _tc_score(tc))
        tc.setdefault("boost_penalty", _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0))
        tc.setdefault("asym_penalty_db", _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0))

    def _target_eval_one(
        preset: dict,
        *,
        base_tc: dict,
        hc_f_arr,
        hc_m_arr,
    ) -> dict:
        trial_data = dict(base_tc)
        trial_data.update(dict(preset or {}))
        if str(filter_key) in ("linear", "asym"):
            trial_data["phase_limit"] = round(
                float(_auto_phase_limit_clip(trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)), default=400.0)),
                1,
            )
        trial_data["comparison_mode"] = True
        trial_measurements = dict(measurements or {})
        trial_measurements["ui_data"] = trial_data

        cfg = build_config(
            trial_data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f_arr,
            hc_m=hc_m_arr,
            pin=pin_obj,
            max_safe_boost=float(MAX_SAFE_BOOST),
        )
        try:
            setattr(cfg, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
        except Exception:
            pass

        res = run_pipeline(cfg, trial_measurements, include_response_arrays=False)
        met = _auto_score_result(
            res,
            auto_exc_freq_hz=_auto_safe_float(trial_data.get("_auto_exc_freq_hz", float("nan")), float("nan")),
            base_data=trial_data,
        )
        trial_preset = dict(preset or {})
        if str(filter_key) == "mixed":
            trial_preset["mixed_freq"] = round(
                _clip(
                    trial_data.get("mixed_freq", base_tc.get("mixed_freq", 180.0)),
                    80.0,
                    320.0,
                ),
                1,
            )
        elif str(filter_key) in ("linear", "asym"):
            trial_preset["phase_limit"] = round(
                float(
                    _auto_phase_limit_clip(
                        trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                        default=400.0,
                    )
                ),
                1,
            )
        return {
            "ok": True,
            "metrics": dict(met or {}),
            "preset": dict(trial_preset),
        }

    def _run_target_trials(
        cands: list[dict],
        *,
        base_tc: dict,
        hc_f_arr,
        hc_m_arr,
        phase_tag: str,
        target_name: str,
    ) -> list[dict]:
        n_total = int(len(cands))
        if n_total <= 0:
            return []
        workers = int(_auto_trial_workers(base_tc, n_total))
        if workers > 1:
            logger.info(
                "Automatic mode target trials: target=%s, phase=%s, parallel workers=%d",
                str(target_name),
                str(phase_tag),
                int(workers),
            )

        idx_presets = list(enumerate(list(cands or []), start=1))
        out_by_idx: dict[int, dict] = {}
        if workers <= 1 or n_total <= 1:
            for idx, preset in idx_presets:
                try:
                    out = _target_eval_one(
                        dict(preset or {}),
                        base_tc=base_tc,
                        hc_f_arr=hc_f_arr,
                        hc_m_arr=hc_m_arr,
                    )
                except Exception as exc:
                    out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                out_by_idx[int(idx)] = dict(out or {})
        else:
            chunk_size = int(_auto_trial_chunk_size(workers))
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
                    chunk = idx_presets[c0 : c0 + int(chunk_size)]
                    fut_map = {
                        ex.submit(
                            _target_eval_one,
                            dict(preset or {}),
                            base_tc=base_tc,
                            hc_f_arr=hc_f_arr,
                            hc_m_arr=hc_m_arr,
                        ): int(idx)
                        for idx, preset in chunk
                    }
                    for fut in as_completed(list(fut_map.keys())):
                        idx = int(fut_map.get(fut, 0))
                        try:
                            out = fut.result()
                            if not isinstance(out, dict):
                                out = {"ok": False, "error": "invalid worker result"}
                        except Exception as exc:
                            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        out_by_idx[int(idx)] = dict(out or {})

        out = []
        for idx, _preset in idx_presets:
            out.append(
                dict(
                    out_by_idx.get(
                        int(idx),
                        {"ok": False, "error": "missing worker result"},
                    )
                    or {}
                )
            )
        return out

    def _evaluate_target_curve(
        tc: dict,
        *,
        t_idx: int,
        emit_status: bool,
        curve_inner_workers: int | None,
    ) -> dict | None:
        hc_name = str(tc.get("hc_mode", "") or "").strip()
        if not hc_name:
            return None
        try:
            hc_f, hc_m = get_house_curve_by_name(hc_name)
            hc_f = np.asarray(hc_f, dtype=float)
            hc_m = np.asarray(hc_m, dtype=float)
        except Exception:
            return None
        if hc_f.size < 4 or hc_m.size != hc_f.size:
            return None

        seed_tc = int(
            (
                _auto_seed_from_signature(
                    base_data=base_data,
                    measurements=measurements,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    hc_mode=str(hc_name),
                    include_hc_mode=True,
                )
                + sum(ord(ch) for ch in hc_name) * 13
            )
            & 0xFFFFFFFF
        )
        if bool(emit_status):
            _auto_apply_seed(seed_tc)
        base_tc = dict(base_data or {})
        base_tc["hc_mode"] = str(hc_name)
        if isinstance(curve_inner_workers, int) and int(curve_inner_workers) > 0:
            base_tc["auto_mode_workers"] = int(curve_inner_workers)
        if str(filter_key) in ("linear", "asym"):
            base_tc["phase_limit"] = round(
                float(_auto_phase_limit_center(base_tc.get("phase_limit", None))),
                1,
            )
        candidates = _build_auto_mode_candidates(
            base_tc,
            n_trials=trials_eff,
            seed=seed_tc,
            optimize_mag_low=False,
        )

        best_metrics = None
        best_preset = None
        ok_n = 0
        rank_sum = 0.0
        avg_score_sum = 0.0
        phase1_scored = []
        trials_total_count = int(len(candidates))
        cb = status_cb if bool(emit_status) else None

        phase1_out = _run_target_trials(
            candidates,
            base_tc=base_tc,
            hc_f_arr=hc_f,
            hc_m_arr=hc_m,
            phase_tag="phase1",
            target_name=hc_name,
        )
        for c_idx, out in enumerate(phase1_out, start=1):
            improved = False
            if bool(out.get("ok", False)):
                met = dict(out.get("metrics", {}) or {})
                trial_preset = dict(out.get("preset", {}) or {})
                ok_n += 1
                rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
                phase1_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
                if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                    best_metrics = dict(met)
                    best_preset = dict(trial_preset)
                    improved = True
            else:
                logger.warning(
                    f"Automatic mode target trial failed: target={hc_name} "
                    f"{c_idx}/{len(candidates)} ({str(out.get('error', 'unknown error') or 'unknown error')})"
                )

            if callable(cb) and bool(improved):
                rank_now = _auto_safe_float((best_metrics or {}).get("rank_score"), 0.0)
                avg_now = _auto_safe_float((best_metrics or {}).get("avg_score"), 0.0)
                cb(
                    "CamillaFIR automatic mode: target trials best improved "
                    f"(target {t_idx}/{len(shortlisted)} {hc_name}, "
                    f"trial {c_idx}/{len(candidates)}{f6_txt}, goal {goal}, "
                    f"rank {rank_now:.3f}, avg {avg_now:.3f}, "
                    f"fit {_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"pre {_auto_safe_float(tc.get('preselect_score', tc.get('fit_rms_db', 0.0)), 0.0):.3f})"
                )

        if phase1_scored and bool(AUTO_MODE_LOCAL_REFINE_ENABLED) and _auto_goal_uses_local_refine(goal):
            top_list = sorted(
                list(phase1_scored),
                key=lambda it: _auto_rank_key(dict(it.get("metrics", {}) or {})),
            )[: int(max(1, AUTO_MODE_LOCAL_REFINE_TOP_K))]
            ref_profile = _auto_build_refine_profile(
                base_data=base_tc,
                phase1_top=top_list,
            )
            phase1_best = dict(top_list[0])
            p1m = dict(phase1_best.get("metrics", {}) or {})
            p1p = dict(phase1_best.get("preset", {}) or {})
            p1_mixed = _auto_safe_float(p1p.get("mixed_freq", base_tc.get("mixed_freq", float("nan"))), float("nan"))
            p1_phase = _auto_safe_float(p1p.get("phase_limit", base_tc.get("phase_limit", float("nan"))), float("nan"))
            p1_tdc = _auto_safe_float(p1p.get("tdc_strength", base_tc.get("tdc_strength", float("nan"))), float("nan"))
            p1_mode = _auto_safe_float(p1m.get("mode_ripple_db"), float("nan"))
            p1_boost = _auto_safe_float(p1m.get("max_net_boost_db"), float("nan"))
            p1_mode_txt = f"{p1_mode:.3f} dB" if np.isfinite(p1_mode) else "n/a"
            p1_boost_txt = f"{p1_boost:.2f} dB" if np.isfinite(p1_boost) else "n/a"
            if str(filter_key) == "mixed":
                p1_detail = f"mixed_freq={p1_mixed:.1f} Hz, tdc={p1_tdc:.1f}"
            elif str(filter_key) in ("linear", "asym"):
                p1_detail = (
                    f"phase_limit={p1_phase:.1f} Hz, tdc={p1_tdc:.1f}"
                    if np.isfinite(p1_phase)
                    else f"phase_limit=n/a, tdc={p1_tdc:.1f}"
                )
            else:
                p1_detail = f"tdc={p1_tdc:.1f}"
            logger.info(
                "Automatic mode target Phase1 done: "
                f"target={hc_name}, avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                f"{p1_detail}"
            )
            if callable(cb):
                cb(
                    "CamillaFIR automatic mode: Phase1 done "
                    f"target={hc_name}, rank={_auto_safe_float(p1m.get('rank_score'), 0.0):.3f}, "
                    f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                    f"mode_ripple={p1_mode_txt}, "
                    f"boost={p1_boost_txt}, "
                    f"{p1_detail}"
                )

            for li, item in enumerate(top_list, start=1):
                center = dict(item.get("preset", {}) or {})
                c_mixed = _auto_safe_float(center.get("mixed_freq", base_tc.get("mixed_freq", float("nan"))), float("nan"))
                c_phase = _auto_safe_float(center.get("phase_limit", base_tc.get("phase_limit", float("nan"))), float("nan"))
                local_detail = None
                if str(filter_key) == "mixed":
                    local_detail = f"mixed_freq={c_mixed:.1f} Hz"
                elif str(filter_key) in ("linear", "asym"):
                    local_detail = (
                        f"phase refine phase_limit={c_phase:.1f} Hz"
                        if np.isfinite(c_phase)
                        else "phase refine phase_limit=n/a"
                    )
                if local_detail is not None:
                    logger.info(
                        "Automatic mode target Local refine: "
                        f"target={hc_name}, center #{li}, {local_detail}"
                    )
                    if callable(cb):
                        cb(
                            f"CamillaFIR automatic mode: Local refine target={hc_name} "
                            f"center #{li} {local_detail}"
                        )

                local_candidates = _build_auto_mode_candidates_local(
                    base_tc,
                    center,
                    int(AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP),
                    int(seed_tc + li * 100003),
                    shrink=float(
                        _auto_adaptive_shrink_factor(
                            top_list,
                            base_shrink=float(AUTO_MODE_LOCAL_REFINEMENT_SHRINK),
                            plateau_hit=False,
                        )
                    ),
                    optimize_mag_low=False,
                )
                for cand in local_candidates:
                    mf = _auto_safe_float(cand.get("mixed_freq"), float("nan"))
                    if np.isfinite(mf):
                        cand["mixed_freq"] = _clip(
                            mf,
                            ref_profile["mixed_center"] - ref_profile["mixed_span"],
                            ref_profile["mixed_center"] + ref_profile["mixed_span"],
                        )

                    td = _auto_safe_float(cand.get("tdc_strength"), float("nan"))
                    if np.isfinite(td):
                        cand["tdc_strength"] = _clip(
                            td,
                            ref_profile["tdc_lo"],
                            ref_profile["tdc_hi"],
                        )
                    if str(filter_key) in ("linear", "asym"):
                        cand["phase_limit"] = round(
                            float(_auto_phase_limit_clip(cand.get("phase_limit", base_tc.get("phase_limit", 400.0)), default=400.0)),
                            1,
                        )
                trials_total_count += int(len(local_candidates))
                local_out = _run_target_trials(
                    local_candidates,
                    base_tc=base_tc,
                    hc_f_arr=hc_f,
                    hc_m_arr=hc_m,
                    phase_tag=f"local_center_{li}",
                    target_name=hc_name,
                )
                for lc_idx, out in enumerate(local_out, start=1):
                    if bool(out.get("ok", False)):
                        met = dict(out.get("metrics", {}) or {})
                        trial_preset = dict(out.get("preset", {}) or {})
                        ok_n += 1
                        rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                        avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
                        if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                            prev = dict(best_metrics or {})
                            best_metrics = dict(met)
                            best_preset = dict(trial_preset)
                            logger.info(
                                "Automatic mode target Local refine winner improved: "
                                f"target={hc_name}, avg_score={_auto_safe_float(prev.get('avg_score'), 0.0):.3f}"
                                f" -> {_auto_safe_float(met.get('avg_score'), 0.0):.3f}, "
                                f"rank_score={_auto_safe_float(prev.get('rank_score'), 0.0):.3f}"
                                f" -> {_auto_safe_float(met.get('rank_score'), 0.0):.3f}"
                            )
                    else:
                        logger.warning(
                            f"Automatic mode target local trial failed: target={hc_name} "
                            f"center={li} {lc_idx}/{len(local_candidates)} "
                            f"({str(out.get('error', 'unknown error') or 'unknown error')})"
                        )

        if ok_n <= 0 or not isinstance(best_metrics, dict):
            return None

        return {
            "hc_mode": str(hc_name),
            "fit_rms_db": _auto_safe_float(tc.get("fit_rms_db"), 0.0),
            "offset_db": _auto_safe_float(tc.get("offset_db"), 0.0),
            "preselect_score": _auto_safe_float(
                tc.get("preselect_score", tc.get("fit_rms_db", float("inf"))),
                float("inf"),
            ),
            "boost_penalty": _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0),
            "asym_penalty_db": _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0),
            "mode_fit_rms_db": _auto_safe_float(tc.get("mode_fit_rms_db", 0.0), 0.0),
            "from_cache_wildcard": bool(tc.get("from_cache_wildcard", False)),
            "trials_total": int(trials_total_count),
            "trials_ok": int(ok_n),
            "avg_rank_score": float(rank_sum / max(1, ok_n)),
            "avg_avg_score": float(avg_score_sum / max(1, ok_n)),
            "best_metrics": dict(best_metrics),
            "best_preset": dict(best_preset or {}),
        }

    evaluated = []
    total_target_trial_load = int(max(1, len(shortlisted) * max(1, trials_eff)))
    curve_budget = int(_auto_trial_workers(base_data, total_target_trial_load))
    curve_workers = int(max(1, min(len(shortlisted), curve_budget)))
    curve_inner_workers = int(max(1, curve_budget // max(1, curve_workers)))
    select_f6_txt = f", -6 dB point {f6_hz:.1f} Hz" if np.isfinite(f6_hz) else ""

    def _curve_item_progress_key(item: dict) -> tuple:
        bm = dict((item or {}).get("best_metrics", {}) or {})
        return (
            -_auto_safe_float(bm.get("rank_score"), 0.0),
            -_auto_safe_float((item or {}).get("avg_rank_score"), 0.0),
            _auto_safe_float((item or {}).get("fit_rms_db"), 1e9),
        )

    if curve_workers > 1:
        logger.info(
            "Automatic mode target select: curve-parallel enabled "
            "(curves=%d, workers=%d, inner_workers=%d)",
            int(len(shortlisted)),
            int(curve_workers),
            int(curve_inner_workers),
        )
        with ThreadPoolExecutor(max_workers=int(curve_workers)) as ex:
            fut_map = {}
            best_done_item = None
            done_n = 0
            for t_idx, tc in enumerate(shortlisted, start=1):
                fut = ex.submit(
                    _evaluate_target_curve,
                    dict(tc or {}),
                    t_idx=int(t_idx),
                    emit_status=False,
                    curve_inner_workers=int(curve_inner_workers),
                )
                fut_map[fut] = (int(t_idx), str((tc or {}).get("hc_mode", "") or "").strip())
            for fut in as_completed(list(fut_map.keys())):
                t_idx, hc_name = fut_map.get(fut, (0, "n/a"))
                done_n += 1
                improved = False
                try:
                    item = fut.result()
                except Exception as exc:
                    logger.warning(
                        "Automatic mode target curve failed: target=%s (%s)",
                        str(hc_name),
                        f"{type(exc).__name__}: {exc}",
                    )
                    item = None
                if isinstance(item, dict):
                    item_d = dict(item)
                    evaluated.append(dict(item_d))
                    if (
                        not isinstance(best_done_item, dict)
                        or _curve_item_progress_key(item_d) < _curve_item_progress_key(best_done_item)
                    ):
                        best_done_item = dict(item_d)
                        improved = True
                if callable(status_cb) and bool(improved) and isinstance(best_done_item, dict):
                    bm_now = dict(best_done_item.get("best_metrics", {}) or {})
                    status_cb(
                        "CamillaFIR automatic mode: selecting target curve "
                        f"(best improved {int(done_n)}/{len(shortlisted)}, "
                        f"leader {str(best_done_item.get('hc_mode', 'n/a') or 'n/a')}, "
                        f"tested {str(hc_name or 'n/a')}, "
                        f"{int(trials_eff)} trials/curve{select_f6_txt}, goal {goal}, "
                        f"rank {_auto_safe_float(bm_now.get('rank_score'), 0.0):.3f}, "
                        f"avg {_auto_safe_float(bm_now.get('avg_score'), 0.0):.3f})"
                    )
    else:
        for t_idx, tc in enumerate(shortlisted, start=1):
            hc_name = str((tc or {}).get("hc_mode", "") or "").strip() or "n/a"
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: selecting target curve "
                    f"(testing {hc_name} {t_idx}/{len(shortlisted)}, "
                    f"{int(trials_eff)} trials/curve{select_f6_txt}, goal {goal})"
                )
            item = _evaluate_target_curve(
                dict(tc or {}),
                t_idx=int(t_idx),
                emit_status=True,
                curve_inner_workers=None,
            )
            if isinstance(item, dict):
                evaluated.append(dict(item))

    if not evaluated:
        quick_out = dict(quick or {})
        quick_out.setdefault("selection_method", "fit_rms")
        quick_out["selection_basis"] = str(rank_basis)
        quick_out["auto_goal"] = str(goal)
        return quick_out

    def _tc_key(item: dict) -> tuple:
        bm = dict(item.get("best_metrics", {}) or {})
        return (
            -_auto_safe_float(bm.get("rank_score"), 0.0),
            -_auto_safe_float(item.get("avg_rank_score"), 0.0),
            _auto_safe_float(item.get("fit_rms_db"), 1e9),
        )

    def _mode_ripple_from_item(item: dict) -> float:
        bm = dict(item.get("best_metrics", {}) or {})
        v = _auto_safe_float(bm.get("mode_ripple_db", float("nan")), float("nan"))
        if np.isfinite(v):
            return float(v)
        return float("inf")

    def _target_mildness_index(hc_name: str) -> int:
        name = str(hc_name or "").strip()
        if not name:
            return 10_000
        ladders = (
            ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
            ("BK_Light", "BK_Medium", "BK_Strong"),
        )
        for ladder in ladders:
            if name in ladder:
                return int(ladder.index(name))
        return 10_000

    evaluated = sorted(evaluated, key=_tc_key)
    winner = evaluated[0]
    selection_method = "top3x10_trials"
    rank_tie_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_BEST_RANK_TIE_EPS, 0.05)))
    winner_rank = _auto_safe_float(dict(winner.get("best_metrics", {}) or {}).get("rank_score"), 0.0)
    near_top = []
    for it in evaluated:
        it_rank = _auto_safe_float(dict(it.get("best_metrics", {}) or {}).get("rank_score"), 0.0)
        if abs(float(winner_rank) - float(it_rank)) < rank_tie_eps:
            near_top.append(dict(it))
    if len(near_top) >= 2:
        near_top = sorted(
            near_top,
            key=lambda it: (
                -_auto_safe_float(it.get("avg_rank_score"), 0.0),
                _mode_ripple_from_item(it),
                _auto_safe_float(it.get("boost_penalty", 0.0), 0.0),
                _auto_safe_float(it.get("fit_rms_db"), 1e9),
                _auto_safe_float(
                    it.get("preselect_score", it.get("fit_rms_db", 1e9)),
                    1e9,
                ),
                _target_mildness_index(str(it.get("hc_mode", "") or "").strip()),
                str(it.get("hc_mode", "") or "").strip(),
            ),
        )
        old_winner = dict(winner)
        winner = dict(near_top[0])
        selection_method = "top3x10_trials_rank_tie_composite"
        logger.info(
            "Automatic mode target select: rank tie-break by avg/mode/boost "
            f"(eps={rank_tie_eps:.3f}) "
            f"{str(old_winner.get('hc_mode', 'n/a'))} -> {str(winner.get('hc_mode', 'n/a'))}, "
            f"avg_rank={_auto_safe_float(old_winner.get('avg_rank_score'), 0.0):.3f}"
            f" -> {_auto_safe_float(winner.get('avg_rank_score'), 0.0):.3f}, "
            f"mode_ripple={_mode_ripple_from_item(old_winner):.4f}"
            f" -> {_mode_ripple_from_item(winner):.4f}, "
            f"boost_penalty={_auto_safe_float(old_winner.get('boost_penalty', 0.0), 0.0):.3f}"
            f" -> {_auto_safe_float(winner.get('boost_penalty', 0.0), 0.0):.3f}"
        )
    if bool(cache_wildcard_participated) and bool(winner.get("from_cache_wildcard", False)):
        selection_method = "trial_with_cache_wildcard"

    winner_mode_ripple = _mode_ripple_from_item(winner)
    logger.info(
        "Automatic mode target select: "
        f"goal={goal}, basis={rank_basis}, winner={str(winner.get('hc_mode', 'n/a'))}, "
        f"{_auto_metric_text(dict(winner.get('best_metrics', {}) or {}), goal)}, "
        f"avg_rank={_auto_safe_float(winner.get('avg_rank_score', 0.0), 0.0):.3f}, "
        f"mode_ripple={winner_mode_ripple:.4f}, "
        f"pre={_auto_safe_float(winner.get('preselect_score', winner.get('fit_rms_db', 1e9)), 1e9):.3f}, "
        f"boost={_auto_safe_float(winner.get('boost_penalty', 0.0), 0.0):.3f}, "
        f"asym={_auto_safe_float(winner.get('asym_penalty_db', 0.0), 0.0):.3f}, "
        f"method={selection_method}"
    )
    if callable(status_cb):
        status_cb(
            "CamillaFIR automatic mode: target finalize "
            f"(winner {str(winner.get('hc_mode', 'n/a'))}, "
            f"method {selection_method}, "
            f"rank {_auto_safe_float(dict(winner.get('best_metrics', {}) or {}).get('rank_score', 0.0), 0.0):.3f}, "
            f"avg {_auto_safe_float(dict(winner.get('best_metrics', {}) or {}).get('avg_score', 0.0), 0.0):.3f}, "
            f"pre {_auto_safe_float(winner.get('preselect_score', winner.get('fit_rms_db', 1e9)), 1e9):.3f}, "
            f"fit {_auto_safe_float(winner.get('fit_rms_db', 0.0), 0.0):.3f} dB)"
        )
    return {
        "selected_hc_mode": str(winner.get("hc_mode", quick.get("selected_hc_mode", "Harman6"))),
        "fit_rms_db": float(winner.get("fit_rms_db", quick.get("fit_rms_db", 0.0))),
        "offset_db": float(winner.get("offset_db", quick.get("offset_db", 0.0))),
        "selection_method": str(selection_method),
        "selection_basis": str(rank_basis),
        "auto_goal": str(goal),
        "top_n": int(len(shortlisted)),
        "trials_per_curve": int(trials_eff),
        "candidates": list(shortlisted),
        "evaluated": list(evaluated),
        "best_preset": dict(winner.get("best_preset", {}) or {}),
    }


def _auto_dsp_quality_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    real_rms = _auto_pick_metric(
        st,
        (
            "real_mag_error_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)
    dbg["real_rms"] = real_rms

    ripple_rms = _auto_pick_metric(
        st,
        (
            "ripple_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)
    dbg["ripple_rms"] = ripple_rms

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        penalty += 0.60 * max(0.0, float(gd_grad_max) - 12.0)
    dbg["gd_grad_max"] = gd_grad_max

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ),
    )
    if pre_ringing_db is not None:
        penalty += 0.70 * max(0.0, float(pre_ringing_db) + 40.0)
    dbg["pre_ringing_db"] = pre_ringing_db

    pre_post_ratio = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ),
        nonneg=True,
    )
    if pre_post_ratio is not None:
        penalty += 30.0 * max(0.0, float(pre_post_ratio) - 0.015)
    dbg["ir_pre_post_ratio"] = pre_post_ratio

    phase_boundary_mdb = _auto_pick_metric(
        st,
        (
            "phase_boundary_peak_mdb",
            "phase_corr_boundary_peak_mdb",
        ),
        abs_value=True,
        nonneg=True,
    )
    if phase_boundary_mdb is not None:
        penalty += 0.015 * max(0.0, float(phase_boundary_mdb) - 120.0)
    dbg["phase_boundary_peak_mdb"] = phase_boundary_mdb

    return float(max(0.0, penalty)), dbg


def _auto_excursion_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    exc_raw = st.get("exc_prot", None)
    exc_known = (exc_raw is not None)
    exc_on = bool(exc_raw) if exc_known else None
    exc_freq = _auto_safe_float(st.get("exc_freq", 0.0), 0.0)

    try:
        exc_bins = int(float(st.get("boost_candidate_bins_excprot", 0) or 0))
    except Exception:
        exc_bins = 0
    lf_boost_max = _auto_safe_float(st.get("lf_boost_max_db", 0.0), 0.0)
    pen_exc_off = 0.0
    pen_exc_invalid = 0.0
    pen_bins = 0.0
    pen_lf = 0.0

    # Prefer presets that keep excursion protection enabled.
    if exc_known and (exc_on is False):
        pen_exc_off = 2.0
        penalty += float(pen_exc_off)

    # If protection is enabled but configured too low/invalid, add a small penalty.
    if exc_known and (exc_on is True) and (not np.isfinite(exc_freq) or exc_freq <= 0.0):
        pen_exc_invalid = 0.8
        penalty += float(pen_exc_invalid)

    # Penalize tendency to boost in excursion-protected region.
    if exc_bins > 0:
        pen_bins = float(min(2.5, 0.10 * float(exc_bins)))
        penalty += float(pen_bins)

    # Penalize remaining LF boost inside guard region.
    pen_lf = float(min(12.0, 1.25 * max(0.0, float(lf_boost_max) - 1.5)))
    penalty += float(pen_lf)

    # Prevent a single excursion metric from collapsing rank to zero.
    penalty = min(16.0, float(penalty))

    dbg["exc_known"] = bool(exc_known)
    dbg["exc_on"] = exc_on
    dbg["exc_freq"] = float(exc_freq)
    dbg["exc_bins"] = int(exc_bins)
    dbg["lf_boost_max_db"] = float(lf_boost_max)
    dbg["pen_exc_off"] = float(pen_exc_off)
    dbg["pen_exc_invalid"] = float(pen_exc_invalid)
    dbg["pen_bins"] = float(pen_bins)
    dbg["pen_lf"] = float(pen_lf)
    dbg["pen_total_pre_cap"] = float(pen_exc_off + pen_exc_invalid + pen_bins + pen_lf)
    dbg["pen_total"] = float(penalty)
    return float(max(0.0, penalty)), dbg


def _auto_exc_penalty_bins_from_dbg(exc_dbg: dict | None) -> float:
    try:
        dbg = dict(exc_dbg or {})
    except Exception:
        dbg = {}
    v = _auto_safe_float(dbg.get("pen_bins", float("nan")), float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    exc_bins = int(_auto_safe_float(dbg.get("exc_bins", 0), 0.0))
    return float(min(2.5, 0.10 * max(0, exc_bins)))


def _auto_exc_zero_penalty_freq_hz_from_stats(st: dict | None) -> float:
    st = dict(st or {})
    v = _auto_safe_float(st.get("boost_candidate_min_hz", float("nan")), float("nan"))
    if not np.isfinite(v) or float(v) <= 0.0:
        return float("nan")
    return float(
        np.clip(
            float(v),
            float(_auto_safe_float(AUTO_MODE_EXC_MIN_HZ, 20.0)),
            float(_auto_safe_float(AUTO_MODE_EXC_MAX_HZ, 80.0)),
        )
    )


def _auto_focus_ripple_from_stats(
    st: dict | None,
    *,
    focus_lo_hz: float,
    focus_hi_hz: float,
) -> float | None:
    st = dict(st or {})
    lo = _auto_safe_float(focus_lo_hz, float("nan"))
    hi = _auto_safe_float(focus_hi_hz, float("nan"))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return None

    f = np.asarray(st.get("freq_axis", []), dtype=float).reshape(-1)
    g_pred = np.asarray(st.get("predicted_filter_mags", []), dtype=float).reshape(-1)
    g_real = np.asarray(
        st.get("realized_filter_mags", st.get("filter_mags", [])),
        dtype=float,
    ).reshape(-1)
    n = int(min(f.size, g_pred.size, g_real.size))
    if n < 8:
        return None
    f = np.asarray(f[:n], dtype=float)
    d = np.asarray(g_real[:n], dtype=float) - np.asarray(g_pred[:n], dtype=float)
    m = np.isfinite(f) & np.isfinite(d) & (f >= float(lo)) & (f <= float(hi))
    if int(np.count_nonzero(m)) < 8:
        return None
    dv = np.asarray(d[m], dtype=float)
    off = float(np.median(dv))
    d_shape = np.asarray(dv, dtype=float) - float(off)
    return float(np.sqrt(np.mean(d_shape * d_shape)))


def _auto_score_result(
    result,
    *,
    auto_exc_freq_hz: float | None = None,
    focus_lo_hz: float | None = None,
    focus_hi_hz: float | None = None,
    base_data: dict | None = None,
) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = plots.calc_ai_summary_from_stats(l_st)
    r_ai = plots.calc_ai_summary_from_stats(r_st)

    def _ai_score_with_fallback(st: dict, ai: dict) -> float:
        score = _auto_safe_float((ai or {}).get("score"), float("nan"))
        if np.isfinite(score):
            return float(score)
        try:
            conf = _auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", 0.0)),
                0.0,
            )
            rms_fb, match_fb = target_match_from_stats(
                st,
                include_filter=False,
                use_confidence=True,
                use_smart_scan_range=True,
            )
            if match_fb is None:
                return 0.0
            rt60 = st.get("rt60_val", None)
            rt_rel = st.get("rt60_reliability", None)
            return _auto_safe_float(
                plots.calc_acoustic_score(conf, float(match_fb), rt60_s=rt60, rt60_rel=rt_rel),
                0.0,
            )
        except Exception:
            return 0.0

    l_score = _ai_score_with_fallback(l_st, l_ai)
    r_score = _ai_score_with_fallback(r_st, r_ai)
    avg_score = (l_score + r_score) / 2.0
    lr_delta = abs(l_score - r_score)

    net_boost_max = max(
        _auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        _auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    l_refs = _auto_collect_reflections(l_st)
    r_refs = _auto_collect_reflections(r_st)
    events_total = int(len(l_refs) + len(r_refs))
    events_severity_l = _auto_event_severity(l_refs)
    events_severity_r = _auto_event_severity(r_refs)
    events_severity_raw = float(events_severity_l + events_severity_r)
    events_severity = float(math.log1p(max(0.0, events_severity_raw) / 6.0))
    dsp_pen_l, dsp_dbg_l = _auto_dsp_quality_penalty(l_st)
    dsp_pen_r, dsp_dbg_r = _auto_dsp_quality_penalty(r_st)
    dsp_penalty_raw = 0.5 * (float(dsp_pen_l) + float(dsp_pen_r))
    exc_pen_l, exc_dbg_l = _auto_excursion_penalty(l_st)
    exc_pen_r, exc_dbg_r = _auto_excursion_penalty(r_st)
    exc_penalty_raw_total = 0.5 * (float(exc_pen_l) + float(exc_pen_r))
    exc_penalty_bins_raw = 0.5 * (
        float(_auto_exc_penalty_bins_from_dbg(exc_dbg_l))
        + float(_auto_exc_penalty_bins_from_dbg(exc_dbg_r))
    )
    exc_penalty_raw = float(exc_penalty_raw_total)
    exc_penalty_waived = bool(np.isfinite(_auto_safe_float(auto_exc_freq_hz, float("nan"))))
    auto_exc_zero_l = _auto_exc_zero_penalty_freq_hz_from_stats(l_st)
    auto_exc_zero_r = _auto_exc_zero_penalty_freq_hz_from_stats(r_st)
    auto_exc_zero_vals = [float(v) for v in (auto_exc_zero_l, auto_exc_zero_r) if np.isfinite(v)]
    auto_exc_zero_penalty_hz = float(min(auto_exc_zero_vals)) if auto_exc_zero_vals else float("nan")
    auto_exc_hz_now = _auto_safe_float(auto_exc_freq_hz, float("nan"))
    exc_penalty_bins_waived = False
    if (
        bool(exc_penalty_waived)
        and np.isfinite(auto_exc_zero_penalty_hz)
        and np.isfinite(auto_exc_hz_now)
        and float(auto_exc_hz_now) > (float(auto_exc_zero_penalty_hz) + 1e-6)
    ):
        exc_penalty_raw = max(0.0, float(exc_penalty_raw_total) - float(exc_penalty_bins_raw))
        exc_penalty_bins_waived = bool(float(exc_penalty_bins_raw) > 1e-9)
    # Auto-excursion frequency is a good sign, but don't fully "waive" excursion risk.
    # Scale down the remaining excursion risk so auto-mode won't over-focus on LF guard heuristics.
    exc_penalty = float(exc_penalty_raw) * (0.35 if exc_penalty_waived else 1.0)

    # Penalty budget normalization (avoid single-term dominance)
    # Smooth the 5 dB boost "knee" to avoid hard-threshold behavior in optimization.
    _BOOST_KNEE_DB = 1.0  # larger = softer knee (e.g. 0.7..1.5)
    _x = (float(net_boost_max) - 5.0) / float(_BOOST_KNEE_DB)
    _x = float(np.clip(_x, -60.0, 60.0))  # guard exp overflow
    _soft_hinge_db = float(_BOOST_KNEE_DB) * float(np.log1p(np.exp(_x)))
    boost_pen = min(12.0, 1.25 * _soft_hinge_db)
    dsp_penalty = min(12.0, 0.07 * float(dsp_penalty_raw))
    all_events = list(l_refs) + list(r_refs)
    event_pen_raw = _auto_event_penalty_weighted(
        all_events,
        base_per_event=float(_auto_safe_float(AUTO_MODE_EVENT_PEN_BASE_PER_EVENT, 0.5)),
        dt_weight=float(_auto_safe_float(AUTO_MODE_EVENT_PEN_DT_WEIGHT, 0.02)),
        power=float(_auto_safe_float(AUTO_MODE_EVENT_PEN_DT_POWER, 2.0)),
        dt_ref_ms=float(_auto_safe_float(AUTO_MODE_EVENT_PEN_DT_REF_MS, 100.0)),
    )
    event_pen_conf_scale = 1.0
    if bool(AUTO_MODE_EVENT_PEN_CONF_GATE_ENABLE):
        conf_vals = []
        for st in (l_st, r_st):
            c = _auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", float("nan"))),
                float("nan"),
            )
            if not np.isfinite(c):
                continue
            # Accept both normalized (0..1) and percentage (0..100) confidence.
            c01 = float(c / 100.0) if float(c) > 1.5 else float(c)
            c01 = float(np.clip(c01, 0.0, 1.0))
            conf_vals.append(float(c01))
        if conf_vals:
            conf_mean = float(np.mean(np.asarray(conf_vals, dtype=float)))
            min_scale = float(np.clip(_auto_safe_float(AUTO_MODE_EVENT_PEN_CONF_GATE_MIN_SCALE, 0.45), 0.0, 1.0))
            full_conf = float(np.clip(_auto_safe_float(AUTO_MODE_EVENT_PEN_CONF_GATE_FULL_CONF, 0.85), 1e-6, 1.0))
            conf_norm = float(np.clip(conf_mean / full_conf, 0.0, 1.0))
            event_pen_conf_scale = float(min_scale + (1.0 - min_scale) * conf_norm)
    event_pen_raw *= float(event_pen_conf_scale)
    event_pen = min(12.0, max(0.0, event_pen_raw))
    lr_pen = min(4.0, 0.03 * lr_delta)
    exc_penalty = min(12.0, float(exc_penalty))
    filter_key = _auto_filter_cache_key(base_data)
    phase_limit_used_hz = _auto_safe_float((base_data or {}).get("phase_limit", float("nan")), float("nan"))
    if _auto_is_phase_search_filter(filter_key):
        phase_limit_used_hz = float(_auto_phase_limit_clip(phase_limit_used_hz, default=AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ))
    phase_limit_penalty = float(
        _auto_phase_limit_prior_penalty(phase_limit_used_hz, filter_key=filter_key)
    )
    def _rank_scale(v: float) -> float:
        g = float(_auto_safe_float(AUTO_MODE_RANK_SCORE_GAIN, 1.0))
        b = float(_auto_safe_float(AUTO_MODE_RANK_SCORE_BIAS, 0.0))
        return float(np.clip(float(g) * float(v) + float(b), 0.0, 100.0))

    rank_raw = float(
        avg_score
        - boost_pen
        - event_pen
        - lr_pen
        - dsp_penalty
        - exc_penalty
        - phase_limit_penalty
    )
    rank_score_base = float(_rank_scale(rank_raw))
    rank_score = float(_rank_scale(rank_raw))
    focus_ripple_l = None
    focus_ripple_r = None
    flo = _auto_safe_float(focus_lo_hz, float("nan"))
    fhi = _auto_safe_float(focus_hi_hz, float("nan"))
    if np.isfinite(flo) and np.isfinite(fhi) and float(fhi) > float(flo):
        focus_ripple_l = _auto_focus_ripple_from_stats(
            l_st,
            focus_lo_hz=float(flo),
            focus_hi_hz=float(fhi),
        )
        focus_ripple_r = _auto_focus_ripple_from_stats(
            r_st,
            focus_lo_hz=float(flo),
            focus_hi_hz=float(fhi),
        )
    if not (np.isfinite(_auto_safe_float(focus_ripple_l, float("nan"))) or np.isfinite(_auto_safe_float(focus_ripple_r, float("nan")))):
        focus_ripple_keys = (
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
            "ripple_rms",
        )
        focus_ripple_l = _auto_pick_metric(
            l_st,
            focus_ripple_keys,
            abs_value=True,
            nonneg=True,
        )
        focus_ripple_r = _auto_pick_metric(
            r_st,
            focus_ripple_keys,
            abs_value=True,
            nonneg=True,
        )
    focus_ripple_vals = []
    for v in (focus_ripple_l, focus_ripple_r):
        x = _auto_safe_float(v, float("nan"))
        if np.isfinite(x):
            focus_ripple_vals.append(float(x))
    focus_ripple = float(np.mean(np.asarray(focus_ripple_vals, dtype=float))) if focus_ripple_vals else 0.0
    # Focus on strongest 1-2 resonances, not just a single worst one.
    top_modes = []
    try:
        if bool(AUTO_MODE_DUAL_MODE_ENABLED):
            top_modes = _auto_get_top_modes_hz(
                result,
                top_n=int(AUTO_MODE_DUAL_MODE_TOP_N),
            )
        else:
            m1 = _auto_get_worst_mode_hz(result)
            top_modes = [float(m1)] if m1 is not None else []
    except Exception:
        top_modes = []

    mode_hz = _auto_safe_float(
        (top_modes[0] if len(top_modes) >= 1 else _auto_get_worst_mode_hz(result)),
        float("nan"),
    )
    mode_band_lo = float("nan")
    mode_band_hi = float("nan")
    mode_ripple_db = float("nan")
    mode2_hz = float("nan")
    mode2_band_lo = float("nan")
    mode2_band_hi = float("nan")
    mode2_ripple_db = float("nan")
    mode_band = _auto_mode_band(mode_hz, base_data=base_data) if np.isfinite(mode_hz) else None
    if isinstance(mode_band, tuple) and len(mode_band) == 2:
        mode_band_lo = float(_auto_safe_float(mode_band[0], float("nan")))
        mode_band_hi = float(_auto_safe_float(mode_band[1], float("nan")))
        if np.isfinite(mode_band_lo) and np.isfinite(mode_band_hi) and (mode_band_hi > mode_band_lo):
            mr_l = _auto_focus_ripple_from_stats(
                l_st,
                focus_lo_hz=float(mode_band_lo),
                focus_hi_hz=float(mode_band_hi),
            )
            mr_r = _auto_focus_ripple_from_stats(
                r_st,
                focus_lo_hz=float(mode_band_lo),
                focus_hi_hz=float(mode_band_hi),
            )
            mr_vals = []
            for mv in (mr_l, mr_r):
                x = _auto_safe_float(mv, float("nan"))
                if np.isfinite(x):
                    mr_vals.append(float(x))
            if mr_vals:
                mode_ripple_db = float(np.mean(np.asarray(mr_vals, dtype=float)))
    if (not np.isfinite(mode_ripple_db)) and np.isfinite(mode_hz):
        # Resonance found, but mode-band curve metric unavailable -> fallback to focus-band ripple.
        mode_ripple_db = float(_auto_safe_float(focus_ripple, float("nan")))

    # Secondary mode (if present).
    if len(top_modes) >= 2:
        mode2_hz = float(_auto_safe_float(top_modes[1], float("nan")))
        mode2_band = _auto_mode_band(mode2_hz, base_data=base_data) if np.isfinite(mode2_hz) else None
        if isinstance(mode2_band, tuple) and len(mode2_band) == 2:
            mode2_band_lo = float(_auto_safe_float(mode2_band[0], float("nan")))
            mode2_band_hi = float(_auto_safe_float(mode2_band[1], float("nan")))
            if np.isfinite(mode2_band_lo) and np.isfinite(mode2_band_hi) and (mode2_band_hi > mode2_band_lo):
                mr2_l = _auto_focus_ripple_from_stats(
                    l_st,
                    focus_lo_hz=float(mode2_band_lo),
                    focus_hi_hz=float(mode2_band_hi),
                )
                mr2_r = _auto_focus_ripple_from_stats(
                    r_st,
                    focus_lo_hz=float(mode2_band_lo),
                    focus_hi_hz=float(mode2_band_hi),
                )
                mr2_vals = []
                for mv in (mr2_l, mr2_r):
                    x = _auto_safe_float(mv, float("nan"))
                    if np.isfinite(x):
                        mr2_vals.append(float(x))
                if mr2_vals:
                    mode2_ripple_db = float(np.mean(np.asarray(mr2_vals, dtype=float)))
        if (not np.isfinite(mode2_ripple_db)) and np.isfinite(mode2_hz):
            mode2_ripple_db = float(_auto_safe_float(focus_ripple, float("nan")))

    # Mode-aware penalty (small but consistent): penalize if ripple is above a "good" baseline.
    mode_r1 = _auto_safe_float(mode_ripple_db, float("nan"))
    mode_r2 = _auto_safe_float(mode2_ripple_db, float("nan"))
    mode_combined = float("nan")
    if np.isfinite(mode_r1) and np.isfinite(mode_r2):
        mode_combined = max(
            float(mode_r1),
            float(AUTO_MODE_MODE_RIPPLE_SECONDARY_W) * float(mode_r2),
        )
    elif np.isfinite(mode_r1):
        mode_combined = float(mode_r1)
    elif np.isfinite(mode_r2):
        mode_combined = float(mode_r2)

    mode_penalty = 0.0
    if np.isfinite(mode_combined):
        mode_penalty = float(AUTO_MODE_MODE_RIPPLE_PENALTY_W) * max(
            0.0,
            float(mode_combined) - float(AUTO_MODE_MODE_RIPPLE_OK_DB),
        )
        mode_penalty = float(np.clip(mode_penalty, 0.0, 6.0))

    # Apply mode penalty after base rank.
    if mode_penalty > 0.0:
        rank_raw = float(rank_raw - float(mode_penalty))
        rank_score = float(_rank_scale(rank_raw))
    realized_keys = (
        "post_to_ir_staged_shape_delta_rms_20_200_db",
        "post_to_ir_shape_delta_rms_20_200_db",
        "post_to_ir_delta_rms_20_200_db",
    )
    realized_l = _auto_pick_metric(l_st, realized_keys, abs_value=True, nonneg=True)
    realized_r = _auto_pick_metric(r_st, realized_keys, abs_value=True, nonneg=True)
    realized_vals = []
    for rv in (realized_l, realized_r):
        x = _auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            realized_vals.append(float(x))
    realized_rms_20_200 = (
        float(np.mean(np.asarray(realized_vals, dtype=float)))
        if realized_vals
        else float("nan")
    )
    ripple_raw_l = _auto_pick_metric(l_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_r = _auto_pick_metric(r_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_vals = []
    for rv in (ripple_raw_l, ripple_raw_r):
        x = _auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            ripple_raw_vals.append(float(x))
    ripple_raw = (
        float(np.mean(np.asarray(ripple_raw_vals, dtype=float)))
        if ripple_raw_vals
        else float("nan")
    )
    pre_post_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    pre_post_l = None if bool(l_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        l_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_r = None if bool(r_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        r_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_l_f = _auto_safe_float(pre_post_l, float("nan"))
    pre_post_r_f = _auto_safe_float(pre_post_r, float("nan"))
    pre_post_max = float("nan")
    pre_post_vals = []
    if np.isfinite(pre_post_l_f):
        pre_post_vals.append(float(pre_post_l_f))
    if np.isfinite(pre_post_r_f):
        pre_post_vals.append(float(pre_post_r_f))
    if pre_post_vals:
        pre_post_max = float(max(pre_post_vals))

    return {
        "rank_score": float(rank_score),
        "rank_score_base": float(rank_score_base),
        "avg_score": float(avg_score),
        "focus_ripple_db": float(focus_ripple or 0.0),
        "mode_hz": float(mode_hz) if np.isfinite(mode_hz) else float("nan"),
        "mode_band_lo": float(mode_band_lo) if np.isfinite(mode_band_lo) else float("nan"),
        "mode_band_hi": float(mode_band_hi) if np.isfinite(mode_band_hi) else float("nan"),
        "mode_ripple_db": float(mode_ripple_db) if np.isfinite(mode_ripple_db) else float("nan"),
        "mode2_hz": float(mode2_hz) if np.isfinite(mode2_hz) else float("nan"),
        "mode2_band_lo": float(mode2_band_lo) if np.isfinite(mode2_band_lo) else float("nan"),
        "mode2_band_hi": float(mode2_band_hi) if np.isfinite(mode2_band_hi) else float("nan"),
        "mode2_ripple_db": float(mode2_ripple_db) if np.isfinite(mode2_ripple_db) else float("nan"),
        "mode_ripple_combined_db": float(mode_combined) if np.isfinite(mode_combined) else float("nan"),
        "mode_penalty": float(mode_penalty),
        "realized_rms_20_200_db": float(realized_rms_20_200) if np.isfinite(realized_rms_20_200) else float("nan"),
        "ir_pre_post_energy_ratio_l": float(pre_post_l_f) if np.isfinite(pre_post_l_f) else float("nan"),
        "ir_pre_post_energy_ratio_r": float(pre_post_r_f) if np.isfinite(pre_post_r_f) else float("nan"),
        "ir_pre_post_energy_ratio_max": float(pre_post_max) if np.isfinite(pre_post_max) else float("nan"),
        "ripple_rms": float(ripple_raw) if np.isfinite(ripple_raw) else float("nan"),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "events_total": int(events_total),
        "events_severity": float(events_severity),
        "events_severity_raw": float(events_severity_raw),
        "events_severity_l": float(events_severity_l),
        "events_severity_r": float(events_severity_r),
        "event_penalty": float(event_pen),
        "dsp_penalty": float(dsp_penalty),
        "dsp_penalty_raw": float(dsp_penalty_raw),
        "dsp_penalty_l": float(dsp_pen_l),
        "dsp_penalty_r": float(dsp_pen_r),
        "exc_penalty": float(exc_penalty),
        "exc_penalty_raw": float(exc_penalty_raw),
        "exc_penalty_raw_total": float(exc_penalty_raw_total),
        "exc_penalty_bins_raw": float(exc_penalty_bins_raw),
        "exc_penalty_bins_waived": bool(exc_penalty_bins_waived),
        "exc_penalty_waived": bool(exc_penalty_waived),
        "exc_penalty_l": float(exc_pen_l),
        "exc_penalty_r": float(exc_pen_r),
        "auto_exc_zero_penalty_hz": (
            float(auto_exc_zero_penalty_hz)
            if np.isfinite(auto_exc_zero_penalty_hz)
            else float("nan")
        ),
        "phase_limit_hz": float(phase_limit_used_hz) if np.isfinite(phase_limit_used_hz) else float("nan"),
        "phase_limit_penalty": float(phase_limit_penalty),
        "dsp_dbg_l": dict(dsp_dbg_l),
        "dsp_dbg_r": dict(dsp_dbg_r),
        "exc_dbg_l": dict(exc_dbg_l),
        "exc_dbg_r": dict(exc_dbg_r),
    }


def _auto_rank_key(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_room_safe(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
    )


def _auto_rank_key_low_ripple(metrics: dict) -> tuple:
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    ripple_fallback = _auto_safe_float(metrics.get("focus_ripple_db"), float("inf"))
    realized_lf = _auto_realized_rms_20_200_for_pareto(metrics)
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        mode_ripple if np.isfinite(mode_ripple) else ripple_fallback,
        realized_lf,
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("mixed_freq_penalty"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_flat(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_realized_rms_20_200_for_pareto(metrics),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
    )


def _auto_rank_key_acoustic(metrics: dict) -> tuple:
    # Legacy alias for older goal name.
    return _auto_rank_key_flat(metrics)


def _auto_rank_key_hybrid(metrics: dict) -> tuple:
    # Legacy alias for older goal name.
    return _auto_rank_key_low_ripple(metrics)


def _auto_hybrid_mixed_freq_penalty(
    preset: dict | None,
    *,
    base_data: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> float:
    if _auto_goal_norm(goal) != AUTO_MODE_GOAL_LOW_RIPPLE:
        return 0.0

    p = dict(base_data or {})
    p.update(dict(preset or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    if "mixed" not in ft:
        return 0.0
    if not bool(p.get("bass_first_ai", True)):
        return 0.0

    mixed_freq = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
    if not np.isfinite(mixed_freq):
        return 0.0
    pen = max(
        0.0,
        (float(mixed_freq) - float(AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ))
        / float(max(1e-6, AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ)),
    )
    return float(np.clip(pen, 0.0, 2.5))


def _auto_apply_goal_tiebreak_metrics(
    metrics: dict,
    *,
    preset: dict | None,
    base_data: dict | None,
    goal: str,
) -> dict:
    out = dict(metrics or {})
    out["mixed_freq_penalty"] = float(
        _auto_hybrid_mixed_freq_penalty(preset, base_data=base_data, goal=goal)
    )
    return out


def _auto_build_refine_profile(
    *,
    base_data: dict,
    phase1_top: list,
) -> dict:
    """
    Build data-driven refine search profile.
    Works for subs, mains, nearfield etc.
    """

    # --- Collect mixed_freq from phase1 winners ---
    mixed_vals = []
    tdc_vals = []

    for it in (phase1_top or []):
        p = dict(it.get("preset", {}) or {})
        mf = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
        td = _auto_safe_float(p.get("tdc_strength", float("nan")), float("nan"))
        if np.isfinite(mf):
            mixed_vals.append(float(mf))
        if np.isfinite(td):
            tdc_vals.append(float(td))

    if not mixed_vals:
        # fallback: generic
        mixed_center = 120.0
        focus_lo = float(max(20.0, float(mixed_center) - 70.0))
        focus_hi = float(min(220.0, float(mixed_center) + 50.0))
        bf_hi = float("nan")
        if bool(base_data.get("bass_first_ai", True)):
            bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
            if np.isfinite(bf_hi):
                focus_hi = min(float(focus_hi), float(bf_hi))
        focus_lo = float(np.clip(focus_lo, 20.0, 200.0))
        focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
        if focus_hi <= focus_lo:
            focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
        if focus_hi <= focus_lo:
            focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))
        return {
            "mixed_center": float(mixed_center),
            "mixed_span": 60.0,
            "focus_lo": float(focus_lo),
            "focus_hi": float(focus_hi),
            "tdc_lo": 45.0,
            "tdc_hi": 70.0,
        }

    mixed_center = float(np.median(mixed_vals))
    mixed_spread = float(np.std(mixed_vals)) if len(mixed_vals) > 1 else 20.0

    # clamp spread to reasonable bounds
    mixed_span = float(np.clip(mixed_spread * 1.5, 25.0, 80.0))
    focus_lo = float(max(20.0, float(mixed_center) - 70.0))
    focus_hi = float(min(220.0, float(mixed_center) + 50.0))
    bf_hi = float("nan")
    if bool(base_data.get("bass_first_ai", True)):
        bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
    focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
    if np.isfinite(bf_hi):
        focus_hi = min(float(focus_hi), float(bf_hi))
    if focus_hi <= focus_lo:
        focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
    if focus_hi <= focus_lo:
        focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))

    if tdc_vals:
        tdc_center = float(np.median(tdc_vals))
    else:
        tdc_center = 60.0

    return {
        "mixed_center": mixed_center,
        "mixed_span": mixed_span,
        "focus_lo": float(focus_lo),
        "focus_hi": float(focus_hi),
        "tdc_lo": float(np.clip(tdc_center - 12.0, 35.0, 80.0)),
        "tdc_hi": float(np.clip(tdc_center + 12.0, 40.0, 85.0)),
    }


def _auto_goal_uses_local_refine(goal: str | None) -> bool:
    g = _auto_goal_norm(goal)
    return bool(
        g in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
        )
    )


def _auto_rank_key_goal(metrics: dict, goal: str = AUTO_MODE_GOAL_DEFAULT) -> tuple:
    g = _auto_goal_norm(goal)
    if g == AUTO_MODE_GOAL_FLAT:
        return _auto_rank_key_flat(metrics)
    if g == AUTO_MODE_GOAL_ROOM_SAFE:
        return _auto_rank_key_room_safe(metrics)
    if g == AUTO_MODE_GOAL_LOW_RIPPLE:
        return _auto_rank_key_low_ripple(metrics)
    return _auto_rank_key(metrics)


def _auto_is_better_refine(
    new_metrics: dict,
    best_metrics: dict,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    *,
    return_reason: bool = False,
) -> bool | tuple[bool, str]:
    new_m = dict(new_metrics or {})
    best_m = dict(best_metrics or {})

    rank_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RANK_EPS, 0.20)))
    ripple_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS, 0.02)))
    mode_guard_gain = float(
        max(0.0, _auto_safe_float(AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB, 0.06))
    )

    new_rank_raw = _auto_safe_float(new_m.get("rank_score"), 0.0)
    best_rank_raw = _auto_safe_float(best_m.get("rank_score"), 0.0)
    raw_rank_diff = float(new_rank_raw - best_rank_raw)
    new_rank_ref = _auto_safe_float(new_m.get("rank_score_refine", new_rank_raw), new_rank_raw)
    best_rank_ref = _auto_safe_float(best_m.get("rank_score_refine", best_rank_raw), best_rank_raw)
    ref_rank_diff = float(new_rank_ref - best_rank_ref)

    new_mode_ripple = _auto_safe_float(new_m.get("mode_ripple_db"), float("nan"))
    best_mode_ripple = _auto_safe_float(best_m.get("mode_ripple_db"), float("nan"))
    mode_pair_ok = bool(np.isfinite(new_mode_ripple) and np.isfinite(best_mode_ripple))

    if ref_rank_diff > 1e-9:
        if abs(raw_rank_diff) <= rank_eps and mode_pair_ok:
            mode_improve = float(best_mode_ripple - new_mode_ripple)
            if mode_improve > ripple_eps:
                new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                boost_rise = float(new_boost - best_boost)
                if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                    out = (False, "mode_guard")
                else:
                    out = (True, "mode_ripple")
            else:
                out = (True, "rank_refine")
        else:
            out = (True, "rank_refine")
    elif ref_rank_diff < -1e-9:
        if abs(raw_rank_diff) <= rank_eps and mode_pair_ok and (float(new_mode_ripple - best_mode_ripple) > ripple_eps):
            out = (False, "mode_ripple")
        else:
            out = (False, "rank_refine")
    else:
        out = (False, "rank_tie")
        if bool(AUTO_MODE_REFINE_TIEBREAK_ENABLE):
            if mode_pair_ok:
                mode_improve = float(best_mode_ripple - new_mode_ripple)
                if mode_improve > ripple_eps:
                    new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                    best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                    boost_rise = float(new_boost - best_boost)
                    if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                        out = (False, "mode_guard")
                    else:
                        out = (True, "mode_ripple")
                elif float(new_mode_ripple - best_mode_ripple) > ripple_eps:
                    out = (False, "mode_ripple")

            if out[1] == "rank_tie":
                new_ripple = _auto_safe_float(new_m.get("focus_ripple_db"), float("nan"))
                best_ripple = _auto_safe_float(best_m.get("focus_ripple_db"), float("nan"))
                if np.isfinite(new_ripple) and np.isfinite(best_ripple):
                    if float(best_ripple - new_ripple) > ripple_eps:
                        out = (True, "focus_ripple")
                    elif float(new_ripple - best_ripple) > ripple_eps:
                        out = (False, "focus_ripple")

        if out[1] == "rank_tie":
            out = (
                bool(_auto_rank_key_goal(new_m, goal) < _auto_rank_key_goal(best_m, goal)),
                "goal_key",
            )

    return out if bool(return_reason) else bool(out[0])


def _m(metrics: dict | None, key: str, default=float("nan")) -> float:
    try:
        v = float((metrics or {}).get(key, default))
    except Exception:
        v = float(default)
    return float(v)


def _auto_mode_ripple_for_pareto(metrics: dict | None) -> float:
    for k in ("mode_ripple_db", "focus_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_realized_rms_20_200_for_pareto(metrics: dict | None) -> float:
    v = _m(metrics, "realized_rms_20_200_db", float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    return float("inf")


def _auto_prepost_lr_for_pareto(metrics: dict | None) -> tuple[float, float, float]:
    m = dict(metrics or {})
    l = _m(m, "ir_pre_post_energy_ratio_l", float("nan"))
    r = _m(m, "ir_pre_post_energy_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(m, "ir_pre_post_ratio_l", float("nan"))
    if not np.isfinite(r):
        r = _m(m, "ir_pre_post_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(dict(m.get("dsp_dbg_l", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(r):
        r = _m(dict(m.get("dsp_dbg_r", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(l):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            l = float(global_v)
    if not np.isfinite(r):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            r = float(global_v)

    vals = [float(v) for v in (l, r) if np.isfinite(v)]
    mx = float(max(vals)) if vals else float("inf")
    return (
        float(l) if np.isfinite(l) else float("nan"),
        float(r) if np.isfinite(r) else float("nan"),
        float(mx),
    )


def _auto_prepost_for_pareto(metrics: dict | None) -> float:
    _, _, mx = _auto_prepost_lr_for_pareto(metrics)
    if np.isfinite(mx):
        return float(max(0.0, mx))
    return float("inf")


def _auto_ripple_metric_for_gate(metrics: dict | None) -> float:
    # Prefer dedicated focus-band ripple; fallback to mode-band / global ripple.
    for k in ("focus_ripple_db", "mode_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_gate_threshold(values: list[float], keep_fraction: float) -> float:
    vals = [float(v) for v in (values or []) if np.isfinite(v)]
    if not vals:
        return float("inf")
    kf = float(np.clip(_auto_safe_float(keep_fraction, 1.0), 0.05, 1.0))
    vals = sorted(vals)
    idx = int(np.floor((len(vals) - 1) * kf))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def _auto_phase2_hard_gate_pool(
    pool: list[dict],
    *,
    min_keep: int = AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    keep_event_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    keep_ripple_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    fallback_to_rank: bool = AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
) -> tuple[list[dict], float, float]:
    """
    Hard-gate phase2 pool before Pareto:
    - compute thresholds for event severity and ripple
    - keep candidates that are <= both thresholds
    - if too few, relax conservatively (first OR, then rank fallback)
    Returns (kept_pool, event_threshold, ripple_threshold).
    """
    if not isinstance(pool, list) or not pool:
        return [], float("inf"), float("inf")
    n_in = int(len(pool))
    min_keep = int(max(1, min_keep))

    # Only meaningful if we have headroom.
    if n_in <= (min_keep + 2):
        return [dict(x or {}) for x in pool], float("inf"), float("inf")

    ev = []
    rp = []
    for it in pool:
        m = dict((it or {}).get("metrics", {}) or {})
        ev.append(_m(m, "events_severity", float("nan")))
        rp.append(_auto_ripple_metric_for_gate(m))

    ev_thr = _auto_gate_threshold(ev, float(keep_event_fraction))
    rp_thr = _auto_gate_threshold(rp, float(keep_ripple_fraction))

    gated = []
    gated_or = []
    for it in pool:
        m = dict((it or {}).get("metrics", {}) or {})
        ev_i = _m(m, "events_severity", float("inf"))
        rp_i = _auto_ripple_metric_for_gate(m)
        ok_ev = bool(np.isfinite(ev_i) and float(ev_i) <= float(ev_thr))
        ok_rp = bool(np.isfinite(rp_i) and float(rp_i) <= float(rp_thr))
        if ok_ev and ok_rp:
            gated.append(dict(it or {}))
        if ok_ev or ok_rp:
            gated_or.append(dict(it or {}))

    if len(gated) >= min_keep:
        return gated, float(ev_thr), float(rp_thr)
    if len(gated_or) >= min_keep:
        return gated_or, float(ev_thr), float(rp_thr)

    if bool(fallback_to_rank):
        kept = sorted(
            [dict(x or {}) for x in pool],
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "rank_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )[:min_keep]
        return kept, float(ev_thr), float(rp_thr)

    kept = gated_or or gated or [dict(x or {}) for x in pool]
    return kept, float(ev_thr), float(rp_thr)


def _auto_adaptive_shrink_factor(
    phase1_top: list[dict],
    *,
    base_shrink: float,
    plateau_hit: bool,
) -> float:
    """
    Derive tighter/looser shrink factor from phase1 stability.
    Lower = tighter search around anchors.
    """
    if not bool(AUTO_MODE_ADAPTIVE_SHRINK_ENABLED):
        return float(base_shrink)

    base = float(np.clip(_auto_safe_float(base_shrink, 0.35), 0.05, 1.0))
    if not isinstance(phase1_top, list) or len(phase1_top) < 2:
        if bool(plateau_hit):
            return float(np.clip(base * 0.85, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))
        return float(np.clip(base, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))

    mixed = []
    tdc = []
    fdw = []
    reg = []
    for it in phase1_top[:4]:
        p = dict((it or {}).get("preset", {}) or {})
        mixed.append(_auto_safe_float(p.get("mixed_freq", float("nan")), float("nan")))
        tdc.append(_auto_safe_float(p.get("tdc_strength", float("nan")), float("nan")))
        fdw.append(_auto_safe_float(p.get("fdw_cycles", float("nan")), float("nan")))
        reg.append(_auto_safe_float(p.get("reg_strength", float("nan")), float("nan")))

    def _spread(vals: list[float]) -> float:
        vv = [float(v) for v in vals if np.isfinite(v)]
        if len(vv) < 2:
            return 0.0
        vv = sorted(vv)
        return float(vv[-1] - vv[0])

    spread_score = 0.0
    spread_score += _spread(mixed) / 80.0
    spread_score += _spread(tdc) / 15.0
    spread_score += _spread(fdw) / 3.0
    spread_score += _spread(reg) / 20.0

    if spread_score <= 0.35:
        mul = 0.75
    elif spread_score <= 0.70:
        mul = 0.85
    elif spread_score <= 1.10:
        mul = 0.95
    else:
        mul = 1.05

    if bool(plateau_hit):
        mul *= 0.90

    out = float(base * mul)
    return float(np.clip(out, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))


def _pareto_dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    le_all = True
    lt_any = False
    for ai, bi in zip(a, b):
        if ai > bi:
            le_all = False
            break
        if ai < bi:
            lt_any = True
    return bool(le_all and lt_any)


def _auto_phase2_pareto_vector(metrics: dict | None) -> tuple[float, float, float, float, float]:
    avg = _m(metrics, "avg_score", float("nan"))
    neg_avg = -float(avg) if np.isfinite(avg) else float("inf")
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    rms_20_200 = _auto_realized_rms_20_200_for_pareto(metrics)
    net_boost = _m(metrics, "max_net_boost_db", float("nan"))
    net_boost = float(net_boost) if np.isfinite(net_boost) else float("inf")
    prepost = _auto_prepost_for_pareto(metrics)
    return (float(neg_avg), float(mode_ripple), float(rms_20_200), float(net_boost), float(prepost))


def _auto_phase2_pareto_front(pool: list[dict]) -> list[dict]:
    front = []
    if not isinstance(pool, list) or not pool:
        return front
    vectors = [_auto_phase2_pareto_vector(dict(it.get("metrics", {}) or {})) for it in pool]
    for i, cand in enumerate(pool):
        dominated = False
        for j, other in enumerate(pool):
            if i == j:
                continue
            if _pareto_dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            front.append(dict(cand or {}))
    return front


def _auto_phase2_pick_pareto_winner(
    front: list[dict],
    pool: list[dict],
    *,
    acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
) -> dict | None:
    def _lex_better(a: dict, b: dict) -> bool:
        ma = dict(a.get("metrics", {}) or {})
        mb = dict(b.get("metrics", {}) or {})

        # 1) maximize avg_score
        avg_a = _m(ma, "avg_score", float("-inf"))
        avg_b = _m(mb, "avg_score", float("-inf"))
        if float(avg_a) > float(avg_b):
            return True
        if float(avg_a) < float(avg_b):
            return False

        # 2) minimize pre/post energy ratio (temporal cleanliness), with tolerance
        prepost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_PREPOST_EPS, 0.002)))
        prepost_a = _auto_prepost_for_pareto(ma)
        prepost_b = _auto_prepost_for_pareto(mb)
        if float(prepost_a) < float(prepost_b) - float(prepost_eps):
            return True
        if float(prepost_b) < float(prepost_a) - float(prepost_eps):
            return False

        # 3) minimize mode_ripple_db, but ignore micro deltas
        mode_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS, 0.005)))
        mode_a = _auto_mode_ripple_for_pareto(ma)
        mode_b = _auto_mode_ripple_for_pareto(mb)
        if float(mode_a) < float(mode_b) - float(mode_eps):
            return True
        if float(mode_b) < float(mode_a) - float(mode_eps):
            return False

        # 4) minimize realized_rms_20_200, also with tolerance
        rms_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS, 0.003)))
        rms_a = _auto_realized_rms_20_200_for_pareto(ma)
        rms_b = _auto_realized_rms_20_200_for_pareto(mb)
        if float(rms_a) < float(rms_b) - float(rms_eps):
            return True
        if float(rms_b) < float(rms_a) - float(rms_eps):
            return False

        # 5) minimize net boost
        boost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_BOOST_EPS, 0.02)))
        boost_a = _m(ma, "max_net_boost_db", float("inf"))
        boost_b = _m(mb, "max_net_boost_db", float("inf"))
        if float(boost_a) < float(boost_b) - float(boost_eps):
            return True
        if float(boost_b) < float(boost_a) - float(boost_eps):
            return False

        # 6) deterministic final fallback
        return bool(_auto_rank_key(ma) < _auto_rank_key(mb))

    front_list = [dict(x or {}) for x in (front or []) if isinstance(x, dict)]
    pool_list = [dict(x or {}) for x in (pool or []) if isinstance(x, dict)]
    if not front_list:
        return None

    avg_vals = [
        _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
        for it in pool_list
    ]
    avg_vals = [float(v) for v in avg_vals if np.isfinite(v)]
    best_avg = max(avg_vals) if avg_vals else float("nan")
    drop = float(max(0.0, _auto_safe_float(acoustic_drop, AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP)))

    # S = {c in front | c.avg_score >= best_avg - AVG_TOL}
    acceptable: list[dict] = []
    if np.isfinite(best_avg):
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg) and float(avg) >= float(best_avg) - float(drop):
                acceptable.append(dict(it))
    choose_from = acceptable

    # If S is empty -> S = {argmax avg_score in front}
    if not choose_from:
        front_with_avg = []
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg):
                front_with_avg.append((float(avg), dict(it)))
        if front_with_avg:
            front_with_avg = sorted(
                front_with_avg,
                key=lambda t: (
                    -float(t[0]),
                    _auto_rank_key(dict((t[1] or {}).get("metrics", {}) or {})),
                ),
            )
            choose_from = [dict(front_with_avg[0][1])]
        else:
            choose_from = list(front_list)

    if choose_from:
        winner = dict(choose_from[0])
        for cand in choose_from[1:]:
            cand_d = dict(cand or {})
            if _lex_better(cand_d, winner):
                winner = cand_d
        return dict(winner)

    # Fallback: highest avg_score from pool, then normal rank key.
    if pool_list:
        pool_sorted = sorted(
            pool_list,
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "avg_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )
        return dict(pool_sorted[0])
    return None


def _auto_reject(metrics: dict, st_l: dict | None, st_r: dict | None, goal: str) -> bool:
    g = _auto_goal_norm(goal)
    if g != AUTO_MODE_GOAL_FLAT:
        return False

    if _auto_safe_float(metrics.get("max_net_boost_db"), 0.0) > float(MAX_SAFE_BOOST):
        return True

    ratio_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    gd_keys = (
        "gd_grad_limiter_after_max_ms_per_oct",
        "gd_grad_limiter_before_max_ms_per_oct",
        "gd_limiter_max_grad_ms_per_oct",
        "gd_grad_limiter_max_grad_ms_per_oct",
        "gd_limiter_max_grad_after_ms_per_oct",
        "gd_grad_limiter_max_grad_after_ms_per_oct",
        "gd_limiter_max_grad_before_ms_per_oct",
        "gd_grad_limiter_max_grad_before_ms_per_oct",
    )
    for st in (dict(st_l or {}), dict(st_r or {})):
        pre_suspect = bool(st.get("pre_energy_metric_suspect", False))
        if not pre_suspect:
            ratio = _auto_pick_metric(st, ratio_keys, nonneg=True)
            if ratio is not None and float(ratio) > 0.05:
                return True
        gd_grad = _auto_pick_metric(st, gd_keys, abs_value=True, nonneg=True)
        if gd_grad is not None and float(gd_grad) > 45.0:
            return True

    return False


def _estimate_auto_mag_c_min_hz(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    default_hz: float = 25.0,
) -> float:
    def _sorted_xy(f, y):
        try:
            ff = np.asarray(f, dtype=float).reshape(-1)
            yy = np.asarray(y, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        if ff.size != yy.size or ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        idx = np.argsort(ff)
        ff = ff[idx]
        yy = yy[idx]
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        ff = ff[m]
        yy = yy[m]
        if ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return ff, yy

    def _f6(ff: np.ndarray, mm: np.ndarray) -> float | None:
        if ff.size < 32 or mm.size != ff.size:
            return None
        try:
            mm_sm, _ = dsp.apply_smoothing_std(
                ff,
                mm,
                np.zeros_like(mm),
                float(AUTO_MODE_MAG_C_MIN_SMOOTH_OCT),
            )
            mm_use = np.asarray(mm_sm, dtype=float)
        except Exception:
            mm_use = np.asarray(mm, dtype=float)

        ref_mask = (ff >= float(AUTO_MODE_MAG_C_MIN_REF_MIN_HZ)) & (ff <= float(AUTO_MODE_MAG_C_MIN_REF_MAX_HZ))
        if int(np.count_nonzero(ref_mask)) < 8:
            ref_mask = (ff >= 63.0) & (ff <= 250.0)
        if int(np.count_nonzero(ref_mask)) < 8:
            return None

        # Robust reference level: prefer upper-quantile to avoid SBIR/nulls pulling ref too low.
        ref_slice = np.asarray(mm_use[ref_mask], dtype=float)
        ref_slice = ref_slice[np.isfinite(ref_slice)]
        if ref_slice.size < 6:
            return None
        ref_db = float(np.quantile(ref_slice, 0.75))
        thr_db = float(ref_db - 6.0)

        lf_hi = float(min(float(AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ), float(AUTO_MODE_MAG_C_MIN_REF_MIN_HZ)))
        lf_mask = (ff >= float(AUTO_MODE_MAG_C_MIN_MIN_HZ)) & (ff <= lf_hi)
        if int(np.count_nonzero(lf_mask)) < 8:
            return None
        f_lo = ff[lf_mask]
        m_lo = mm_use[lf_mask]
        # Enforce monotonic LF envelope (rising with frequency) to reject local dips.
        m_env = np.maximum.accumulate(np.asarray(m_lo, dtype=float))
        above = np.asarray(m_env >= thr_db, dtype=bool)
        if not np.any(above):
            # If we never cross -6 dB, prefer a conservative fallback instead of pinning to lf_hi.
            return float(_auto_safe_float(default_hz, 25.0))

        # Require the envelope to stay above threshold for N consecutive points.
        # This reduces false early crossings caused by sparse LF bins or residual noise.
        try:
            df = float(np.median(np.diff(f_lo))) if f_lo.size > 2 else float("nan")
        except Exception:
            df = float("nan")
        # Target ~3 Hz minimum "stay above" span, but at least 3 points.
        if np.isfinite(df) and df > 0.0:
            N = int(max(3, round(3.0 / df)))
        else:
            N = 3
        N = int(np.clip(N, 3, 12))

        i1 = None
        if above.size >= N:
            # Find first index where above[i:i+N] are all True.
            for i in range(0, int(above.size) - N + 1):
                if bool(np.all(above[i : i + N])):
                    i1 = int(i)
                    break
        if i1 is None:
            # No stable crossing found -> conservative fallback.
            return float(_auto_safe_float(default_hz, 25.0))
        if i1 <= 0:
            return float(AUTO_MODE_MAG_C_MIN_MIN_HZ)

        x1, y1 = float(f_lo[i1 - 1]), float(m_env[i1 - 1])
        x2, y2 = float(f_lo[i1]), float(m_env[i1])
        if np.isfinite(y2 - y1) and abs(float(y2 - y1)) > 1e-9:
            f6 = float(x1 + (thr_db - y1) * (x2 - x1) / (y2 - y1))
        else:
            f6 = float(x2)
        return float(f6)

    fl, ml = _sorted_xy(f_l, m_l)
    fr, mr = _sorted_xy(f_r, m_r)
    f6_l = _f6(fl, ml)
    f6_r = _f6(fr, mr)
    if f6_l is None or not np.isfinite(f6_l):
        f6_l = None
    if f6_r is None or not np.isfinite(f6_r):
        f6_r = None

    if f6_l is None and f6_r is None:
        est = _auto_safe_float(default_hz, 25.0)
    elif f6_l is None:
        est = float(f6_r)
    elif f6_r is None:
        est = float(f6_l)
    else:
        # If channels agree reasonably well, average for stability; otherwise stay conservative.
        if abs(float(f6_l) - float(f6_r)) <= 8.0:
            est = 0.5 * (float(f6_l) + float(f6_r))
        else:
            est = max(float(f6_l), float(f6_r))

    est = float(np.clip(est, float(AUTO_MODE_MAG_C_MIN_MIN_HZ), float(AUTO_MODE_MAG_C_MIN_MAX_HZ)))
    return float(round(est, 1))


def _estimate_auto_hpf_from_response(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    default_freq_hz: float = 20.0,
    default_slope_db_oct: int = 24,
) -> dict:
    allowed_slopes = sorted(
        {
            int(v)
            for v in tuple(AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT)
            if int(v) > 0
        }
    )
    if not allowed_slopes:
        allowed_slopes = [24]
    min_hz = float(AUTO_MODE_HPF_MIN_HZ)
    max_hz = float(AUTO_MODE_HPF_MAX_HZ)
    ref_min = float(AUTO_MODE_HPF_REF_MIN_HZ)
    ref_max = float(AUTO_MODE_HPF_REF_MAX_HZ)
    search_max = float(AUTO_MODE_HPF_SEARCH_MAX_HZ)

    def _nearest_slope(v: float) -> int:
        try:
            x = float(v)
        except Exception:
            x = float(default_slope_db_oct)
        return int(min(allowed_slopes, key=lambda s: abs(float(s) - x)))

    def _sorted_xy(f, y):
        try:
            ff = np.asarray(f, dtype=float).reshape(-1)
            yy = np.asarray(y, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        if ff.size != yy.size or ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        idx = np.argsort(ff)
        ff = ff[idx]
        yy = yy[idx]
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        ff = ff[m]
        yy = yy[m]
        if ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return ff, yy

    def _stable_crossing_hz(fx: np.ndarray, yx: np.ndarray, thr_db: float) -> float | None:
        if fx.size < 8 or yx.size != fx.size:
            return None
        above = np.asarray(yx >= float(thr_db), dtype=bool)
        if not np.any(above):
            return None
        try:
            df = float(np.median(np.diff(fx))) if fx.size > 2 else float("nan")
        except Exception:
            df = float("nan")
        if np.isfinite(df) and df > 0.0:
            n_cons = int(max(3, round(3.0 / df)))
        else:
            n_cons = 3
        n_cons = int(np.clip(n_cons, 3, 12))

        i1 = None
        if above.size >= n_cons:
            for i in range(0, int(above.size) - n_cons + 1):
                if bool(np.all(above[i : i + n_cons])):
                    i1 = int(i)
                    break
        if i1 is None:
            return None
        if i1 <= 0:
            return float(fx[0])
        x1, y1 = float(fx[i1 - 1]), float(yx[i1 - 1])
        x2, y2 = float(fx[i1]), float(yx[i1])
        if np.isfinite(y2 - y1) and abs(float(y2 - y1)) > 1e-9:
            return float(x1 + (float(thr_db) - y1) * (x2 - x1) / (y2 - y1))
        return float(x2)

    def _butter_mag_db(freqs: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
        f = np.asarray(freqs, dtype=float)
        fc = float(cutoff_hz)
        n = int(order)
        with np.errstate(divide="ignore"):
            return -10.0 * np.log10(1.0 + np.power(fc / np.maximum(f, 1e-9), 2 * n))

    def _fit_one_channel(ff: np.ndarray, mm: np.ndarray, *, ch_name: str) -> dict | None:
        if ff.size < 32 or mm.size != ff.size:
            return None
        try:
            mm_sm, _ = dsp.apply_smoothing_std(
                ff,
                mm,
                np.zeros_like(mm),
                float(AUTO_MODE_HPF_SMOOTH_OCT),
            )
            mm_use = np.asarray(mm_sm, dtype=float)
        except Exception:
            mm_use = np.asarray(mm, dtype=float)

        ref_mask = (ff >= ref_min) & (ff <= ref_max)
        if int(np.count_nonzero(ref_mask)) < 8:
            ref_mask = (ff >= 80.0) & (ff <= 320.0)
        if int(np.count_nonzero(ref_mask)) < 8:
            return None
        ref_slice = np.asarray(mm_use[ref_mask], dtype=float)
        ref_slice = ref_slice[np.isfinite(ref_slice)]
        if ref_slice.size < 6:
            return None
        ref_db = float(np.quantile(ref_slice, 0.75))

        lf_hi = float(min(search_max, ref_min))
        lf_mask = (ff >= min_hz) & (ff <= lf_hi)
        if int(np.count_nonzero(lf_mask)) < 10:
            return None
        f_lo = np.asarray(ff[lf_mask], dtype=float)
        m_lo = np.asarray(mm_use[lf_mask], dtype=float)
        # Monotonic LF envelope rejects modal dips and sparse-bin spikes.
        m_env = np.maximum.accumulate(m_lo)

        thr3 = float(ref_db - 3.0)
        thr6 = float(ref_db - 6.0)
        thr12 = float(ref_db - 12.0)
        f3 = _stable_crossing_hz(f_lo, m_env, thr3)
        f6 = _stable_crossing_hz(f_lo, m_env, thr6)
        f12 = _stable_crossing_hz(f_lo, m_env, thr12)
        if f3 is None or f6 is None:
            return None
        if not (np.isfinite(f3) and np.isfinite(f6)):
            return None
        if float(f6) >= float(f3):
            return None

        c6 = float((10.0 ** 0.6) - 1.0)
        c12 = float((10.0 ** 1.2) - 1.0)
        n_estimates = []
        try:
            den6 = float(np.log(float(f3) / float(f6)))
            if np.isfinite(den6) and den6 > 1e-9:
                n_estimates.append(float(np.log(c6) / (2.0 * den6)))
        except Exception:
            pass
        if f12 is not None and np.isfinite(f12) and float(f12) < float(f3):
            try:
                den12 = float(np.log(float(f3) / float(f12)))
                if np.isfinite(den12) and den12 > 1e-9:
                    n_estimates.append(float(np.log(c12) / (2.0 * den12)))
            except Exception:
                pass
        if n_estimates:
            n_est = float(np.median(np.asarray(n_estimates, dtype=float)))
        else:
            fit_mask = (
                (f_lo >= min_hz)
                & (f_lo <= min(float(f3) * 0.95, float(np.max(f_lo))))
                & (m_env <= float(ref_db - 6.0))
            )
            if int(np.count_nonzero(fit_mask)) >= 8:
                x = np.log2(np.asarray(f_lo[fit_mask], dtype=float))
                y = np.asarray(m_env[fit_mask], dtype=float)
                try:
                    p = np.polyfit(x, y, 1)
                    slope_db_oct = abs(float(p[0]))
                    n_est = float(np.clip(slope_db_oct / 6.0, 1.0, 8.0))
                except Exception:
                    n_est = float(default_slope_db_oct) / 6.0
            else:
                n_est = float(default_slope_db_oct) / 6.0

        if not np.isfinite(n_est):
            n_est = float(default_slope_db_oct) / 6.0
        n_est = float(np.clip(n_est, 1.0, 8.0))
        slope_db_oct = int(_nearest_slope(6.0 * n_est))
        order = max(1, int(round(float(slope_db_oct) / 6.0)))
        # If the observed -12 dB crossing is near the analysis floor, steep orders are often
        # an artifact of truncated LF data. Keep slope conservative in that case.
        if (f12 is not None and np.isfinite(f12) and float(f12) <= (min_hz * 1.05)) or float(f6) <= (min_hz + 1.8):
            slope_db_oct = int(min(int(slope_db_oct), 24))
            order = max(1, int(round(float(slope_db_oct) / 6.0)))

        ratio6 = float(np.power(c6, 1.0 / (2.0 * float(order))))
        ratio12 = float(np.power(c12, 1.0 / (2.0 * float(order))))
        fc_candidates = []
        if np.isfinite(f3):
            fc_candidates.append(float(f3))
        if np.isfinite(f6):
            fc_candidates.append(float(f6) * ratio6)
        if f12 is not None and np.isfinite(f12):
            fc_candidates.append(float(f12) * ratio12)
        if not fc_candidates:
            return None
        fc_hz = float(np.median(np.asarray(fc_candidates, dtype=float)))
        fc_hz = float(np.clip(fc_hz, min_hz, max_hz))

        fit_mask = (f_lo >= max(min_hz, fc_hz / 4.0)) & (f_lo <= min(search_max, fc_hz * 2.8))
        if int(np.count_nonzero(fit_mask)) < 12:
            fit_mask = (f_lo >= min_hz) & (f_lo <= search_max)
        model = float(ref_db) + _butter_mag_db(f_lo, fc_hz, order)
        if int(np.count_nonzero(fit_mask)) >= 4:
            fit_meas = np.asarray(m_env[fit_mask], dtype=float)
            fit_mod = np.asarray(model[fit_mask], dtype=float)
            # Robust offset compensation: LF room gain can shift absolute level while shape is valid.
            delta_db = float(np.median(fit_meas - fit_mod))
            fit_mod = fit_mod + delta_db
            err = np.asarray(fit_mod - fit_meas, dtype=float)
            # Robust RMSE: trim tails to avoid a single deep mode/null dominating confidence.
            if err.size >= 8:
                q_lo, q_hi = np.quantile(err, [0.10, 0.90])
                keep = (err >= float(q_lo)) & (err <= float(q_hi))
                err_use = err[keep] if int(np.count_nonzero(keep)) >= 4 else err
            else:
                err_use = err
            fit_rmse_db = float(np.sqrt(np.mean(np.square(err_use))))
        else:
            fit_rmse_db = float("nan")

        conf = 1.0
        if np.isfinite(fit_rmse_db):
            conf *= float(np.clip(1.0 - (fit_rmse_db / 10.0), 0.0, 1.0))
        else:
            conf *= 0.35
        if not (f12 is not None and np.isfinite(f12)):
            conf *= 0.88
        if float(f3) <= (min_hz + 0.5):
            conf *= 0.82
        # Coherent crossing geometry around selected order boosts confidence.
        try:
            expected_ratio = float(np.power(c6, 1.0 / (2.0 * float(order))))
            obs_ratio = float(max(1e-9, float(f3)) / max(1e-9, float(f6)))
            ratio_err = abs(np.log2(max(1e-9, obs_ratio / expected_ratio)))
            conf *= float(np.clip(1.0 - ratio_err / 1.5, 0.45, 1.0))
        except Exception:
            pass
        conf = float(np.clip(conf, 0.0, 1.0))

        return {
            "channel": str(ch_name),
            "freq": float(fc_hz),
            "slope_db_oct": int(slope_db_oct),
            "order": int(order),
            "f3_hz": float(f3),
            "f6_hz": float(f6),
            "f12_hz": float(f12) if (f12 is not None and np.isfinite(f12)) else float("nan"),
            "fit_rmse_db": float(fit_rmse_db),
            "confidence": float(conf),
        }

    fl, ml = _sorted_xy(f_l, m_l)
    fr, mr = _sorted_xy(f_r, m_r)
    ch_l = _fit_one_channel(fl, ml, ch_name="L")
    ch_r = _fit_one_channel(fr, mr, ch_name="R")
    valid = [c for c in (ch_l, ch_r) if isinstance(c, dict)]

    default_freq = float(np.clip(_auto_safe_float(default_freq_hz, 20.0), min_hz, max_hz))
    default_slope = int(_nearest_slope(float(default_slope_db_oct)))
    if not valid:
        return {
            "enabled": False,
            "freq": float(round(default_freq, 1)),
            "slope_db_oct": int(default_slope),
            "order": int(max(1, round(float(default_slope) / 6.0))),
            "confidence": 0.0,
            "fit_rmse_db": float("nan"),
            "method": "fallback_default",
            "channels": [],
        }

    if len(valid) == 1:
        v = dict(valid[0])
        freq_hz = float(v.get("freq", default_freq))
        slope_db = int(_nearest_slope(float(v.get("slope_db_oct", default_slope))))
        conf = float(np.clip(_auto_safe_float(v.get("confidence", 0.0), 0.0), 0.0, 1.0))
        fit_rmse_db = _auto_safe_float(v.get("fit_rmse_db", float("nan")), float("nan"))
    else:
        f1 = float(valid[0].get("freq", default_freq))
        f2 = float(valid[1].get("freq", default_freq))
        c1 = float(np.clip(_auto_safe_float(valid[0].get("confidence", 0.0), 0.0), 0.05, 1.0))
        c2 = float(np.clip(_auto_safe_float(valid[1].get("confidence", 0.0), 0.0), 0.05, 1.0))
        cw = c1 + c2
        freq_hz = float((f1 * c1 + f2 * c2) / cw) if cw > 0.0 else float(0.5 * (f1 + f2))
        denom = max(1e-9, float(0.5 * (f1 + f2)))
        freq_delta_rel = abs(f1 - f2) / denom
        if np.isfinite(freq_delta_rel) and float(freq_delta_rel) > 0.28:
            # Channel disagreement: choose safer (higher) corner and lower confidence.
            freq_hz = float(max(f1, f2))
            conf_agree = 0.78
        else:
            conf_agree = 1.0

        s1 = float(valid[0].get("slope_db_oct", default_slope))
        s2 = float(valid[1].get("slope_db_oct", default_slope))
        slope_w = float((s1 * c1 + s2 * c2) / cw) if cw > 0.0 else float(0.5 * (s1 + s2))
        slope_db = int(_nearest_slope(slope_w))

        rmse_vals = [
            _auto_safe_float(valid[0].get("fit_rmse_db", float("nan")), float("nan")),
            _auto_safe_float(valid[1].get("fit_rmse_db", float("nan")), float("nan")),
        ]
        rmse_ok = [float(x) for x in rmse_vals if np.isfinite(x)]
        fit_rmse_db = float(np.mean(rmse_ok)) if rmse_ok else float("nan")
        conf = float(np.clip(0.5 * (c1 + c2) * float(conf_agree), 0.0, 1.0))

    freq_hz = float(np.clip(freq_hz, min_hz, max_hz))
    auto_enable = bool(
        np.isfinite(freq_hz)
        and (conf >= float(AUTO_MODE_HPF_AUTO_ENABLE_MIN_CONF))
        and (min_hz <= freq_hz <= max_hz)
    )

    return {
        "enabled": bool(auto_enable),
        "freq": float(round(freq_hz, 1)),
        "slope_db_oct": int(slope_db),
        "order": int(max(1, round(float(slope_db) / 6.0))),
        "confidence": float(round(conf, 3)),
        "fit_rmse_db": float(fit_rmse_db) if np.isfinite(fit_rmse_db) else float("nan"),
        "method": "response_fit",
        "channels": valid,
    }


def _build_auto_mode_candidates(
    base_data: dict,
    *,
    n_trials: int,
    seed: int,
    optimize_mag_low: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_eff = max(1, int(n_trials))
    goal = _auto_goal(base_data)
    tune_mag_low = bool(optimize_mag_low)

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    mixed_center = _auto_safe_float(base_data.get("mixed_freq", 180.0), 180.0)
    if not np.isfinite(mixed_center) or mixed_center <= 0.0:
        mixed_center = 180.0
    phase_center = _auto_phase_limit_center(base_data.get("phase_limit", None))
    mag_c_min_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low_bass_cut_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )

    out_seed = {}
    if _auto_goal_norm(goal) == AUTO_MODE_GOAL_LOW_RIPPLE and bool(keep_tdc):
        out_seed["tdc_strength"] = round(
            float(max(_auto_safe_float(base_data.get("tdc_strength", 55.0), 55.0), 55.0)),
            1,
        )
    if bool(is_phase_search):
        out_seed["phase_limit"] = round(float(phase_center), 1)
    out: list[dict] = [out_seed]
    tdc_min = 55.0 if (_auto_goal_norm(goal) == AUTO_MODE_GOAL_LOW_RIPPLE and bool(keep_tdc)) else 15.0
    for _ in range(max(0, n_eff - 1)):
        if bool(tune_mag_low):
            mag_c_min_cand, low_bass_cut_cand = _auto_sample_mag_low_pair(
                rng,
                mag_center=float(mag_c_min_seed),
                low_center=float(low_bass_cut_seed),
                mag_sigma=2.6,
                low_sigma=3.2,
            )
        else:
            mag_c_min_cand = float(round(mag_c_min_seed, 1))
            low_bass_cut_cand = float(round(low_bass_cut_seed, 1))
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(rng.uniform(5.0, 16.0)), 2),
            "tdc_strength": round(float(rng.uniform(float(tdc_min), 75.0)), 1),
            "tdc_max_reduction_db": round(float(rng.uniform(6.0, 36.0)), 1),
            "tdc_slope_db_per_oct": float(rng.choice(np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]))),
            "reg_strength": round(float(rng.uniform(15.0, 45.0)), 1),
            "max_slope_db_per_oct": float(rng.choice(np.array([8.0, 10.0, 12.0, 14.0, 16.0]))),
            "max_boost": round(float(rng.uniform(3.0, 8.0)), 2),
            "mag_c_min": float(mag_c_min_cand),
            "mag_c_max": round(float(rng.uniform(170.0, 300.0)), 1),
            "trans_width": round(float(rng.uniform(70.0, 150.0)), 1),
            "filter_smooth": int(rng.choice(np.array([12, 24, 48, 96]))),
            "bass_first_mode_max_hz": round(float(rng.uniform(150.0, 220.0)), 1),
            "low_bass_cut_hz": float(low_bass_cut_cand),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(loc=mixed_center, scale=35.0), 80.0, 320.0)), 1)
        if is_phase_search:
            phase_lo = float(AUTO_MODE_PHASE_LIMIT_MIN_HZ)
            phase_hi = float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)
            phase_global_frac = float(np.clip(_auto_safe_float(AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_FRAC, 0.35), 0.0, 1.0))
            phase_uniform_frac = float(np.clip(_auto_safe_float(AUTO_MODE_PHASE_LIMIT_EXPLORE_UNIFORM_FRAC, 0.20), 0.0, 1.0))
            # Keep mixture probabilities sane.
            phase_uniform_frac = min(phase_uniform_frac, 1.0 - 1e-6)
            phase_global_frac = min(phase_global_frac, max(0.0, 1.0 - phase_uniform_frac - 1e-6))
            phase_u = float(rng.random())
            if phase_u < phase_uniform_frac:
                phase_draw = float(rng.uniform(phase_lo, phase_hi))
            elif phase_u < (phase_uniform_frac + phase_global_frac):
                phase_draw = float(
                    rng.normal(
                        loc=float(AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ),
                        scale=float(AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ),
                    )
                )
            else:
                phase_draw = float(
                    rng.normal(
                        loc=float(phase_center),
                        scale=float(AUTO_MODE_PHASE_LIMIT_SIGMA_HZ),
                    )
                )
            cand["phase_limit"] = round(
                float(_clip(phase_draw, phase_lo, phase_hi)),
                1,
            )
        out.append(cand)
    return out


def _build_auto_mode_candidates_optuna(
    base_data: dict,
    *,
    n_trials: int,
    seed: int,
    startup_trials: int = AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
    optimize_mag_low: bool = True,
) -> list[dict] | None:
    try:
        import optuna  # type: ignore
    except Exception:
        return None

    n_eff = max(1, int(n_trials))
    startup = int(max(1, min(int(startup_trials), int(n_eff))))
    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=int(startup))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    goal = _auto_goal(base_data)
    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    tune_mag_low = bool(optimize_mag_low)

    mag_c_min_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low_bass_cut_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )

    out_seed = {}
    if _auto_goal_norm(goal) == AUTO_MODE_GOAL_LOW_RIPPLE and bool(keep_tdc):
        out_seed["tdc_strength"] = round(
            float(max(_auto_safe_float(base_data.get("tdc_strength", 55.0), 55.0), 55.0)),
            1,
        )
    if bool(is_phase_search):
        out_seed["phase_limit"] = round(float(_auto_phase_limit_center(base_data.get("phase_limit", None))), 1)

    out: list[dict] = [dict(out_seed)]
    tdc_min = 55.0 if (_auto_goal_norm(goal) == AUTO_MODE_GOAL_LOW_RIPPLE and bool(keep_tdc)) else 15.0

    for _ in range(max(0, n_eff - 1)):
        tr = study.ask()
        if bool(tune_mag_low):
            mag_c_min = float(
                tr.suggest_float(
                    "mag_c_min",
                    float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                    float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                )
            )
            low_delta = float(tr.suggest_float("low_bass_delta_hz", -8.0, 10.0))
            low_bass_cut_hz = float(
                np.clip(
                    float(mag_c_min) + float(low_delta),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                )
            )
        else:
            mag_c_min = float(round(mag_c_min_seed, 1))
            low_bass_cut_hz = float(round(low_bass_cut_seed, 1))

        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(tr.suggest_float("fdw_cycles", 5.0, 16.0)), 2),
            "tdc_strength": round(float(tr.suggest_float("tdc_strength", float(tdc_min), 75.0)), 1),
            "tdc_max_reduction_db": round(float(tr.suggest_float("tdc_max_reduction_db", 6.0, 36.0)), 1),
            "tdc_slope_db_per_oct": float(tr.suggest_categorical("tdc_slope_db_per_oct", [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0])),
            "reg_strength": round(float(tr.suggest_float("reg_strength", 15.0, 45.0)), 1),
            "max_slope_db_per_oct": float(tr.suggest_categorical("max_slope_db_per_oct", [8.0, 10.0, 12.0, 14.0, 16.0])),
            "max_boost": round(float(tr.suggest_float("max_boost", 3.0, 8.0)), 2),
            "mag_c_min": float(mag_c_min),
            "mag_c_max": round(float(tr.suggest_float("mag_c_max", 170.0, 300.0)), 1),
            "trans_width": round(float(tr.suggest_float("trans_width", 70.0, 150.0)), 1),
            "filter_smooth": int(tr.suggest_categorical("filter_smooth", [12, 24, 48, 96])),
            "bass_first_mode_max_hz": round(float(tr.suggest_float("bass_first_mode_max_hz", 150.0, 220.0)), 1),
            "low_bass_cut_hz": float(low_bass_cut_hz),
        }
        if bool(is_mixed):
            cand["mixed_freq"] = round(float(tr.suggest_float("mixed_freq", 80.0, 320.0)), 1)
        if bool(is_phase_search):
            cand["phase_limit"] = round(
                float(
                    tr.suggest_float(
                        "phase_limit",
                        float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                        float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                    )
                ),
                1,
            )

        # Pilot mode: lightweight surrogate so TPE has feedback between asks.
        s = 0.0
        s -= abs(float(cand.get("max_boost", 4.0)) - 4.5)
        s -= abs(float(cand.get("trans_width", 100.0)) - 100.0) / 40.0
        s -= abs(float(cand.get("reg_strength", 30.0)) - 30.0) / 20.0
        if bool(is_mixed):
            s -= abs(float(cand.get("mixed_freq", 180.0)) - 180.0) / 120.0
        if bool(is_phase_search):
            s -= abs(float(cand.get("phase_limit", _auto_phase_limit_center(base_data.get("phase_limit", None)))) - _auto_phase_limit_center(base_data.get("phase_limit", None))) / 120.0
        study.tell(tr, float(s))

        out.append(cand)

    return out


def _build_auto_mode_refine_candidates(
    base_data: dict,
    *,
    anchors: list[dict],
    n_trials: int,
    seed: int,
    optimize_mag_low: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_eff = max(0, int(n_trials))
    if n_eff <= 0:
        return []
    tune_mag_low = bool(optimize_mag_low)

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)

    anchor_items = list(anchors or [])
    if not anchor_items:
        anchor_items = [{"preset": {}}]

    def _anchor_val(anchor: dict, key: str, default: float) -> float:
        p = dict(anchor.get("preset", {}) or {})
        if key in p:
            return _auto_safe_float(p.get(key), default)
        return _auto_safe_float(base_data.get(key), default)

    def _near_discrete(center: float, choices: list[float], sigma: float) -> float:
        if not choices:
            return float(center)
        x = float(rng.normal(loc=float(center), scale=float(max(0.01, sigma))))
        return float(min(choices, key=lambda c: abs(float(c) - x)))

    out: list[dict] = []
    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0]
    max_slope_choices = [8.0, 10.0, 12.0, 14.0, 16.0]
    smooth_choices = [96]
    mag_c_min_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low_bass_cut_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )

    for _ in range(n_eff):
        a = anchor_items[int(rng.integers(0, len(anchor_items)))]
        if bool(tune_mag_low):
            mag_c_min_cand, low_bass_cut_cand = _auto_sample_mag_low_pair(
                rng,
                mag_center=_anchor_val(a, "mag_c_min", mag_c_min_seed),
                low_center=_anchor_val(a, "low_bass_cut_hz", low_bass_cut_seed),
                mag_sigma=1.8,
                low_sigma=2.4,
            )
        else:
            mag_c_min_cand = round(
                _clip(
                    _anchor_val(a, "mag_c_min", mag_c_min_seed),
                    float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                    float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                ),
                1,
            )
            low_bass_cut_cand = round(
                _clip(
                    _anchor_val(a, "low_bass_cut_hz", low_bass_cut_seed),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                ),
                1,
            )
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(np.clip(rng.normal(_anchor_val(a, "fdw_cycles", 10.0), 1.2), 8.0, 16.0)), 2),
            "tdc_strength": round(float(np.clip(rng.normal(_anchor_val(a, "tdc_strength", 50.0), 5.0), 35.0, 75.0)), 1),
            "tdc_max_reduction_db": round(float(np.clip(rng.normal(_anchor_val(a, "tdc_max_reduction_db", 9.0), 1.0), 6.0, 12.0)), 1),
            "tdc_slope_db_per_oct": _near_discrete(_anchor_val(a, "tdc_slope_db_per_oct", 6.0), slope_choices, 0.8),
            "reg_strength": round(float(np.clip(rng.normal(_anchor_val(a, "reg_strength", 30.0), 4.0), 15.0, 45.0)), 1),
            "max_slope_db_per_oct": _near_discrete(_anchor_val(a, "max_slope_db_per_oct", 12.0), max_slope_choices, 1.5),
            "max_boost": round(float(np.clip(rng.normal(_anchor_val(a, "max_boost", 4.0), 0.45), 3.0, 8.0)), 2),
            "mag_c_min": float(mag_c_min_cand),
            "mag_c_max": round(float(np.clip(rng.normal(_anchor_val(a, "mag_c_max", 220.0), 15.0), 170.0, 300.0)), 1),
            "trans_width": round(float(np.clip(rng.normal(_anchor_val(a, "trans_width", 100.0), 10.0), 70.0, 150.0)), 1),
            "filter_smooth": int(_near_discrete(_anchor_val(a, "filter_smooth", 96.0), [float(x) for x in smooth_choices], 96.0)),
            "bass_first_mode_max_hz": round(float(np.clip(rng.normal(_anchor_val(a, "bass_first_mode_max_hz", 180.0), 10.0), 150.0, 220.0)), 1),
            "low_bass_cut_hz": float(low_bass_cut_cand),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(_anchor_val(a, "mixed_freq", 180.0), 12.0), 80.0, 320.0)), 1)
        if is_phase_search:
            phase_anchor = _auto_phase_limit_center(_anchor_val(a, "phase_limit", AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ))
            cand["phase_limit"] = round(
                float(
                    np.clip(
                        rng.normal(
                            phase_anchor,
                            float(AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ),
                        ),
                        float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                        float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                    )
                ),
                1,
            )
        out.append(cand)
    return out


def _build_auto_mode_candidates_local(
    base_data: dict,
    center: dict,
    n_trials: int,
    seed: int,
    shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK,
    optimize_mag_low: bool = True,
) -> list[dict]:
    n_eff = max(1, int(n_trials))
    rng = np.random.default_rng(int(seed))
    s = float(np.clip(_auto_safe_float(shrink, AUTO_MODE_LOCAL_REFINE_SHRINK), 0.05, 1.50))
    tune_mag_low = bool(optimize_mag_low)

    base = dict(base_data or {})
    c = dict(base)
    c.update(dict(center or {}))

    ft = str(c.get("filter_type", base.get("filter_type", "")) or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    phase_center = _auto_phase_limit_center(c.get("phase_limit", base.get("phase_limit", None)))

    keep_tdc = bool(c.get("enable_tdc", True))
    keep_afdw = bool(c.get("enable_afdw", True))
    keep_bass_first = bool(c.get("bass_first_ai", True))
    mag_c_min_center = round(
        _clip(
            c.get("mag_c_min", base.get("mag_c_min", 25.0)),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        ),
        1,
    )
    low_bass_cut_center = round(
        _clip(
            c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", 40.0)),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        ),
        1,
    )

    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]
    slope_center = _auto_safe_float(c.get("tdc_slope_db_per_oct", base.get("tdc_slope_db_per_oct", 6.0)), 6.0)
    slope_idx = int(min(range(len(slope_choices)), key=lambda i: abs(float(slope_choices[i]) - float(slope_center))))

    center_out = dict(c)
    center_out["comparison_mode"] = True
    center_out["enable_tdc"] = bool(keep_tdc)
    center_out["enable_afdw"] = bool(keep_afdw)
    center_out["bass_first_ai"] = bool(keep_bass_first)
    center_out["mag_c_min"] = float(mag_c_min_center)
    center_out["low_bass_cut_hz"] = float(low_bass_cut_center)
    if bool(is_phase_search):
        center_out["phase_limit"] = round(float(phase_center), 1)

    out: list[dict] = [center_out]
    for _ in range(max(0, n_eff - 1)):
        step = int(rng.choice(np.array([-1, 0, 1], dtype=int), p=np.array([0.20, 0.60, 0.20])))
        idx = int(np.clip(int(slope_idx + step), 0, len(slope_choices) - 1))
        if bool(tune_mag_low):
            mag_c_min_cand, low_bass_cut_cand = _auto_sample_mag_low_pair(
                rng,
                mag_center=_auto_safe_float(c.get("mag_c_min", base.get("mag_c_min", mag_c_min_center)), mag_c_min_center),
                low_center=_auto_safe_float(c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", low_bass_cut_center)), low_bass_cut_center),
                mag_sigma=max(0.4, 3.2 * s),
                low_sigma=max(0.6, 4.0 * s),
            )
        else:
            mag_c_min_cand = float(mag_c_min_center)
            low_bass_cut_cand = float(low_bass_cut_center)
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(_jitter(rng, c.get("fdw_cycles", None), 2.5 * s, 8.0, 16.0, base_data=base, key="fdw_cycles", default=10.0), 2),
            "tdc_strength": round(_jitter(rng, c.get("tdc_strength", None), 12.0 * s, 15.0, 75.0, base_data=base, key="tdc_strength", default=50.0), 1),
            "tdc_max_reduction_db": round(_jitter(rng, c.get("tdc_max_reduction_db", None), 6.0 * s, 6.0, 36.0, base_data=base, key="tdc_max_reduction_db", default=9.0), 1),
            "tdc_slope_db_per_oct": float(slope_choices[idx]),
            "reg_strength": round(_jitter(rng, c.get("reg_strength", None), 10.0 * s, 15.0, 45.0, base_data=base, key="reg_strength", default=30.0), 1),
            "max_boost": round(_jitter(rng, c.get("max_boost", None), 1.0 * s, 3.0, 8.0, base_data=base, key="max_boost", default=4.0), 2),
            "mag_c_min": float(mag_c_min_cand),
            "mag_c_max": round(_jitter(rng, c.get("mag_c_max", None), 25.0 * s, 170.0, 300.0, base_data=base, key="mag_c_max", default=220.0), 1),
            "trans_width": round(_jitter(rng, c.get("trans_width", None), 25.0 * s, 70.0, 150.0, base_data=base, key="trans_width", default=100.0), 1),
            "bass_first_mode_max_hz": round(_jitter(rng, c.get("bass_first_mode_max_hz", None), 25.0 * s, 150.0, 220.0, base_data=base, key="bass_first_mode_max_hz", default=180.0), 1),
            "low_bass_cut_hz": float(low_bass_cut_cand),
        }
        if is_mixed:
            cand["mixed_freq"] = round(_jitter(rng, c.get("mixed_freq", None), 35.0 * s, 80.0, 320.0, base_data=base, key="mixed_freq", default=180.0), 1)
        if is_phase_search:
            cand["phase_limit"] = round(
                _jitter(
                    rng,
                    c.get("phase_limit", None),
                    float(AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ) * s,
                    float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                    float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                    base_data=base,
                    key="phase_limit",
                    default=float(phase_center),
                ),
                1,
            )
        out.append(cand)
    return out


def _build_auto_mode_candidates_micro(
    base_data: dict,
    center: dict,
    *,
    n_trials: int = AUTO_MODE_PHASE3_MICRO_TRIALS,
    shrink: float = 1.0,
) -> list[dict]:
    n_eff = max(1, int(n_trials))
    p = dict(base_data or {})
    p.update(dict(center or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)

    # shrink=1.0 keeps original steps; smaller tightens local search.
    s = float(np.clip(_auto_safe_float(shrink, 1.0), 0.25, 1.0))
    mixed_steps = [0.0, -16.0 * s, -8.0 * s, +8.0 * s, +16.0 * s]
    phase_steps = [0.0, -28.0 * s, -14.0 * s, +14.0 * s, +28.0 * s]
    tdc_steps = [0.0, -8.0 * s, -4.0 * s, +4.0 * s, +8.0 * s]
    fdw_steps = [0.0, -1.0 * s, +1.0 * s]
    reg_steps = [0.0, -6.0 * s, +6.0 * s]
    tw_steps = [0.0, -15.0 * s, +15.0 * s]
    # Deterministic pattern list: repeatable and intentionally diverse.
    patterns = [
        (0, 0, 0, 0, 0),
        (2, 2, 1, 1, 1),
        (3, 3, 2, 2, 2),
        (1, 1, 2, 1, 2),
        (4, 4, 1, 2, 1),
        (2, 1, 2, 2, 0),
        (3, 4, 1, 0, 2),
        (1, 3, 0, 2, 0),
        (4, 2, 2, 0, 1),
        (0, 4, 1, 2, 2),
        (0, 1, 2, 0, 1),
        (2, 0, 0, 1, 2),
    ]

    base_mixed = _auto_safe_float(p.get("mixed_freq", 180.0), 180.0)
    base_phase = _auto_phase_limit_center(p.get("phase_limit", None))
    base_tdc = _auto_safe_float(p.get("tdc_strength", 55.0), 55.0)
    base_fdw = _auto_safe_float(p.get("fdw_cycles", 10.0), 10.0)
    base_reg = _auto_safe_float(p.get("reg_strength", 30.0), 30.0)
    base_tw = _auto_safe_float(p.get("trans_width", 100.0), 100.0)

    out: list[dict] = []
    seen = set()
    for i in range(max(1, n_eff)):
        pi = patterns[int(i % len(patterns))]
        cand = dict(center or {})
        cand["comparison_mode"] = True
        cand["tdc_strength"] = round(_clip(base_tdc + float(tdc_steps[int(pi[1])]), 35.0, 80.0), 1)
        cand["fdw_cycles"] = round(_clip(base_fdw + float(fdw_steps[int(pi[2])]), 6.0, 16.0), 2)
        cand["reg_strength"] = round(_clip(base_reg + float(reg_steps[int(pi[3])]), 15.0, 45.0), 1)
        cand["trans_width"] = round(_clip(base_tw + float(tw_steps[int(pi[4])]), 70.0, 150.0), 1)
        if bool(is_mixed):
            cand["mixed_freq"] = round(_clip(base_mixed + float(mixed_steps[int(pi[0])]), 80.0, 320.0), 1)
        if bool(is_phase_search):
            cand["phase_limit"] = round(
                _clip(
                    base_phase + float(phase_steps[int(pi[0])]),
                    float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                    float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                ),
                1,
            )

        sig = (
            float(_auto_safe_float(cand.get("mixed_freq", float("nan")), float("nan"))) if bool(is_mixed) else float("nan"),
            float(_auto_safe_float(cand.get("phase_limit", float("nan")), float("nan"))) if bool(is_phase_search) else float("nan"),
            float(_auto_safe_float(cand.get("tdc_strength", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("fdw_cycles", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("reg_strength", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("trans_width", float("nan")), float("nan"))),
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= n_eff:
            break

    if not out:
        base_c = dict(center or {})
        base_c["comparison_mode"] = True
        out = [base_c]
    return out


@dataclass
class _AutoModeSearchState:
    best_result: object | None = None
    best_metrics: dict | None = None
    best_preset: dict | None = None
    scored: list[dict] = field(default_factory=list)
    phase2_pool: list[dict] = field(default_factory=list)


@dataclass
class _AutoModePhaseState:
    ok_n: int = 0
    tried_n: int = 0
    plateau_hit: bool = False
    no_improve_streak: int = 0
    improved_any: bool = False


def _run_auto_mode_search_impl(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int = AUTO_MODE_TRIALS,
) -> dict | None:
    search_base_data = dict(base_data or {})
    cfg = AutoModeConfig.from_base_data(search_base_data)
    n_trials_eff = int(max(1, _auto_safe_int(n_trials, cfg.trials)))
    program_version = _auto_program_version(search_base_data)
    goal = _auto_goal(search_base_data)
    filter_key = _auto_filter_cache_key(search_base_data)
    rank_basis = _auto_goal_basis_text(goal)
    optimizer_backend = _auto_optimizer_backend(
        search_base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled),
    )
    try:
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
    except Exception:
        seed_preset = {}
    if seed_preset:
        search_base_data.update(seed_preset)

    # --- Auto-mode cache: load best preset for this measurement+settings signature ---
    if bool(cfg.cache_enabled) and not seed_preset:
        try:
            sig = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(search_base_data.get("hc_mode", "") or "").strip() or None,
                include_hc_mode=True,
            )
            cached = _auto_cache_get_best(
                sig,
                filter_key=filter_key,
                program_version=program_version,
            )
            cache_seed_source = "signature"
            if not (isinstance(cached, dict) and cached):
                cache_seed_source = "last_used"
                cached_entry = _auto_cache_get_last_used_best(
                    goal=goal,
                    filter_key=filter_key,
                    program_version=program_version,
                )
                cached = dict((cached_entry or {}).get("best_preset", {}) or {})
            if isinstance(cached, dict) and cached:
                # Use as seed preset (your code already merges _auto_target_seed_preset at top)
                search_base_data["_auto_target_seed_preset"] = dict(cached)
                # Also apply immediately so phase-1 includes this "known good" point
                search_base_data.update(dict(cached))
                if str(cache_seed_source) == "last_used":
                    logger.info(
                        "Automatic mode: loaded filter-specific last-used preset seed."
                    )
                else:
                    logger.info("Automatic mode: loaded cached best preset seed.")
        except Exception:
            pass

    if str(filter_key) in ("linear", "asym"):
        prev_phase_limit = _auto_safe_float(search_base_data.get("phase_limit", float("nan")), float("nan"))
        clamped_phase_limit = round(
            float(_auto_phase_limit_center(search_base_data.get("phase_limit", None))),
            1,
        )
        search_base_data["phase_limit"] = float(clamped_phase_limit)
        if np.isfinite(prev_phase_limit) and abs(float(prev_phase_limit) - float(clamped_phase_limit)) > 1e-9:
            logger.info(
                "Automatic mode: clamped phase_limit seed "
                f"{float(prev_phase_limit):.1f} -> {float(clamped_phase_limit):.1f} Hz "
                f"for {str(filter_key)} filter"
            )

    seed = int(20260302 + int(fs_v) * 17 + int(taps_v))
    candidates = _build_auto_mode_candidates(search_base_data, n_trials=int(n_trials_eff), seed=seed)
    if str(optimizer_backend) == "optuna":
        if int(n_trials_eff) >= int(cfg.optuna_pilot_min_trials):
            optuna_candidates = _build_auto_mode_candidates_optuna(
                search_base_data,
                n_trials=int(n_trials_eff),
                seed=int(seed),
                startup_trials=int(cfg.optuna_pilot_startup_trials),
            )
            if isinstance(optuna_candidates, list) and optuna_candidates:
                candidates = list(optuna_candidates)
                logger.info(
                    "Automatic mode phase1 sampler: optuna "
                    f"({int(len(candidates))} candidates)"
                )
            else:
                logger.warning(
                    "Automatic mode optuna sampler requested but unavailable; "
                    "falling back to builtin candidate sampler."
                )
        else:
            logger.info(
                "Automatic mode optuna pilot skipped: "
                f"trials={int(n_trials_eff)} < min={int(cfg.optuna_pilot_min_trials)}; "
                "using builtin sampler."
            )
    try:
        target_label = str(search_base_data.get("hc_mode", "") or "").strip()
    except Exception:
        target_label = ""
    if not target_label:
        target_label = "n/a"
    f6_hz = _auto_safe_float(
        search_base_data.get("_auto_mag_c_min_hz", search_base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    low_bass_hz = _auto_safe_float(
        search_base_data.get("_auto_low_bass_cut_hz", search_base_data.get("low_bass_cut_hz", float("nan"))),
        float("nan"),
    )
    exc_hz = _auto_safe_float(
        search_base_data.get("_auto_exc_freq_hz", search_base_data.get("exc_freq", float("nan"))),
        float("nan"),
    )
    hpf_enabled = bool(search_base_data.get("hpf_enable", False))
    hpf_freq = _auto_safe_float(search_base_data.get("hpf_freq", float("nan")), float("nan"))
    hpf_slope = _auto_safe_float(search_base_data.get("hpf_slope", float("nan")), float("nan"))
    hpf_meta = dict(search_base_data.get("_auto_hpf_meta", {}) or {})
    if isinstance(hpf_meta, dict):
        hpf_meta_enabled = bool(hpf_meta.get("applied", hpf_meta.get("enabled", False)))
        if not hpf_enabled:
            hpf_enabled = bool(hpf_meta_enabled)
        if not np.isfinite(hpf_freq):
            hpf_freq = _auto_safe_float(hpf_meta.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_slope = _auto_safe_float(hpf_meta.get("slope_db_oct", float("nan")), float("nan"))
    if isinstance(hpf, dict):
        if not hpf_enabled:
            hpf_enabled = bool(hpf.get("enabled", False))
        if not np.isfinite(hpf_freq):
            hpf_freq = _auto_safe_float(hpf.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_order = _auto_safe_float(hpf.get("order", float("nan")), float("nan"))
            if np.isfinite(hpf_order) and float(hpf_order) > 0.0:
                hpf_slope = float(6.0 * float(hpf_order))

    low_txt = f"low-cut {low_bass_hz:.1f} Hz" if np.isfinite(low_bass_hz) else "low-cut n/a"
    exc_txt = f"exc seed {exc_hz:.1f} Hz" if np.isfinite(exc_hz) else "exc seed n/a"
    if bool(hpf_enabled) and np.isfinite(hpf_freq):
        if np.isfinite(hpf_slope):
            hpf_txt = f"hpf {hpf_freq:.1f} Hz/{int(round(hpf_slope))} dB/oct"
        else:
            hpf_txt = f"hpf {hpf_freq:.1f} Hz"
    else:
        hpf_txt = "hpf off"

    if np.isfinite(f6_hz):
        status_prefix = (
            f"CamillaFIR automatic mode [{target_label}] "
            f"(-6 dB {f6_hz:.1f} Hz, {low_txt}, {exc_txt}, {hpf_txt})"
        )
    else:
        status_prefix = f"CamillaFIR automatic mode [{target_label}] ({low_txt}, {exc_txt}, {hpf_txt})"
    refine_trial_hint = int(cfg.refine_trial_hint(goal))
    logger.info(
        "Automatic mode search: "
        f"goal={goal}, basis={rank_basis}, target={target_label}, "
        f"trials={int(n_trials_eff)}+{int(refine_trial_hint)}"
    )

    search_state = _AutoModeSearchState()
    phase1_ok = 0
    phase2_ok = 0
    phase1_tried = 0
    phase2_tried = 0
    phase1_plateau_hit = False
    phase2_plateau_hit = False

    def _eval_candidates(
        cands: list[dict],
        *,
        phase_label: str,
        plateau_after_no_improve: int = 0,
        use_refine_tiebreak: bool = False,
        focus_lo_hz: float | None = None,
        focus_hi_hz: float | None = None,
    ) -> dict:
        phase_state = _AutoModePhaseState()
        n_total = int(len(cands))
        workers = int(_auto_trial_workers(search_base_data, n_total))
        if workers > 1:
            logger.info(
                "Automatic mode %s: parallel trial run enabled (%d workers)",
                str(phase_label),
                int(workers),
            )

        def _eval_one(idx: int, preset: dict) -> dict:
            trial_data = dict(search_base_data or {})
            trial_data.update(dict(preset or {}))
            if str(filter_key) in ("linear", "asym"):
                trial_data["phase_limit"] = round(
                    float(
                        _auto_phase_limit_clip(
                            trial_data.get("phase_limit", search_base_data.get("phase_limit", 400.0)),
                            default=400.0,
                        )
                    ),
                    1,
                )
            trial_data["comparison_mode"] = True
            trial_measurements = dict(measurements or {})
            trial_measurements["ui_data"] = trial_data
            cfg_trial = build_config(
                trial_data,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin=pin_obj,
                max_safe_boost=float(MAX_SAFE_BOOST),
            )
            try:
                setattr(cfg_trial, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
                setattr(cfg_trial, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
            except Exception:
                pass
            result = run_pipeline(cfg_trial, trial_measurements, include_response_arrays=False)
            metrics = _auto_score_result(
                result,
                auto_exc_freq_hz=_auto_safe_float(trial_data.get("_auto_exc_freq_hz", float("nan")), float("nan")),
                focus_lo_hz=focus_lo_hz if bool(use_refine_tiebreak) else None,
                focus_hi_hz=focus_hi_hz if bool(use_refine_tiebreak) else None,
                base_data=trial_data,
            )
            if bool(use_refine_tiebreak):
                soft_k = float(max(0.0, _auto_safe_float(cfg.refine_mode_soft_k, AUTO_MODE_REFINE_MODE_SOFT_K)))
                mode_ripple = _auto_safe_float(metrics.get("mode_ripple_db"), float("nan"))
                rank_base = _auto_safe_float(metrics.get("rank_score"), 0.0)
                soft_pen = float(soft_k) * max(0.0, float(mode_ripple)) if np.isfinite(mode_ripple) else 0.0
                metrics["mode_ripple_soft_penalty"] = float(soft_pen)
                metrics["rank_score_refine"] = float(rank_base - float(soft_pen))

            trial_preset = dict(preset or {})
            if str(filter_key) == "mixed":
                trial_preset["mixed_freq"] = round(
                    _clip(
                        trial_data.get("mixed_freq", search_base_data.get("mixed_freq", 180.0)),
                        80.0,
                        320.0,
                    ),
                    1,
                )
            elif str(filter_key) in ("linear", "asym"):
                trial_preset["phase_limit"] = round(
                    float(
                        _auto_phase_limit_clip(
                            trial_data.get("phase_limit", search_base_data.get("phase_limit", 400.0)),
                            default=400.0,
                        )
                    ),
                    1,
                )

            return {
                "idx": int(idx),
                "ok": True,
                "metrics": dict(metrics or {}),
                "trial_preset": dict(trial_preset),
            }

        def _consume_one(idx: int, out: dict) -> bool:
            phase_state.tried_n = int(idx)
            improved = False

            if bool(out.get("ok", False)):
                metrics = dict(out.get("metrics", {}) or {})
                metrics["trial"] = int(len(search_state.scored) + 1)
                metrics["phase"] = str(phase_label)
                trial_preset = dict(out.get("trial_preset", {}) or {})
                search_state.scored.append({"metrics": dict(metrics), "preset": dict(trial_preset)})
                if bool(use_refine_tiebreak):
                    search_state.phase2_pool.append(
                        {
                            "preset": dict(trial_preset),
                            "metrics": dict(metrics or {}),
                        }
                    )
                phase_state.ok_n += 1

                better = False
                refine_reason = "rank"
                if search_state.best_metrics is None:
                    better = True
                elif bool(use_refine_tiebreak):
                    better, refine_reason = _auto_is_better_refine(
                        metrics,
                        search_state.best_metrics,
                        goal,
                        return_reason=True,
                    )
                else:
                    better = bool(_auto_rank_key(metrics) < _auto_rank_key(search_state.best_metrics))

                if better:
                    prev_best = dict(search_state.best_metrics or {})
                    search_state.best_metrics = dict(metrics)
                    search_state.best_preset = dict(trial_preset)
                    improved = True
                    phase_state.improved_any = True
                    if bool(use_refine_tiebreak) and str(refine_reason) == "mode_ripple":
                        rank_prev = _auto_safe_float(prev_best.get("rank_score"), 0.0)
                        rank_new = _auto_safe_float(metrics.get("rank_score"), 0.0)
                        rank_eps = float(max(0.0, _auto_safe_float(cfg.refine_tiebreak_rank_eps, AUTO_MODE_REFINE_TIEBREAK_RANK_EPS)))
                        if abs(float(rank_new) - float(rank_prev)) <= float(rank_eps):
                            mode_hz_i = int(round(_auto_safe_float(metrics.get("mode_hz"), 0.0)))
                            band_lo_i = int(round(_auto_safe_float(metrics.get("mode_band_lo"), 0.0)))
                            band_hi_i = int(round(_auto_safe_float(metrics.get("mode_band_hi"), 0.0)))
                            ripple_prev = _auto_safe_float(prev_best.get("mode_ripple_db"), float("nan"))
                            ripple_new = _auto_safe_float(metrics.get("mode_ripple_db"), float("nan"))
                            logger.info(
                                "REFINE MODE-TIE: rank within eps (A=%.3f, B=%.3f), mode=%dHz band=%d-%dHz ripple %.3f -> %.3f",
                                float(rank_prev),
                                float(rank_new),
                                int(mode_hz_i),
                                int(band_lo_i),
                                int(band_hi_i),
                                float(ripple_prev if np.isfinite(ripple_prev) else 0.0),
                                float(ripple_new if np.isfinite(ripple_new) else 0.0),
                            )
            else:
                err_txt = str(out.get("error", "unknown error") or "unknown error")
                logger.warning(
                    f"Automatic mode trial {idx}/{n_total} failed "
                    f"({phase_label}): {err_txt}"
                )

            if callable(status_cb) and bool(improved):
                rank_now = _auto_safe_float((search_state.best_metrics or {}).get("rank_score"), 0.0)
                avg_now = _auto_safe_float((search_state.best_metrics or {}).get("avg_score"), 0.0)
                mode_now = _auto_safe_float((search_state.best_metrics or {}).get("mode_ripple_db"), float("nan"))
                boost_now = _auto_safe_float((search_state.best_metrics or {}).get("max_net_boost_db"), float("nan"))
                status_cb(
                    f"{status_prefix}: {phase_label} best improved {idx}/{n_total} "
                    f"(goal {goal}, rank {rank_now:.3f}, avg {avg_now:.3f}, "
                    f"mode {'n/a' if not np.isfinite(mode_now) else f'{mode_now:.3f} dB'}, "
                    f"boost {'n/a' if not np.isfinite(boost_now) else f'{boost_now:.2f} dB'}, "
                    f"ok {int(phase_state.ok_n)}/{int(phase_state.tried_n)})"
                )

            if int(plateau_after_no_improve) > 0:
                if improved:
                    phase_state.no_improve_streak = 0
                else:
                    phase_state.no_improve_streak += 1
                if phase_state.no_improve_streak >= int(plateau_after_no_improve):
                    phase_state.plateau_hit = True
                    best_now = "n/a" if not search_state.best_metrics else _auto_metric_text(search_state.best_metrics, goal)
                    move_txt = "plateau -> phase 2" if "1/2" in str(phase_label) else "plateau -> stop"
                    logger.info(
                        f"Automatic mode {phase_label}: no-improve plateau detected "
                        f"({int(plateau_after_no_improve)} rounds), {move_txt}."
                    )
                    if callable(status_cb):
                        status_cb(
                            f"{status_prefix}: {phase_label} {idx}/{n_total} "
                            f"(best {best_now}, {move_txt})"
                        )
                    return True
            return False

        idx_presets = list(enumerate(list(cands or []), start=1))
        stop_now = False
        if workers <= 1 or n_total <= 1:
            for idx, preset in idx_presets:
                try:
                    out = _eval_one(int(idx), dict(preset or {}))
                except Exception as exc:
                    out = {
                        "idx": int(idx),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if _consume_one(int(idx), out):
                    stop_now = True
                    break
        else:
            chunk_size = int(_auto_trial_chunk_size(workers))
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
                    chunk = idx_presets[c0 : c0 + int(chunk_size)]
                    fut_map = {
                        ex.submit(_eval_one, int(idx), dict(preset or {})): int(idx)
                        for idx, preset in chunk
                    }
                    chunk_out: dict[int, dict] = {}
                    for fut in as_completed(list(fut_map.keys())):
                        idx = int(fut_map.get(fut, 0))
                        try:
                            out = fut.result()
                            if not isinstance(out, dict):
                                out = {"idx": int(idx), "ok": False, "error": "invalid worker result"}
                        except Exception as exc:
                            out = {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        chunk_out[int(idx)] = dict(out)
                    for idx, _preset in chunk:
                        out = dict(
                            chunk_out.get(
                                int(idx),
                                {"idx": int(idx), "ok": False, "error": "missing worker result"},
                            )
                            or {}
                        )
                        if _consume_one(int(idx), out):
                            stop_now = True
                            break
                    if stop_now:
                        break

        return {
            "ok": int(phase_state.ok_n),
            "tried": int(phase_state.tried_n),
            "plateau_hit": bool(phase_state.plateau_hit),
            "improved_any": bool(phase_state.improved_any),
        }

    phase1_stats = _eval_candidates(
        candidates,
        phase_label="phase 1/2",
        plateau_after_no_improve=int(cfg.phase1_plateau_rounds),
        use_refine_tiebreak=False,
    )
    phase1_ok = int(phase1_stats.get("ok", 0) or 0)
    phase1_tried = int(phase1_stats.get("tried", 0) or 0)
    phase1_plateau_hit = bool(phase1_stats.get("plateau_hit", False))

    phase1_entries = [
        dict(it)
        for it in list(search_state.scored)
        if str(dict(it.get("metrics", {}) or {}).get("phase", "")) == "phase 1/2"
    ]
    phase1_top = sorted(
        phase1_entries,
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[: int(max(1, cfg.local_refine_top_k))]
    if phase1_top:
        p1m = dict(phase1_top[0].get("metrics", {}) or {})
        p1p = dict(phase1_top[0].get("preset", {}) or {})
        p1_mixed = _auto_safe_float(p1p.get("mixed_freq", search_base_data.get("mixed_freq", float("nan"))), float("nan"))
        p1_phase = _auto_safe_float(p1p.get("phase_limit", search_base_data.get("phase_limit", float("nan"))), float("nan"))
        p1_tdc = _auto_safe_float(p1p.get("tdc_strength", search_base_data.get("tdc_strength", float("nan"))), float("nan"))
        p1_mode = _auto_safe_float(p1m.get("mode_ripple_db"), float("nan"))
        p1_boost = _auto_safe_float(p1m.get("max_net_boost_db"), float("nan"))
        p1_mode_txt = f"{p1_mode:.3f} dB" if np.isfinite(p1_mode) else "n/a"
        p1_boost_txt = f"{p1_boost:.2f} dB" if np.isfinite(p1_boost) else "n/a"
        if str(filter_key) == "mixed":
            p1_detail = f"mixed_freq={p1_mixed:.1f} Hz, tdc={p1_tdc:.1f}"
        elif str(filter_key) in ("linear", "asym"):
            p1_detail = (
                f"phase_limit={p1_phase:.1f} Hz, tdc={p1_tdc:.1f}"
                if np.isfinite(p1_phase)
                else f"phase_limit=n/a, tdc={p1_tdc:.1f}"
            )
        else:
            p1_detail = f"tdc={p1_tdc:.1f}"
        logger.info(
            "Automatic mode Phase1 done: "
            f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
            f"{p1_detail}"
        )
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: Phase1 done "
                f"rank={_auto_safe_float(p1m.get('rank_score'), 0.0):.3f}, "
                f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                f"mode_ripple={p1_mode_txt}, "
                f"boost={p1_boost_txt}, "
                f"{p1_detail}"
            )

    phase1_best_metrics = dict(phase1_top[0].get("metrics", {}) or {}) if phase1_top else None
    phase1_best_preset = dict(phase1_top[0].get("preset", {}) or {}) if phase1_top else None
    phase2_focus_lo = float("nan")
    phase2_focus_hi = float("nan")
    if bool(cfg.local_refine_enabled) and phase1_top and _auto_goal_uses_local_refine(goal):
        ref_profile = _auto_build_refine_profile(
            base_data=search_base_data,
            phase1_top=phase1_top,
        )
        phase2_focus_lo = float(_auto_safe_float(ref_profile.get("focus_lo", float("nan")), float("nan")))
        phase2_focus_hi = float(_auto_safe_float(ref_profile.get("focus_hi", float("nan")), float("nan")))
        for ci, item in enumerate(phase1_top, start=1):
            center = dict(item.get("preset", {}) or {})
            c_mixed = _auto_safe_float(center.get("mixed_freq", search_base_data.get("mixed_freq", float("nan"))), float("nan"))
            c_phase = _auto_safe_float(center.get("phase_limit", search_base_data.get("phase_limit", float("nan"))), float("nan"))
            local_detail = None
            if str(filter_key) == "mixed":
                local_detail = f"mixed_freq={c_mixed:.1f} Hz"
            elif str(filter_key) in ("linear", "asym"):
                local_detail = (
                    f"phase refine phase_limit={c_phase:.1f} Hz"
                    if np.isfinite(c_phase)
                    else "phase refine phase_limit=n/a"
                )
            if local_detail is not None:
                logger.info(
                    "Automatic mode Local refine: "
                    f"center #{ci} {local_detail}"
                )
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: Local refine "
                        f"center #{ci} {local_detail}"
                    )

            local_seed = int(seed + 7919 + ci * 100003)
            local_shrink = float(
                _auto_adaptive_shrink_factor(
                    phase1_top,
                    base_shrink=float(cfg.local_refine_shrink),
                    plateau_hit=bool(phase1_plateau_hit),
                )
            )
            local_candidates = _build_auto_mode_candidates_local(
                search_base_data,
                center,
                int(cfg.local_refine_trials_per_top),
                int(local_seed),
                shrink=float(local_shrink),
            )
            before = dict(search_state.best_metrics or {})
            stats = _eval_candidates(
                local_candidates,
                phase_label=f"phase 2/2 local center#{ci}",
                plateau_after_no_improve=0,
                use_refine_tiebreak=True,
                focus_lo_hz=float(phase2_focus_lo) if np.isfinite(phase2_focus_lo) else None,
                focus_hi_hz=float(phase2_focus_hi) if np.isfinite(phase2_focus_hi) else None,
            )
            phase2_ok += int(stats.get("ok", 0) or 0)
            phase2_tried += int(stats.get("tried", 0) or 0)
            if bool(stats.get("improved_any", False)):
                logger.info(
                    "Automatic mode Local refine winner improved: "
                    f"avg_score {_auto_safe_float(before.get('avg_score'), 0.0):.3f}"
                    f" -> {_auto_safe_float((search_state.best_metrics or {}).get('avg_score'), 0.0):.3f}, "
                    f"rank_score {_auto_safe_float(before.get('rank_score'), 0.0):.3f}"
                    f" -> {_auto_safe_float((search_state.best_metrics or {}).get('rank_score'), 0.0):.3f}"
                )

    if bool(cfg.local_refine_keep_best_phase1) and isinstance(phase1_best_metrics, dict):
        if search_state.best_metrics is None or _auto_rank_key(search_state.best_metrics) > _auto_rank_key(phase1_best_metrics):
            search_state.best_metrics = dict(phase1_best_metrics)
            search_state.best_preset = dict(phase1_best_preset or {})

    if (
        bool(cfg.phase3_micro_enabled)
        and _auto_goal_uses_local_refine(goal)
        and isinstance(search_state.best_preset, dict)
        and bool(search_state.best_preset)
    ):
        micro_shrink = float(
            _auto_adaptive_shrink_factor(
                phase1_top,
                base_shrink=float(cfg.adaptive_shrink_max),
                plateau_hit=bool(phase1_plateau_hit),
            )
        )
        micro_candidates = _build_auto_mode_candidates_micro(
            search_base_data,
            dict(search_state.best_preset or {}),
            n_trials=int(cfg.phase3_micro_trials),
            shrink=float(micro_shrink),
        )
        logger.info(f"Phase3 micro size: {int(len(micro_candidates))}")
        if callable(status_cb):
            status_cb(
                f"CamillaFIR automatic mode: Phase3 micro "
                f"{int(len(micro_candidates))} trials around current best"
            )
        before_micro = dict(search_state.best_metrics or {})
        micro_stats = _eval_candidates(
            micro_candidates,
            phase_label="phase 3/3 micro",
            plateau_after_no_improve=0,
            use_refine_tiebreak=True,
            focus_lo_hz=float(phase2_focus_lo) if np.isfinite(phase2_focus_lo) else None,
            focus_hi_hz=float(phase2_focus_hi) if np.isfinite(phase2_focus_hi) else None,
        )
        phase2_ok += int(micro_stats.get("ok", 0) or 0)
        phase2_tried += int(micro_stats.get("tried", 0) or 0)
        if bool(micro_stats.get("improved_any", False)):
            logger.info(
                "Automatic mode Phase3 micro improved: "
                f"avg_score {_auto_safe_float(before_micro.get('avg_score'), 0.0):.3f}"
                f" -> {_auto_safe_float((search_state.best_metrics or {}).get('avg_score'), 0.0):.3f}, "
                f"rank_score {_auto_safe_float(before_micro.get('rank_score'), 0.0):.3f}"
                f" -> {_auto_safe_float((search_state.best_metrics or {}).get('rank_score'), 0.0):.3f}"
            )

    phase2_pool_raw = [dict(it or {}) for it in (search_state.phase2_pool or []) if isinstance(it, dict)]
    if phase2_pool_raw:
        phase2_rank_vals = [
            _m(dict(it.get("metrics", {}) or {}), "rank_score", float("nan"))
            for it in phase2_pool_raw
        ]
        phase2_rank_vals = [float(v) for v in phase2_rank_vals if np.isfinite(v)]
        phase2_best_rank = max(phase2_rank_vals) if phase2_rank_vals else float("nan")
        rank_win = float(max(0.0, _auto_safe_float(cfg.phase2_pareto_rank_window, AUTO_MODE_PHASE2_PARETO_RANK_WINDOW)))
        phase2_kept = []
        for it in phase2_pool_raw:
            r = _m(dict(it.get("metrics", {}) or {}), "rank_score", float("nan"))
            if np.isfinite(phase2_best_rank):
                if np.isfinite(r) and float(r) >= float(phase2_best_rank) - float(rank_win):
                    phase2_kept.append(dict(it))
            else:
                phase2_kept.append(dict(it))
        phase2_kept = sorted(
            phase2_kept,
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "rank_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )[: int(max(1, cfg.phase2_pareto_pool_max))]
        logger.info(
            "Phase2 pool size: "
            f"{int(len(phase2_pool_raw))} (kept {int(len(phase2_kept))})"
        )
        if bool(cfg.phase2_hard_gate_enabled) and len(phase2_kept) >= int(max(3, cfg.phase2_hard_gate_min_keep)):
            pre_n = int(len(phase2_kept))
            phase2_kept, ev_thr, rp_thr = _auto_phase2_hard_gate_pool(
                phase2_kept,
                min_keep=int(cfg.phase2_hard_gate_min_keep),
                keep_event_fraction=float(cfg.phase2_hard_gate_keep_event_fraction),
                keep_ripple_fraction=float(cfg.phase2_hard_gate_keep_ripple_fraction),
                fallback_to_rank=bool(cfg.phase2_hard_gate_fallback_to_rank),
            )
            post_n = int(len(phase2_kept))
            logger.info(
                "Phase2 hard-gate: kept %d/%d (event<=%.3f, ripple<=%.3f)",
                int(post_n),
                int(pre_n),
                float(ev_thr) if np.isfinite(ev_thr) else float("nan"),
                float(rp_thr) if np.isfinite(rp_thr) else float("nan"),
            )

        pareto_min_n = int(max(1, cfg.phase2_pareto_pool_min))
        if len(phase2_kept) >= pareto_min_n:
            front = _auto_phase2_pareto_front(phase2_kept)
            logger.info(f"Pareto front size: {int(len(front))}")
            rank_best = dict(phase2_kept[0])
            pareto_winner = _auto_phase2_pick_pareto_winner(
                front,
                phase2_kept,
                acoustic_drop=float(_auto_safe_float(cfg.phase2_pareto_acoustic_drop, AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP)),
            )
            if isinstance(pareto_winner, dict):
                w_metrics = dict(pareto_winner.get("metrics", {}) or {})
                w_preset = dict(pareto_winner.get("preset", {}) or {})
                w_mode_ripple = _auto_mode_ripple_for_pareto(w_metrics)
                w_rms20 = _auto_realized_rms_20_200_for_pareto(w_metrics)
                w_pre_l, w_pre_r, w_prepost = _auto_prepost_lr_for_pareto(w_metrics)
                w_boost = _m(w_metrics, "max_net_boost_db", float("nan"))
                logger.info(
                    "Pareto objectives include prepost: L=%.4f R=%.4f -> max=%.4f",
                    float(w_pre_l) if np.isfinite(w_pre_l) else float("nan"),
                    float(w_pre_r) if np.isfinite(w_pre_r) else float("nan"),
                    float(w_prepost) if np.isfinite(w_prepost) else float("nan"),
                )
                logger.info(
                    "Pareto winner: "
                    f"avg={_m(w_metrics, 'avg_score', 0.0):.3f}, "
                    f"prepost={w_prepost if np.isfinite(w_prepost) else float('nan'):.4f}, "
                    f"mode_ripple={w_mode_ripple if np.isfinite(w_mode_ripple) else float('nan'):.3f}, "
                    f"rms20_200={w_rms20 if np.isfinite(w_rms20) else float('nan'):.3f}, "
                    f"net_boost={w_boost if np.isfinite(w_boost) else float('nan'):.3f}"
                )
                rb_metrics = dict(rank_best.get("metrics", {}) or {})
                rb_prepost = _auto_prepost_for_pareto(rb_metrics)
                rb_mode_ripple = _auto_mode_ripple_for_pareto(rb_metrics)
                logger.info(
                    "Best-by-rank would have been: "
                    f"avg={_m(rb_metrics, 'avg_score', 0.0):.3f}, "
                    f"prepost={rb_prepost if np.isfinite(rb_prepost) else float('nan'):.4f}, "
                    f"mode_ripple={rb_mode_ripple:.3f}, "
                    f"rms20_200={_auto_realized_rms_20_200_for_pareto(rb_metrics):.3f}, "
                    f"net_boost={_m(rb_metrics, 'max_net_boost_db', float('nan')):.3f}"
                )
                logger.info(
                    "Pareto winner vs best-by-rank: "
                    f"avg {_m(w_metrics, 'avg_score', 0.0):.3f} vs {_m(rb_metrics, 'avg_score', 0.0):.3f}, "
                    f"prepost {w_prepost if np.isfinite(w_prepost) else float('nan'):.4f} vs "
                    f"{rb_prepost if np.isfinite(rb_prepost) else float('nan'):.4f}, "
                    f"mode_ripple {w_mode_ripple if np.isfinite(w_mode_ripple) else float('nan'):.3f} vs "
                    f"{rb_mode_ripple if np.isfinite(rb_mode_ripple) else float('nan'):.3f}"
                )
                search_state.best_metrics = dict(w_metrics)
                search_state.best_preset = dict(w_preset)
        else:
            logger.info(
                "Pareto front skipped: "
                f"phase2 kept pool too small ({int(len(phase2_kept))} < {int(pareto_min_n)})"
            )

    if search_state.best_metrics is None or not isinstance(search_state.best_preset, dict):
        return None

    # Materialize full output only once for the final winner.
    try:
        final_data = dict(search_base_data or {})
        final_data.update(dict(search_state.best_preset or {}))
        if str(filter_key) in ("linear", "asym"):
            final_data["phase_limit"] = round(
                float(
                    _auto_phase_limit_clip(
                        final_data.get("phase_limit", search_base_data.get("phase_limit", 400.0)),
                        default=400.0,
                    )
                ),
                1,
            )
        final_data["comparison_mode"] = True
        final_measurements = dict(measurements or {})
        final_measurements["ui_data"] = final_data

        cfg_final = build_config(
            final_data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin=pin_obj,
            max_safe_boost=float(MAX_SAFE_BOOST),
        )
        try:
            setattr(cfg_final, "bass_smooth_w_gamma", float(final_data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg_final, "bass_smooth_w_max", float(final_data.get("bass_smooth_w_max", 0.45)))
        except Exception:
            pass
        search_state.best_result = run_pipeline(cfg_final, final_measurements, include_response_arrays=True)
        search_state.best_result.metrics["summary"] = summarize_run(search_state.best_result)
    except Exception as exc:
        logger.warning(
            "Automatic mode final materialization failed: "
            f"{type(exc).__name__}: {exc}"
        )
        if search_state.best_result is None:
            return None

    top = sorted(
        search_state.scored,
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[:5]
    logger.info(
        "Automatic mode search result: "
        f"goal={goal}, basis={rank_basis}, {_auto_metric_text(search_state.best_metrics, goal)}, "
        f"rank={_auto_safe_float(search_state.best_metrics.get('rank_score'), 0.0):.3f}"
    )

    # --- Auto-mode cache: save best preset for this signature ---
    if bool(cfg.cache_enabled):
        try:
            best_hc_mode = str(search_base_data.get("hc_mode", "") or "").strip() or None
            best_hc_mode_builtin = _auto_builtin_target_name(best_hc_mode)
            measurement_sig = _auto_measurement_signature(measurements)
            sig = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=best_hc_mode,
                include_hc_mode=True,
            )
            sig_target = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            _auto_cache_put_best(
                sig,
                best_preset=dict(search_state.best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                best_hc_mode=best_hc_mode,
                measurement_sig=measurement_sig,
                goal=goal,
                filter_key=filter_key,
                program_version=program_version,
            )
            _auto_cache_put_best(
                sig_target,
                best_preset=dict(search_state.best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                best_hc_mode=best_hc_mode_builtin,
                measurement_sig=measurement_sig,
                goal=goal,
                filter_key=filter_key,
                program_version=program_version,
            )
            _auto_cache_put_target_for_measurements(
                measurements=measurements,
                best_hc_mode=best_hc_mode_builtin,
                best_preset=dict(search_state.best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                goal=goal,
                filter_key=filter_key,
                program_version=program_version,
            )
            _auto_cache_put_last_used_best(
                best_preset=dict(search_state.best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                best_hc_mode=best_hc_mode,
                measurement_sig=measurement_sig,
                goal=goal,
                filter_key=filter_key,
                program_version=program_version,
            )
            logger.info("Automatic mode: saved best preset to cache.")
        except Exception:
            pass

    best_auto_exc_hz = _auto_safe_float(
        dict(search_state.best_metrics or {}).get("auto_exc_zero_penalty_hz", float("nan")),
        float("nan"),
    )
    if np.isfinite(best_auto_exc_hz):
        best_auto_exc_hz = float(
            np.clip(
                float(best_auto_exc_hz),
                float(_auto_safe_float(cfg.exc_min_hz, AUTO_MODE_EXC_MIN_HZ)),
                float(_auto_safe_float(cfg.exc_max_hz, AUTO_MODE_EXC_MAX_HZ)),
            )
        )

    return {
        "best_result": search_state.best_result,
        "best_metrics": dict(search_state.best_metrics),
        "best_preset": dict(search_state.best_preset or {}),
        "best_auto_exc_freq_hz": float(best_auto_exc_hz) if np.isfinite(best_auto_exc_hz) else float("nan"),
        "auto_goal": str(goal),
        "selection_basis": str(rank_basis),
        "top": top,
        "trials_total": int(phase1_tried + phase2_tried),
        "trials_ok": int(len(search_state.scored)),
        "trials_phase1_total": int(phase1_tried),
        "trials_phase1_ok": int(phase1_ok),
        "trials_phase2_total": int(phase2_tried),
        "trials_phase2_ok": int(phase2_ok),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }


class _AutoModeSearcher:
    def __init__(
        self,
        *,
        base_data: dict,
        measurements: dict,
        fs_v: int,
        taps_v: int,
        xos: list,
        hpf: dict | None,
        hc_f,
        hc_m,
        pin_obj,
        status_cb,
        n_trials: int = AUTO_MODE_TRIALS,
    ):
        self.base_data = dict(base_data or {})
        self.measurements = dict(measurements or {})
        self.fs_v = int(fs_v)
        self.taps_v = int(taps_v)
        self.xos = list(xos or [])
        self.hpf = hpf
        self.hc_f = hc_f
        self.hc_m = hc_m
        self.pin_obj = pin_obj
        self.status_cb = status_cb
        self.n_trials = int(n_trials)

    def run(self) -> dict | None:
        return _run_auto_mode_search_impl(
            base_data=dict(self.base_data or {}),
            measurements=dict(self.measurements or {}),
            fs_v=int(self.fs_v),
            taps_v=int(self.taps_v),
            xos=list(self.xos or []),
            hpf=self.hpf,
            hc_f=self.hc_f,
            hc_m=self.hc_m,
            pin_obj=self.pin_obj,
            status_cb=self.status_cb,
            n_trials=int(self.n_trials),
        )


def _run_auto_mode_search(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int = AUTO_MODE_TRIALS,
) -> dict | None:
    return _AutoModeSearcher(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        n_trials=int(n_trials),
    ).run()


