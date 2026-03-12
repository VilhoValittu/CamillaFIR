import logging
import math
import json
import os
import hashlib
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..dsp import camillafir_dsp as dsp
from ..dsp.target_match import target_match_from_stats
from ..engine import build_config, run_pipeline, summarize_run
from ..app_paths import camillafir_data_dir
from ..ui import camillafir_plot as plots
from ..ui.camillafir_housecurve import get_house_curve_by_name
from .auto_mode.cache_signature import (
    _auto_apply_seed,
    _auto_cache_bucket,
    _auto_cache_bucket_template,
    _auto_cache_empty,
    _auto_cache_get_best,
    _auto_cache_get_best_target,
    _auto_cache_get_entry,
    _auto_cache_get_last_used_best,
    _auto_cache_get_target_for_measurements,
    _auto_cache_load,
    _auto_cache_path,
    _auto_cache_put_best,
    _auto_cache_put_last_used_best,
    _auto_cache_put_target_for_measurements,
    _auto_cache_save,
    _auto_measurement_signature,
    _auto_program_version,
    _auto_seed_from_signature,
    _auto_signature,
    get_auto_mode_cache_path,
)
from .auto_mode.candidate_generation import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _build_auto_mode_candidates_micro,
    _build_auto_mode_candidates_optuna,
    _build_auto_mode_refine_candidates,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_micro_optuna_params,
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
    _suggest_auto_mode_candidate_micro_optuna,
    _suggest_auto_mode_candidate_optuna,
)
from .auto_mode.filter_priors import get_auto_mode_filter_seed_preset
from .auto_mode.scoring_ranking import (
    _auto_adaptive_shrink_factor,
    _auto_apply_goal_tiebreak_metrics,
    _auto_build_refine_profile,
    _auto_build_winner_explanation,
    _auto_gate_threshold,
    _auto_goal_uses_local_refine,
    _auto_hybrid_mixed_freq_penalty,
    _auto_is_better_refine,
    _auto_mode_ripple_for_pareto,
    _auto_phase2_hard_gate_pool,
    _auto_phase2_pareto_front,
    _auto_phase2_pareto_vector,
    _auto_phase2_pick_pareto_winner,
    _auto_prepost_for_pareto,
    _auto_prepost_lr_for_pareto,
    _auto_rank_key,
    _auto_rank_key_acoustic,
    _auto_rank_key_flat,
    _auto_rank_key_goal,
    _auto_rank_key_hybrid,
    _auto_rank_key_low_ripple,
    _auto_rank_key_room_safe,
    _auto_realized_rms_20_200_for_pareto,
    _auto_reject,
    _auto_ripple_metric_for_gate,
    _auto_select_best_scored,
    _auto_target_mildness_index,
    _auto_target_result_mode_ripple,
    _auto_target_result_rank_key,
    _auto_target_result_tie_key,
    _pareto_dominates,
    _tc_score,
)
from .auto_mode.search_state import (
    _AutoModePhaseState,
    _AutoModeSearchState,
    _auto_set_search_winner,
)
from .auto_mode.target_preselection import (
    _auto_select_builtin_target_curve,
    _auto_target_adaptive_shortlist,
    _auto_target_insert_cached_wildcard,
    _auto_target_one_step_milder,
    _auto_target_preselect_score,
    _auto_target_slope_estimate,
)
from .auto_mode.shared import (
    AutoModeConfig,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
    AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    AUTO_MODE_OPTUNA_TELEMETRY,
    AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    _auto_builtin_target_name,
    _auto_filter_cache_key,
    _auto_goal,
    _auto_goal_basis_text,
    _auto_goal_norm,
    _auto_hash_array,
    _auto_is_phase_search_filter,
    _auto_metric_text,
    _auto_optimizer_backend,
    _auto_optuna_sampler_kwargs,
    _auto_phase_limit_center,
    _auto_phase_limit_clip,
    _auto_phase_limit_prior_penalty,
    _auto_safe_bool,
    _auto_safe_float,
    _auto_safe_int,
    _auto_sample_mag_low_pair,
    _auto_trial_chunk_size,
    _auto_trial_workers,
    _clip,
    _jitter,
    _m,
)

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

# --------------------------------------------------------------------
# Auto-mode cache + signature
# --------------------------------------------------------------------

# Stores best preset per (measurement + key settings) signature, so next run can start from it.
AUTO_MODE_CACHE_ENABLED = True
AUTO_MODE_CACHE_MAX_ITEMS = 64
AUTO_MODE_CACHE_FILENAME = "camillafir_auto_mode_cache.json"
AUTO_MODE_CACHE_FILTER_KEYS = ("linear", "mixed", "minimum", "asym")
_AUTO_CACHE_VERSION_MISMATCH_LOGGED = False
AUTO_MODE_OPTUNA_STORAGE_FILENAME = "camillafir_auto_mode_optuna_journal.log"
AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS = 24
AUTO_MODE_OPTUNA_USER_ATTR_OUT = "camillafir_out"


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
AUTO_MODE_PHASE3_MICRO_TRIALS = 6
AUTO_MODE_CACHE_REFINE_MICRO_TRIALS = 20
AUTO_MODE_CACHE_REFINE_MAX_ROUNDS = 8
AUTO_MODE_CACHE_REFINE_MIN_RANK_IMPROVEMENT = 0.02
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


def _auto_import_optuna():
    try:
        import optuna  # type: ignore
    except Exception:
        return None
    return optuna


def _auto_optuna_module_ready(optuna_mod) -> bool:
    if optuna_mod is None:
        return False
    try:
        sampler_cls = getattr(getattr(optuna_mod, "samplers", None), "TPESampler", None)
        create_study = getattr(optuna_mod, "create_study", None)
        trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    except Exception:
        return False
    return bool(
        callable(sampler_cls)
        and callable(create_study)
        and trial_state is not None
        and hasattr(trial_state, "FAIL")
    )


def _auto_optuna_storage_path() -> str:
    preferred_base = os.fspath(camillafir_data_dir())
    preferred_path = os.path.join(preferred_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME)

    try:
        os.makedirs(preferred_base, exist_ok=True)
    except Exception:
        try:
            os.makedirs(legacy_base, exist_ok=True)
        except Exception:
            pass
        return legacy_path
    return preferred_path


def _auto_optuna_study_name(*, study_sig: str | None, scope: str | None) -> str:
    sig_txt = str(study_sig or "").strip().lower()
    scope_txt = str(scope or "study").strip().lower()
    scope_tok = re.sub(r"[^a-z0-9._-]+", "-", scope_txt).strip("-") or "study"
    scope_hash = hashlib.sha1(scope_txt.encode("utf-8", "ignore")).hexdigest()[:12]
    sig_tok = sig_txt[:32] if sig_txt else "nosig"
    return f"camillafir-{scope_tok[:48]}-{scope_hash}-{sig_tok}"


def _auto_optuna_create_storage(optuna_mod, *, base_data: dict | None):
    if not _auto_safe_bool((base_data or {}).get("auto_mode_optuna_persistent_study", True), True):
        return None
    storages_mod = getattr(optuna_mod, "storages", None)
    if storages_mod is None:
        return None
    path = _auto_optuna_storage_path()
    candidates = [
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileOpenLock", None),
        ),
        (
            getattr(storages_mod, "JournalStorage", None),
            getattr(storages_mod, "JournalFileStorage", None),
            getattr(storages_mod, "JournalFileOpenLock", None),
        ),
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            None,
        ),
    ]
    for storage_cls, backend_cls, open_lock_cls in candidates:
        if not callable(storage_cls) or not callable(backend_cls):
            continue
        try:
            if callable(open_lock_cls):
                return storage_cls(backend_cls(path, lock_obj=open_lock_cls(path)))
            return storage_cls(backend_cls(path))
        except Exception:
            continue
    return None


def _auto_optuna_create_study(
    optuna_mod,
    *,
    sampler,
    base_data: dict | None,
    study_name: str | None,
):
    storage = _auto_optuna_create_storage(optuna_mod, base_data=base_data)
    if storage is not None and study_name:
        try:
            return optuna_mod.create_study(
                direction="maximize",
                sampler=sampler,
                storage=storage,
                study_name=str(study_name),
                load_if_exists=True,
            )
        except TypeError:
            pass
        except Exception as exc:
            logger.warning(
                "Automatic mode Optuna storage unavailable for study %s (%s: %s). "
                "Falling back to in-memory study.",
                str(study_name),
                type(exc).__name__,
                exc,
            )
    return optuna_mod.create_study(direction="maximize", sampler=sampler)


def _auto_optuna_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _auto_optuna_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_auto_optuna_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v):
            return "nan"
        if np.isposinf(v):
            return "inf"
        if np.isneginf(v):
            return "-inf"
        return round(v, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def _auto_optuna_scope_context_hash(
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    payload = {
        "center": _auto_optuna_jsonable(dict(center or {})),
        "shrink": None if shrink is None else round(float(shrink), 6),
        "extra": _auto_optuna_jsonable(dict(extra or {})),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]


def _auto_optuna_scope_with_context(
    scope_base: str,
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    ctx = _auto_optuna_scope_context_hash(center=center, shrink=shrink, extra=extra)
    return f"{str(scope_base)}-{ctx}"


def _auto_optuna_param_signature(params: dict | None) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    try:
        payload = json.dumps(_auto_optuna_jsonable(params), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(params)
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()


def _auto_optuna_trial_params(
    *,
    trial_obj,
    preset: dict | None,
    seed_to_params=None,
) -> dict:
    if callable(seed_to_params):
        try:
            params = dict(seed_to_params(dict(preset or {})) or {})
        except Exception:
            params = {}
        if params:
            return params
    try:
        params = dict(getattr(trial_obj, "params", {}) or {})
    except Exception:
        params = {}
    if params:
        return params
    return dict(preset or {})


def _auto_optuna_trial_payload_preset(user_attrs: dict | None) -> dict:
    payload = dict((user_attrs or {}).get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
    preset = payload.get("trial_preset", None)
    if not isinstance(preset, dict) or not preset:
        preset = payload.get("preset", None)
    return dict(preset or {})


def _auto_optuna_tdc_min(base_data: dict | None) -> float:
    if (_auto_goal_norm(_auto_goal(base_data)) == AUTO_MODE_GOAL_LOW_RIPPLE) and bool((base_data or {}).get("enable_tdc", True)):
        return 55.0
    return 15.0


def _auto_optuna_trial_distributions(optuna_mod, *, params: dict | None, base_data: dict | None) -> dict | None:
    params_in = dict(params or {})
    if not params_in:
        return None
    dist_mod = getattr(optuna_mod, "distributions", None)
    float_dist = getattr(dist_mod, "FloatDistribution", None)
    cat_dist = getattr(dist_mod, "CategoricalDistribution", None)
    if not callable(float_dist) or not callable(cat_dist):
        return None

    tdc_min = float(_auto_optuna_tdc_min(base_data))
    categorical_choices = {
        "tdc_slope_db_per_oct": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0],
        "max_slope_db_per_oct": [8.0, 10.0, 12.0, 14.0, 16.0],
        "max_slope_boost_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "max_slope_cut_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "filter_smooth": [12, 24, 48, 96],
    }
    float_ranges = {
        "fdw_cycles": (5.0, 16.0, 0.01),
        "tdc_strength": (tdc_min, 75.0, 0.1),
        "tdc_max_reduction_db": (6.0, 36.0, 0.1),
        "reg_strength": (15.0, 45.0, 0.1),
        "max_boost": (3.0, 8.0, 0.01),
        "mag_c_min": (float(AUTO_MODE_MAG_C_MIN_MIN_HZ), float(AUTO_MODE_MAG_C_MIN_MAX_HZ), 0.1),
        "mag_c_max": (170.0, 300.0, 0.1),
        "trans_width": (70.0, 150.0, 0.1),
        "bass_first_mode_max_hz": (150.0, 220.0, 0.1),
        "conf_pull_max_hz": (120.0, 260.0, 5.0),
        "low_bass_cut_hz": (float(AUTO_MODE_LOW_BASS_MIN_HZ), float(AUTO_MODE_LOW_BASS_MAX_HZ), 0.1),
        "mixed_freq": (80.0, 320.0, 0.1),
        "phase_limit": (float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ), 0.1),
    }

    out: dict[str, object] = {}
    for name in list(params_in.keys()):
        key = str(name)
        if key.endswith("_u"):
            try:
                out[key] = float_dist(0.0, 1.0)
            except Exception:
                return None
            continue
        if key in categorical_choices:
            try:
                out[key] = cat_dist(list(categorical_choices[key]))
            except Exception:
                return None
            continue
        if key in float_ranges:
            lo, hi, step = float_ranges[key]
            try:
                out[key] = float_dist(float(lo), float(hi), step=float(step))
            except Exception:
                return None
            continue
        return None
    return dict(out)


def _auto_optuna_build_completed_trial(
    optuna_mod,
    *,
    params: dict | None,
    value: float,
    user_attrs: dict | None,
    base_data: dict | None,
):
    create_trial = getattr(getattr(optuna_mod, "trial", None), "create_trial", None)
    if not callable(create_trial):
        create_trial = getattr(optuna_mod, "create_trial", None)
    if not callable(create_trial):
        return None
    distributions = _auto_optuna_trial_distributions(optuna_mod, params=params, base_data=base_data)
    if not distributions:
        return None

    trial_kwargs = {
        "params": dict(params or {}),
        "distributions": dict(distributions),
        "value": float(value),
        "user_attrs": dict(user_attrs or {}),
    }
    try:
        return create_trial(**trial_kwargs)
    except TypeError:
        pass
    except Exception:
        return None

    trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    complete_state = getattr(trial_state, "COMPLETE", None) if trial_state is not None else None
    if complete_state is None:
        return None
    try:
        return create_trial(state=complete_state, **trial_kwargs)
    except Exception:
        return None


def _auto_optuna_startup_for_phase_kind(cfg, *, phase_kind: str | None, total: int) -> int:
    kind = str(phase_kind or "").strip().lower()
    total_i = int(max(1, total))

    if kind == "phase1":
        base = int(getattr(cfg, "optuna_startup_phase1", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "target":
        base = int(getattr(cfg, "optuna_startup_target", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "local":
        base = int(getattr(cfg, "optuna_startup_local", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "micro":
        base = int(getattr(cfg, "optuna_startup_micro", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    else:
        base = int(getattr(cfg, "optuna_pilot_startup_trials", 12))

    return int(max(1, min(base, total_i)))


def _auto_optuna_is_refine_phase_kind(phase_kind: str | None) -> bool:
    kind = str(phase_kind or "").strip().lower()
    return kind in {"local", "micro"}


def _auto_optuna_constraint_scope_kind(scope: str | None) -> str:
    scope_txt = str(scope or "").strip().lower()
    if not scope_txt:
        return ""
    if "phase2-local" in scope_txt or "local_center_" in scope_txt:
        return "local"
    if "phase3-micro" in scope_txt or "cache-micro" in scope_txt or "cache_micro" in scope_txt:
        return "micro"
    return ""


def _auto_optuna_constraints_enabled_for_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> bool:
    data = dict(base_data or {})
    enabled = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints", AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    )
    if not enabled:
        return False
    refine_only = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints_refine_only", AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY),
        AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    )
    if not refine_only:
        return True
    if str(phase_kind or "").strip():
        return bool(_auto_optuna_is_refine_phase_kind(phase_kind))
    return bool(_auto_optuna_constraint_scope_kind(scope))


def _auto_optuna_effective_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> str:
    scope_txt = str(scope or "study").strip() or "study"
    if _auto_optuna_is_refine_phase_kind(phase_kind) and str(phase_kind or "").strip().lower() == "local":
        if not str(scope_txt).lower().endswith("-locv2") and "-locv2-" not in str(scope_txt).lower():
            scope_txt = f"{scope_txt}-locv2"
    if str(scope_txt).lower().endswith("-c1"):
        return str(scope_txt)
    if _auto_optuna_constraints_enabled_for_scope(base_data, scope_txt, phase_kind=phase_kind):
        return f"{scope_txt}-c1"
    return str(scope_txt)


def _auto_optuna_constraint_thresholds(base_data: dict | None, scope: str | None) -> dict:
    data = dict(base_data or {})
    kind = _auto_optuna_constraint_scope_kind(scope)

    max_mode_ripple = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_mode_ripple_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
        ),
    )
    max_events = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_events_severity",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
        ),
    )
    max_boost = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_net_boost_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
        ),
    )

    return {
        "kind": str(kind),
        "max_mode_ripple_db": float(max_mode_ripple),
        "max_events_severity": float(max_events),
        "max_net_boost_db": float(max_boost),
    }


def _auto_optuna_trial_out_payload(trial) -> dict:
    try:
        user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
    except Exception:
        user_attrs = {}
    out = user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {})
    if isinstance(out, dict):
        return dict(out or {})
    return {}


def _auto_optuna_constraint_vector_from_metrics(
    metrics: dict | None,
    *,
    max_mode_ripple_db: float,
    max_events_severity: float,
    max_net_boost_db: float,
    use_events: bool = True,
) -> tuple[float, float, float]:
    met = dict(metrics or {})

    ripple = _auto_ripple_metric_for_gate(met)
    events = _auto_safe_float(met.get("events_severity", float("nan")), float("nan"))
    boost = _auto_safe_float(met.get("max_net_boost_db", float("nan")), float("nan"))

    ripple_violation = 0.0
    event_violation = 0.0
    boost_violation = 0.0

    if np.isfinite(ripple):
        ripple_violation = float(max(0.0, float(ripple) - float(max_mode_ripple_db)))
    if bool(use_events) and np.isfinite(events):
        event_violation = float(max(0.0, float(events) - float(max_events_severity)))
    if np.isfinite(boost):
        boost_violation = float(max(0.0, float(boost) - float(max_net_boost_db)))

    return (
        float(ripple_violation),
        float(event_violation),
        float(boost_violation),
    )


def _auto_optuna_use_events_constraint(
    base_data: dict | None,
    *,
    phase_kind: str | None,
) -> bool:
    data = dict(base_data or {})
    kind = str(phase_kind or "").strip().lower()

    if kind in {"local", "micro"}:
        return _auto_safe_bool(
            data.get(
                "auto_mode_optuna_constraints_use_events_in_refine",
                AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
        )

    return True


def _auto_optuna_constraints_func(
    *,
    base_data: dict | None,
    scope: str | None,
    phase_kind: str | None = None,
):
    if not _auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind):
        return None

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    logger.info(
        "Automatic mode Optuna constraints: phase_kind=%s use_events=%s scope=%s",
        str(phase_kind or ""),
        str(bool(use_events)),
        str(scope or ""),
    )

    def _constraints(trial):
        out = _auto_optuna_trial_out_payload(trial)
        metrics = dict(out.get("metrics", {}) or {})
        return _auto_optuna_constraint_vector_from_metrics(
            metrics,
            max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
            max_events_severity=float(thr["max_events_severity"]),
            max_net_boost_db=float(thr["max_net_boost_db"]),
            use_events=bool(use_events),
        )

    return _constraints


def _auto_optuna_run_token(
    *,
    study_name: str | None,
    study_scope: str | None,
    seed: int,
    total: int,
    startup_trials: int,
) -> str:
    payload = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "seed": int(seed),
        "total": int(total),
        "startup_trials": int(startup_trials),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def _auto_optuna_constraint_info_for_metrics(
    *,
    base_data: dict | None,
    scope: str | None,
    metrics: dict | None,
    phase_kind: str | None = None,
) -> dict:
    enabled = bool(_auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind))
    if not enabled:
        return {
            "constraints_active": False,
            "feasible": None,
            "violations": {},
            "constraint_flags": {},
        }

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    vec = _auto_optuna_constraint_vector_from_metrics(
        dict(metrics or {}),
        max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
        max_events_severity=float(thr["max_events_severity"]),
        max_net_boost_db=float(thr["max_net_boost_db"]),
        use_events=bool(use_events),
    )
    ripple_v, events_v, boost_v = vec
    feasible = bool(
        float(ripple_v) <= 0.0
        and float(events_v) <= 0.0
        and float(boost_v) <= 0.0
    )
    return {
        "constraints_active": True,
        "feasible": bool(feasible),
        "violations": {
            "ripple": float(ripple_v),
            "events": float(events_v),
            "boost": float(boost_v),
        },
        "constraint_flags": {
            "use_events": bool(use_events),
        },
    }


def _auto_optuna_trial_objective_value(trial, out_payload: dict | None = None) -> float | None:
    out = dict(out_payload or {})
    opt_meta = dict(out.get("optuna", {}) or {})
    val = opt_meta.get("objective_value", None)
    try:
        if val is not None and np.isfinite(float(val)):
            return float(val)
    except Exception:
        pass

    direct_val = getattr(trial, "value", None)
    try:
        if direct_val is not None and np.isfinite(float(direct_val)):
            return float(direct_val)
    except Exception:
        pass

    vals = getattr(trial, "values", None)
    try:
        if vals and np.isfinite(float(vals[0])):
            return float(vals[0])
    except Exception:
        pass

    return None


def _auto_optuna_attach_out_telemetry(
    out: dict | None,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None = None,
    run_token: str,
    source: str,
    objective_value_num: float | None,
) -> dict:
    out2 = dict(out or {})
    metrics = dict(out2.get("metrics", {}) or {})
    cinfo = _auto_optuna_constraint_info_for_metrics(
        base_data=base_data,
        scope=study_scope,
        metrics=metrics,
        phase_kind=phase_kind,
    )
    out2["optuna"] = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(phase_kind or ""),
        "run_token": str(run_token),
        "source": str(source or ""),
        "objective_value": (
            None
            if objective_value_num is None or not np.isfinite(float(objective_value_num))
            else float(objective_value_num)
        ),
        "constraints_active": bool(cinfo.get("constraints_active", False)),
        "feasible": cinfo.get("feasible", None),
        "violations": dict(cinfo.get("violations", {}) or {}),
        "constraint_flags": dict(cinfo.get("constraint_flags", {}) or {}),
    }
    return out2


def _auto_metric_summary(values) -> dict:
    vals = []
    for v in list(values or []):
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(float(fv))
        except Exception:
            pass

    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}

    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _auto_metric_summary_text(name: str, summary: dict | None, ndigits: int = 3) -> str:
    s = dict(summary or {})
    if int(s.get("count", 0) or 0) <= 0:
        return f"{str(name)} n/a"

    def _fmt(x):
        try:
            fx = float(x)
            if np.isfinite(fx):
                return f"{fx:.{int(ndigits)}f}"
        except Exception:
            pass
        return "n/a"

    return (
        f"{str(name)} min/med/max "
        f"{_fmt(s.get('min'))}/{_fmt(s.get('median'))}/{_fmt(s.get('max'))}"
    )


def _auto_optuna_base_data_without_constraints(base_data: dict | None) -> dict:
    data = dict(base_data or {})
    data["auto_mode_optuna_constraints"] = False
    return data


def _auto_optuna_needs_zero_feasible_rescue(
    *,
    base_data: dict | None,
    phase_kind: str | None,
    telemetry: dict | None,
) -> bool:
    tel = dict(telemetry or {})
    enabled = _auto_safe_bool(
        (base_data or {}).get(
            "auto_mode_optuna_constraints_zero_feasible_fallback",
            AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
        ),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    )
    if not enabled:
        return False

    kind = str(phase_kind or "").strip().lower()
    if kind not in {"local", "micro"}:
        return False
    if not bool(tel.get("constraints_active", False)):
        return False
    if _auto_optuna_is_refine_phase_kind(phase_kind):
        cflags = dict(tel.get("constraint_flags", {}) or {})
        if not bool(cflags.get("use_events", True)):
            vc = dict(tel.get("violation_counts", {}) or {})
            ripple_bad = int(vc.get("ripple", 0) or 0)
            boost_bad = int(vc.get("boost", 0) or 0)
            if ripple_bad == 0 and boost_bad == 0:
                return False

    complete_n = int(tel.get("complete_trials", 0) or 0)
    feasible_n = int(tel.get("feasible_trials", 0) or 0)
    infeasible_n = int(tel.get("infeasible_trials", 0) or 0)
    return bool(complete_n > 0 and feasible_n == 0 and infeasible_n > 0)


def _auto_optuna_build_run_telemetry(
    study,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None,
    run_token: str,
    requested_total: int,
    startup_trials: int,
    duplicate_skips: int,
    duplicate_replays: int,
    duplicate_reserved: int,
) -> dict:
    try:
        trials = list(
            study.get_trials(deepcopy=False)
            if hasattr(study, "get_trials")
            else getattr(study, "trials", [])
        )
    except Exception:
        trials = []

    run_trials = []
    for tr in list(trials or []):
        try:
            user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
        except Exception:
            user_attrs = {}
        out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
        opt_meta = dict(out.get("optuna", {}) or {})
        if str(opt_meta.get("run_token", "")) == str(run_token):
            run_trials.append((tr, out, opt_meta))

    state_counts = {}
    complete_n = 0
    fail_n = 0
    feasible_n = 0
    infeasible_n = 0
    best_raw_value = None
    best_raw_trial = None
    best_feasible_value = None
    best_feasible_trial = None

    violation_counts = {"ripple": 0, "events": 0, "boost": 0}
    violation_max = {"ripple": 0.0, "events": 0.0, "boost": 0.0}
    source_counts = {}
    events_all = []
    events_feasible = []
    events_infeasible = []
    constraint_flags = {}
    run_phase_kind = ""

    for tr, out, opt_meta in list(run_trials):
        state_obj = getattr(tr, "state", None)
        state_name = str(getattr(state_obj, "name", state_obj or "UNKNOWN"))
        state_counts[state_name] = int(state_counts.get(state_name, 0) or 0) + 1

        source = str(opt_meta.get("source", "") or "")
        if source:
            source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
        if not run_phase_kind:
            run_phase_kind = str(opt_meta.get("phase_kind", "") or "")

        if state_name == "COMPLETE":
            complete_n += 1
        elif state_name == "FAIL":
            fail_n += 1

        constraints_active = bool(opt_meta.get("constraints_active", False))
        feasible = opt_meta.get("feasible", None)
        violations = dict(opt_meta.get("violations", {}) or {})
        trial_constraint_flags = dict(opt_meta.get("constraint_flags", {}) or {})
        if trial_constraint_flags and not constraint_flags:
            constraint_flags = dict(trial_constraint_flags)

        if constraints_active and feasible is True:
            feasible_n += 1
        elif constraints_active and feasible is False:
            infeasible_n += 1

        for key in ("ripple", "events", "boost"):
            v = _auto_safe_float(violations.get(key, 0.0), 0.0)
            if float(v) > 0.0:
                violation_counts[key] = int(violation_counts.get(key, 0) or 0) + 1
                violation_max[key] = float(max(float(violation_max.get(key, 0.0) or 0.0), float(v)))

        if state_name == "COMPLETE":
            metrics = dict(out.get("metrics", {}) or {})
            events_val = _auto_safe_float(metrics.get("events_severity", float("nan")), float("nan"))
            if np.isfinite(events_val):
                events_all.append(float(events_val))
                if constraints_active and feasible is True:
                    events_feasible.append(float(events_val))
                elif constraints_active and feasible is False:
                    events_infeasible.append(float(events_val))
            obj_val = _auto_optuna_trial_objective_value(tr, out)
            if obj_val is not None:
                if best_raw_value is None or float(obj_val) > float(best_raw_value):
                    best_raw_value = float(obj_val)
                    best_raw_trial = int(getattr(tr, "number", -1))

                feasible_ok = opt_meta.get("feasible", None)
                if feasible_ok is True:
                    if best_feasible_value is None or float(obj_val) > float(best_feasible_value):
                        best_feasible_value = float(obj_val)
                        best_feasible_trial = int(getattr(tr, "number", -1))

    startup_complete = int(min(max(1, int(startup_trials)), int(complete_n))) if complete_n > 0 else 0
    model_complete = int(max(0, int(complete_n) - int(startup_complete)))
    constraints_active_any = bool(_auto_optuna_constraints_enabled_for_scope(base_data, study_scope, phase_kind=phase_kind or run_phase_kind))
    constraint_thresholds = {}
    try:
        thr = _auto_optuna_constraint_thresholds(base_data, study_scope)
        constraint_thresholds = {
            "max_events_severity": float(thr["max_events_severity"]),
            "max_mode_ripple_db": float(thr["max_mode_ripple_db"]),
            "max_net_boost_db": float(thr["max_net_boost_db"]),
        }
    except Exception:
        constraint_thresholds = {}

    return {
        "study_name": str(study_name or "in-memory"),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(run_phase_kind or phase_kind or ""),
        "run_token": str(run_token),
        "requested_total": int(requested_total),
        "run_trials": int(len(run_trials)),
        "complete_trials": int(complete_n),
        "failed_trials": int(fail_n),
        "state_counts": dict(state_counts or {}),
        "startup_trials": int(startup_trials),
        "startup_complete": int(startup_complete),
        "model_complete": int(model_complete),
        "duplicate_skips": int(duplicate_skips),
        "duplicate_replays": int(duplicate_replays),
        "duplicate_reserved": int(duplicate_reserved),
        "constraints_active": bool(constraints_active_any),
        "feasible_trials": int(feasible_n) if bool(constraints_active_any) else 0,
        "infeasible_trials": int(infeasible_n) if bool(constraints_active_any) else 0,
        "best_raw_value": best_raw_value,
        "best_raw_trial": best_raw_trial,
        "best_feasible_value": best_feasible_value if bool(constraints_active_any) else None,
        "best_feasible_trial": best_feasible_trial if bool(constraints_active_any) else None,
        "violation_counts": dict(violation_counts or {}),
        "violation_max": dict(violation_max or {}),
        "events_summary": _auto_metric_summary(events_all),
        "events_feasible_summary": _auto_metric_summary(events_feasible),
        "events_infeasible_summary": _auto_metric_summary(events_infeasible),
        "constraint_thresholds": dict(constraint_thresholds or {}),
        "constraint_flags": dict(constraint_flags or {}),
        "source_counts": dict(source_counts or {}),
    }


def _auto_optuna_log_run_telemetry(logger, *, phase_label: str, tel: dict | None) -> None:
    tel = dict(tel or {})
    if not tel:
        return

    msg = (
        "Automatic mode Optuna telemetry [%s]: requested=%d run=%d complete=%d fail=%d "
        "startup=%d model=%d dup=%d(replay=%d,reserved=%d)"
        % (
            str(phase_label),
            int(tel.get("requested_total", 0) or 0),
            int(tel.get("run_trials", 0) or 0),
            int(tel.get("complete_trials", 0) or 0),
            int(tel.get("failed_trials", 0) or 0),
            int(tel.get("startup_complete", 0) or 0),
            int(tel.get("model_complete", 0) or 0),
            int(tel.get("duplicate_skips", 0) or 0),
            int(tel.get("duplicate_replays", 0) or 0),
            int(tel.get("duplicate_reserved", 0) or 0),
        )
    )
    logger.info(msg)

    if bool(tel.get("constraints_active", False)):
        cflags = dict(tel.get("constraint_flags", {}) or {})
        use_events = bool(cflags.get("use_events", True))
        logger.info(
            "Automatic mode Optuna feasible [%s]: feasible=%d infeasible=%d "
            "best_raw=%s best_feasible=%s violations(r=%d,e=%d,b=%d)",
            str(phase_label),
            int(tel.get("feasible_trials", 0) or 0),
            int(tel.get("infeasible_trials", 0) or 0),
            "n/a" if tel.get("best_raw_value", None) is None else f"{float(tel['best_raw_value']):.6f}",
            "n/a" if tel.get("best_feasible_value", None) is None else f"{float(tel['best_feasible_value']):.6f}",
            int((tel.get("violation_counts", {}) or {}).get("ripple", 0) or 0),
            int((tel.get("violation_counts", {}) or {}).get("events", 0) or 0),
            int((tel.get("violation_counts", {}) or {}).get("boost", 0) or 0),
        )
        if not use_events:
            logger.info(
                "Automatic mode Optuna refine constraints [%s]: events constraint disabled for refine scope",
                str(phase_label),
            )
        if (
            use_events
            and
            int(tel.get("complete_trials", 0) or 0) > 0
            and int(tel.get("feasible_trials", 0) or 0) == 0
            and int(tel.get("infeasible_trials", 0) or 0) > 0
        ):
            ev_all_txt = _auto_metric_summary_text("events", tel.get("events_summary", {}), 3)
            ev_bad_txt = _auto_metric_summary_text("events_bad", tel.get("events_infeasible_summary", {}), 3)
            ev_thr = ((tel.get("constraint_thresholds", {}) or {}).get("max_events_severity", None))
            logger.warning(
                "Automatic mode Optuna zero-feasible [%s]: all complete trials violated constraints, "
                "events<=%s required, %s, %s",
                str(phase_label),
                "n/a" if ev_thr is None else f"{float(ev_thr):.3f}",
                str(ev_all_txt),
                str(ev_bad_txt),
            )


def _auto_optuna_fmt_value(v, ndigits: int = 3) -> str:
    try:
        fv = float(v)
        if np.isfinite(fv):
            return f"{fv:.{int(ndigits)}f}"
    except Exception:
        pass
    return "n/a"


def _auto_optuna_telemetry_text(tel: dict | None) -> str:
    return _auto_optuna_telemetry_text_ex(tel, include_phase_kind=False)


def _auto_optuna_telemetry_text_ex(tel: dict | None, *, include_phase_kind: bool = False) -> str:
    t = dict(tel or {})
    if not t:
        return ""

    run_n = int(t.get("run_trials", 0) or 0)
    complete_n = int(t.get("complete_trials", 0) or 0)
    startup_n = int(t.get("startup_complete", 0) or 0)
    model_n = int(t.get("model_complete", 0) or 0)
    dup_n = int(t.get("duplicate_skips", 0) or 0)

    parts = [
        f"optuna run={run_n}",
        f"ok={complete_n}",
        f"startup={startup_n}",
        f"model={model_n}",
    ]
    if bool(include_phase_kind):
        phase_kind = str(t.get("phase_kind", "") or "").strip()
        if phase_kind:
            parts.insert(0, f"phase={phase_kind}")
    if dup_n > 0:
        parts.append(f"dup={dup_n}")

    if bool(t.get("constraints_active", False)):
        cflags = dict(t.get("constraint_flags", {}) or {})
        feas_n = int(t.get("feasible_trials", 0) or 0)
        infeas_n = int(t.get("infeasible_trials", 0) or 0)
        parts.append(f"feas={feas_n}/{feas_n + infeas_n}")
        if not bool(cflags.get("use_events", True)):
            parts.append("events=off")
        best_raw = t.get("best_raw_value", None)
        best_feas = t.get("best_feasible_value", None)
        if best_raw is not None:
            parts.append(f"raw={_auto_optuna_fmt_value(best_raw, 3)}")
        if best_feas is not None:
            parts.append(f"best={_auto_optuna_fmt_value(best_feas, 3)}")

        vc = dict(t.get("violation_counts", {}) or {})
        vr = int(vc.get("ripple", 0) or 0)
        ve = int(vc.get("events", 0) or 0)
        vb = int(vc.get("boost", 0) or 0)
        if (vr + ve + vb) > 0:
            parts.append(f"viol r/e/b={vr}/{ve}/{vb}")
    else:
        best_raw = t.get("best_raw_value", None)
        if best_raw is not None:
            parts.append(f"best={_auto_optuna_fmt_value(best_raw, 3)}")

    return ", ".join(parts)


def _auto_optuna_events_debug_text(tel: dict | None, ndigits: int = 3) -> str:
    t = dict(tel or {})
    thr = dict(t.get("constraint_thresholds", {}) or {})
    cflags = dict(t.get("constraint_flags", {}) or {})
    use_events = bool(cflags.get("use_events", True))
    ev_thr = thr.get("max_events_severity", None)
    summ = dict(t.get("events_summary", {}) or {})

    def _fmt(x):
        try:
            fx = float(x)
            if np.isfinite(fx):
                return f"{fx:.{int(ndigits)}f}"
        except Exception:
            pass
        return "n/a"

    ev_body = "events n/a"
    if int(summ.get("count", 0) or 0) > 0:
        ev_body = (
            f"events min/med/max "
            f"{_fmt(summ.get('min'))}/{_fmt(summ.get('median'))}/{_fmt(summ.get('max'))}"
        )
    if not use_events:
        return f"events=off, {ev_body}"
    if ev_thr is None:
        return str(ev_body)
    return f"events<={_fmt(ev_thr)}, {ev_body}"


def _auto_optuna_fallback_summary_text(tel: dict | None) -> str:
    t = dict(tel or {})
    fallback_tel = dict(t.get("fallback_telemetry", {}) or {})
    constrained_txt = _auto_optuna_telemetry_text(t)
    fallback_txt = _auto_optuna_telemetry_text(fallback_tel)
    events_txt = _auto_optuna_events_debug_text(t, 3)

    parts = []
    if constrained_txt:
        parts.append(f"constrained {constrained_txt}")
    if fallback_txt:
        parts.append(f"fallback {fallback_txt}")
    if events_txt:
        parts.append(str(events_txt))
    return "; ".join(parts)


def _auto_optuna_telemetry_rollup(items: list[dict] | None) -> dict:
    arr = [dict(x or {}) for x in list(items or []) if isinstance(x, dict) and x]
    if not arr:
        return {}

    out = {
        "run_trials": 0,
        "complete_trials": 0,
        "failed_trials": 0,
        "startup_complete": 0,
        "model_complete": 0,
        "duplicate_skips": 0,
        "duplicate_replays": 0,
        "duplicate_reserved": 0,
        "constraints_active": False,
        "feasible_trials": 0,
        "infeasible_trials": 0,
        "best_raw_value": None,
        "best_feasible_value": None,
        "violation_counts": {"ripple": 0, "events": 0, "boost": 0},
    }

    for t in arr:
        out["run_trials"] += int(t.get("run_trials", 0) or 0)
        out["complete_trials"] += int(t.get("complete_trials", 0) or 0)
        out["failed_trials"] += int(t.get("failed_trials", 0) or 0)
        out["startup_complete"] += int(t.get("startup_complete", 0) or 0)
        out["model_complete"] += int(t.get("model_complete", 0) or 0)
        out["duplicate_skips"] += int(t.get("duplicate_skips", 0) or 0)
        out["duplicate_replays"] += int(t.get("duplicate_replays", 0) or 0)
        out["duplicate_reserved"] += int(t.get("duplicate_reserved", 0) or 0)

        if bool(t.get("constraints_active", False)):
            out["constraints_active"] = True
            out["feasible_trials"] += int(t.get("feasible_trials", 0) or 0)
            out["infeasible_trials"] += int(t.get("infeasible_trials", 0) or 0)

        br = t.get("best_raw_value", None)
        if br is not None:
            try:
                brf = float(br)
                if np.isfinite(brf) and (
                    out["best_raw_value"] is None or brf > float(out["best_raw_value"])
                ):
                    out["best_raw_value"] = float(brf)
            except Exception:
                pass

        bf = t.get("best_feasible_value", None)
        if bf is not None:
            try:
                bff = float(bf)
                if np.isfinite(bff) and (
                    out["best_feasible_value"] is None or bff > float(out["best_feasible_value"])
                ):
                    out["best_feasible_value"] = float(bff)
            except Exception:
                pass

        vc = dict(t.get("violation_counts", {}) or {})
        out["violation_counts"]["ripple"] += int(vc.get("ripple", 0) or 0)
        out["violation_counts"]["events"] += int(vc.get("events", 0) or 0)
        out["violation_counts"]["boost"] += int(vc.get("boost", 0) or 0)

    return out


def _auto_optuna_study_records(study, *, seed_to_params=None) -> dict[str, dict]:
    try:
        trials = study.get_trials(deepcopy=False)
    except TypeError:
        trials = study.get_trials()
    except Exception:
        trials = getattr(study, "trials", [])
    out: dict[str, dict] = {}
    for tr in list(trials or []):
        try:
            user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
        except Exception:
            user_attrs = {}
        payload_preset = _auto_optuna_trial_payload_preset(user_attrs)
        try:
            params = dict(getattr(tr, "params", {}) or {})
        except Exception:
            params = {}
        if callable(seed_to_params) and payload_preset:
            try:
                canonical_params = dict(seed_to_params(dict(payload_preset)) or {})
            except Exception:
                canonical_params = {}
            if canonical_params:
                params = dict(canonical_params)
        sig = _auto_optuna_param_signature(params)
        if not sig:
            continue
        rec = {"params": dict(params)}
        val = getattr(tr, "value", None)
        if val is None:
            vals = getattr(tr, "values", None)
            if isinstance(vals, (list, tuple)) and vals:
                val = vals[0]
        try:
            val_f = float(val)
        except Exception:
            val_f = float("nan")
        if np.isfinite(val_f):
            rec["value"] = float(val_f)
        state = getattr(tr, "state", None)
        if state is not None:
            rec["state"] = state
        cached_out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
        if cached_out:
            rec["out"] = cached_out
        out[sig] = rec
    return out


def _auto_optuna_remember_result(
    optuna_mod,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None = None,
    phase_kind: str | None = None,
    seed: int,
    preset: dict | None,
    metrics: dict | None,
    seed_to_params=None,
    use_refine_tiebreak: bool = False,
    out_payload: dict | None = None,
) -> bool:
    if (not _auto_optuna_module_ready(optuna_mod)) or not study_name or not callable(seed_to_params):
        return False
    params = {}
    try:
        params = dict(seed_to_params(dict(preset or {})) or {})
    except Exception:
        params = {}
    params_sig = _auto_optuna_param_signature(params)
    if not params_sig:
        return False
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=1,
        startup_trials=1,
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=1) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=1,
        **sampler_kwargs,
    )
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        base_data=base_data,
        study_name=study_name,
    )
    if params_sig in _auto_optuna_study_records(study, seed_to_params=seed_to_params):
        return False
    value = _auto_optuna_objective_value(
        dict(metrics or {}),
        use_refine_tiebreak=bool(use_refine_tiebreak),
    )
    payload = _auto_optuna_attach_out_telemetry(
        out_payload
        or {
            "ok": True,
            "metrics": dict(metrics or {}),
            "trial_preset": dict(preset or {}),
            "replayed_from_cache": True,
        },
        base_data=base_data,
        study_name=study_name,
        study_scope=scope_eff,
        phase_kind=phase_kind,
        run_token=run_token,
        source="remembered",
        objective_value_num=float(value) if np.isfinite(value) else None,
    )
    payload_json = _auto_optuna_jsonable(dict(payload or {}))
    if hasattr(study, "add_trial"):
        add_trial_obj = _auto_optuna_build_completed_trial(
            optuna_mod,
            params=params,
            value=float(value),
            user_attrs={AUTO_MODE_OPTUNA_USER_ATTR_OUT: payload_json},
            base_data=base_data,
        )
        if add_trial_obj is not None:
            try:
                study.add_trial(add_trial_obj)
                return True
            except Exception:
                pass
    if not hasattr(study, "enqueue_trial") or not hasattr(study, "ask") or not hasattr(study, "tell"):
        return False
    try:
        study.enqueue_trial(dict(params))
        trial_obj = study.ask()
    except Exception:
        return False
    try:
        if hasattr(trial_obj, "set_user_attr"):
            trial_obj.set_user_attr(
                AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                payload_json,
            )
    except Exception:
        pass
    try:
        study.tell(trial_obj, float(value))
    except Exception:
        return False
    return True


def _auto_optuna_objective_value(metrics: dict | None, *, use_refine_tiebreak: bool = False) -> float:
    met = dict(metrics or {})
    key = "rank_score_refine" if bool(use_refine_tiebreak) else "rank_score"
    value = _auto_safe_float(met.get(key, float("nan")), float("nan"))
    if (not np.isfinite(value)) and bool(use_refine_tiebreak):
        value = _auto_safe_float(met.get("rank_score", float("nan")), float("nan"))
    if np.isfinite(value):
        return float(value)
    return float(-1e12)


def _auto_run_optuna_eval_loop(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None = None,
    n_total: int,
    seed: int,
    startup_trials: int | None = None,
    base_data: dict | None,
    seed_presets: list[dict] | None,
    build_preset,
    eval_one,
    consume_one,
    objective_value,
    workers: int,
    seed_to_params=None,
    study_name: str | None = None,
    study_scope: str | None = None,
    phase_label: str | None = None,
    phase_kind: str | None = None,
) -> dict:
    if not _auto_optuna_module_ready(optuna_mod):
        return {}
    total = int(max(0, n_total))
    if total <= 0:
        return {}
    cfg_optuna = cfg if isinstance(cfg, AutoModeConfig) else AutoModeConfig.from_base_data(base_data)
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    startup_effective = _auto_optuna_startup_for_phase_kind(
        cfg_optuna,
        phase_kind=phase_kind,
        total=int(total),
    )
    if startup_trials is not None and not str(phase_kind or "").strip():
        startup_effective = int(max(1, min(int(total), _auto_safe_int(startup_trials, startup_effective))))
    logger.info(
        "Automatic mode Optuna startup policy: phase_kind=%s scope=%s total=%d startup=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(total),
        int(startup_effective),
    )
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=int(total),
        startup_trials=int(startup_effective),
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=int(workers)) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(startup_effective),
        **sampler_kwargs,
    )
    logger.info(
        "Automatic mode Optuna study %s: startup=%d total=%d",
        str(study_name or "in-memory"),
        int(startup_effective),
        int(total),
    )
    logger.info(
        "Automatic mode Optuna phase=%s scope=%s startup=%d total=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(startup_effective),
        int(total),
    )
    if callable(constraint_fn):
        thr = _auto_optuna_constraint_thresholds(base_data, scope_eff)
        use_events = _auto_optuna_use_events_constraint(
            base_data,
            phase_kind=phase_kind,
        )
        logger.info(
            "Automatic mode Optuna constraints enabled: scope=%s ripple<=%.3f events=%s boost<=%.3f",
            str(scope_eff),
            float(thr["max_mode_ripple_db"]),
            "off" if not bool(use_events) else f"{float(thr['max_events_severity']):.3f}",
            float(thr["max_net_boost_db"]),
        )
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        base_data=base_data,
        study_name=study_name,
    )
    fail_state = optuna_mod.trial.TrialState.FAIL
    duplicate_guard = bool(
        _auto_safe_bool((base_data or {}).get("auto_mode_optuna_avoid_duplicates", True), True)
    )
    known_records = _auto_optuna_study_records(study, seed_to_params=seed_to_params) if bool(duplicate_guard) else {}
    reserved_signatures: set[str] = set()
    duplicate_skips = 0
    duplicate_replays = 0
    duplicate_reserved = 0

    def _finalize_telemetry() -> dict:
        if not bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry", AUTO_MODE_OPTUNA_TELEMETRY),
                AUTO_MODE_OPTUNA_TELEMETRY,
            )
        ):
            return {}
        telemetry = _auto_optuna_build_run_telemetry(
            study,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            requested_total=int(total),
            startup_trials=int(startup_effective),
            duplicate_skips=int(duplicate_skips),
            duplicate_replays=int(duplicate_replays),
            duplicate_reserved=int(duplicate_reserved),
        )
        if bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry_log_summary", AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY),
                AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
            )
        ):
            _auto_optuna_log_run_telemetry(
                logger,
                phase_label=str(phase_label or scope_eff or "optuna"),
                tel=telemetry,
            )
        return dict(telemetry or {})

    def _tell(trial_obj, out: dict, *, params_sig: str = "", source: str = "optuna") -> None:
        value = None
        out_payload = dict(out or {})
        if bool(out_payload.get("ok", False)):
            try:
                value = float(objective_value(dict(out_payload or {})))
                if not np.isfinite(value):
                    value = float(-1e12)
            except Exception:
                value = float(-1e12)
        out_payload = _auto_optuna_attach_out_telemetry(
            out_payload,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            source=str(source or "optuna"),
            objective_value_num=value,
        )
        try:
            if hasattr(trial_obj, "set_user_attr"):
                trial_obj.set_user_attr(
                    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                    _auto_optuna_jsonable(dict(out_payload or {})),
                )
        except Exception:
            pass
        try:
            if bool(dict(out_payload or {}).get("ok", False)):
                study.tell(trial_obj, float(value))
            else:
                study.tell(trial_obj, state=fail_state)
        except Exception:
            pass
        if params_sig:
            reserved_signatures.discard(str(params_sig))
            rec = {"params_sig": str(params_sig)}
            if bool(dict(out_payload or {}).get("ok", False)) and value is not None and np.isfinite(value):
                rec["value"] = float(value)
            else:
                rec["state"] = fail_state
            if isinstance(out_payload, dict) and out_payload:
                rec["out"] = dict(out_payload or {})
            known_records[str(params_sig)] = rec

    def _reuse_duplicate_trial(trial_obj, params_sig: str) -> None:
        rec = dict(known_records.get(str(params_sig), {}) or {})
        out_prev = dict(rec.get("out", {}) or {})
        val = rec.get("value", None)
        out_payload = _auto_optuna_attach_out_telemetry(
            out_prev,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            source="replayed",
            objective_value_num=(
                float(val)
                if val is not None and np.isfinite(_auto_safe_float(val, float("nan")))
                else None
            ),
        )
        if out_payload and hasattr(trial_obj, "set_user_attr"):
            try:
                trial_obj.set_user_attr(
                    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                    _auto_optuna_jsonable(out_payload),
                )
            except Exception:
                pass
        try:
            if val is not None and np.isfinite(float(val)):
                study.tell(trial_obj, float(val))
            else:
                study.tell(trial_obj, state=fail_state)
        except Exception:
            pass

    def _ask_new_trial():
        nonlocal duplicate_reserved, duplicate_replays, duplicate_skips
        attempts = int(max(1, AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS))
        last_error = None
        for _ in range(attempts):
            try:
                trial_obj = study.ask()
                preset = dict(build_preset(trial_obj) or {})
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            params = _auto_optuna_trial_params(
                trial_obj=trial_obj,
                preset=preset,
                seed_to_params=seed_to_params,
            )
            params_sig = _auto_optuna_param_signature(params)
            if (not bool(duplicate_guard)) or (not params_sig):
                if params_sig:
                    reserved_signatures.add(str(params_sig))
                return trial_obj, preset, str(params_sig), None
            if params_sig in reserved_signatures:
                duplicate_skips += 1
                duplicate_reserved += 1
                reserved_out = _auto_optuna_attach_out_telemetry(
                    {
                        "ok": False,
                        "error": "duplicate suggestion reserved in current batch",
                    },
                    base_data=base_data,
                    study_name=study_name,
                    study_scope=scope_eff,
                    phase_kind=phase_kind,
                    run_token=run_token,
                    source="reserved",
                    objective_value_num=None,
                )
                try:
                    if hasattr(trial_obj, "set_user_attr"):
                        trial_obj.set_user_attr(
                            AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                            _auto_optuna_jsonable(dict(reserved_out or {})),
                        )
                    study.tell(trial_obj, state=fail_state)
                except Exception:
                    pass
                continue
            if params_sig in known_records:
                duplicate_skips += 1
                duplicate_replays += 1
                _reuse_duplicate_trial(trial_obj, str(params_sig))
                continue
            reserved_signatures.add(str(params_sig))
            return trial_obj, preset, str(params_sig), None
        return None, {}, "", str(last_error or "no unique optuna candidate available")

    idx_next = 1
    seed_items = list(seed_presets or [])[: int(total)]
    if callable(seed_to_params) and hasattr(study, "enqueue_trial"):
        seed_items_filtered = []
        enqueued_signatures: set[str] = set()
        for preset in list(seed_items):
            try:
                params = dict(seed_to_params(dict(preset or {})) or {})
            except Exception:
                params = {}
            params_sig = _auto_optuna_param_signature(params)
            if bool(duplicate_guard) and params_sig and (
                params_sig in known_records or params_sig in enqueued_signatures
            ):
                duplicate_skips += 1
                continue
            if params:
                try:
                    study.enqueue_trial(dict(params))
                    if params_sig:
                        enqueued_signatures.add(str(params_sig))
                except Exception:
                    pass
            seed_items_filtered.append(dict(preset or {}))
        seed_items = list(seed_items_filtered)

    for preset in list(seed_items):
        if idx_next > total:
            return _finalize_telemetry()
        trial_obj = None
        params_sig = ""
        preset_eval = dict(preset or {})
        if callable(seed_to_params):
            trial_obj, preset_eval, params_sig, ask_error = _ask_new_trial()
            if trial_obj is None:
                out = {
                    "idx": int(idx_next),
                    "ok": False,
                    "error": str(ask_error or "no unique optuna candidate available"),
                }
                if consume_one(int(idx_next), dict(out or {})):
                    return _finalize_telemetry()
                idx_next += 1
                continue
        try:
            out = eval_one(int(idx_next), dict(preset_eval or {}))
        except Exception as exc:
            out = {
                "idx": int(idx_next),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if trial_obj is not None:
            _tell(trial_obj, out, params_sig=params_sig, source="seed")
        if consume_one(int(idx_next), dict(out or {})):
            return _finalize_telemetry()
        idx_next += 1
    if idx_next > total:
        if duplicate_skips > 0:
            logger.info(
                "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
                int(duplicate_skips),
                str(study_name or "in-memory"),
            )
        return _finalize_telemetry()

    remaining = int(total - idx_next + 1)
    if workers <= 1 or remaining <= 1:
        for idx in range(int(idx_next), int(total) + 1):
            trial_obj, preset, params_sig, ask_error = _ask_new_trial()
            if trial_obj is None:
                out = {
                    "idx": int(idx),
                    "ok": False,
                    "error": str(ask_error or "no unique optuna candidate available"),
                }
                if consume_one(int(idx), dict(out or {})):
                    break
                continue
            try:
                out = eval_one(int(idx), dict(preset))
            except Exception as exc:
                out = {
                    "idx": int(idx),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _tell(trial_obj, out, params_sig=params_sig, source="optuna")
            if consume_one(int(idx), dict(out or {})):
                break
        if duplicate_skips > 0:
            logger.info(
                "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
                int(duplicate_skips),
                str(study_name or "in-memory"),
            )
        return _finalize_telemetry()

    chunk_size = int(_auto_trial_chunk_size(workers))
    while idx_next <= total:
        chunk_items = []
        while idx_next <= total and len(chunk_items) < int(chunk_size):
            trial_obj, preset, params_sig, ask_error = _ask_new_trial()
            if trial_obj is None:
                chunk_items.append(
                    (
                        int(idx_next),
                        None,
                        {},
                        "",
                        {
                            "idx": int(idx_next),
                            "ok": False,
                            "error": str(ask_error or "no unique optuna candidate available"),
                        },
                    )
                )
                idx_next += 1
                continue
            chunk_items.append((int(idx_next), trial_obj, dict(preset), str(params_sig), None))
            idx_next += 1
        if not chunk_items:
            break

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            fut_map = {
                ex.submit(eval_one, int(idx), dict(preset)): (int(idx), trial_obj, str(params_sig))
                for idx, trial_obj, preset, params_sig, pre_out in chunk_items
                if trial_obj is not None and pre_out is None
            }
            chunk_out: dict[int, dict] = {}
            for fut in as_completed(list(fut_map.keys())):
                idx, trial_obj, params_sig = fut_map.get(fut, (0, None, ""))
                try:
                    out = fut.result()
                    if not isinstance(out, dict):
                        out = {"idx": int(idx), "ok": False, "error": "invalid worker result"}
                except Exception as exc:
                    out = {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                _tell(trial_obj, out, params_sig=params_sig, source="optuna")
                chunk_out[int(idx)] = dict(out or {})

        for idx, _trial_obj, _preset, _params_sig, pre_out in chunk_items:
            if isinstance(pre_out, dict):
                out = dict(pre_out or {})
            else:
                out = dict(
                    chunk_out.get(
                        int(idx),
                        {"idx": int(idx), "ok": False, "error": "missing worker result"},
                    )
                    or {}
                )
            if consume_one(int(idx), out):
                if duplicate_skips > 0:
                    logger.info(
                        "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
                        int(duplicate_skips),
                        str(study_name or "in-memory"),
                    )
                return _finalize_telemetry()
    if duplicate_skips > 0:
        logger.info(
            "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
            int(duplicate_skips),
            str(study_name or "in-memory"),
        )
    return _finalize_telemetry()


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


# --------------------------------------------------------------------
# Auto-mode target preselection
# --------------------------------------------------------------------

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
    cfg = AutoModeConfig.from_base_data(base_data)
    program_version = _auto_program_version(base_data)
    filter_key = _auto_filter_cache_key(base_data)
    rank_basis = _auto_goal_basis_text(goal)
    optimizer_backend = _auto_optimizer_backend(
        base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled),
    )
    optuna_mod = _auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode target select: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
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
    target_study_sig = _auto_signature(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=None,
        include_hc_mode=False,
    )

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
            if _cache_target_valid(cached_hc):
                cached_target_hc = str(cached_hc)
                cached_target_preset = _auto_cache_get_best(
                    sig_target,
                    filter_key=filter_key,
                    program_version=program_version,
                ) or {}
                cached_target_source = "cache_signature"
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

    if (
        str(cached_target_source) == "cache_signature"
        and _cache_target_valid(cached_target_hc)
        and isinstance(cached_target_preset, dict)
        and bool(cached_target_preset)
    ):
        fallback = _cached_target_return(
            cached_target_hc,
            cached_target_preset,
            selection_method="cache_signature_hit",
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: exact cache hit for same measurements + settings, "
                "using cached target=%s and skipping all target comparison trials",
                str(cached_target_hc),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target loaded directly from cache "
                    f"(same measurements + settings -> {str(cached_target_hc)}, "
                    "skipping target comparison trials)"
                )
            return dict(fallback)

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
        phase_kind: str | None = None,
        n_total_override: int | None = None,
        seed_presets: list[dict] | None = None,
        optuna_builder=None,
        seed_to_params=None,
    ) -> list[dict]:
        use_optuna_trials = bool(
            str(optimizer_backend) == "optuna"
            and _auto_optuna_module_ready(optuna_mod)
            and callable(optuna_builder)
        )
        n_total = int(n_total_override) if n_total_override is not None else int(len(cands))
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

        if bool(use_optuna_trials):
            out_by_idx: dict[int, dict] = {}
            raw_scope = f"target-{str(target_name)}-{str(phase_tag)}"
            scope_eff = _auto_optuna_effective_scope(base_tc, raw_scope, phase_kind=phase_kind)
            study_name = _auto_optuna_study_name(
                study_sig=target_study_sig,
                scope=scope_eff,
            )

            def _eval_one(idx: int, preset: dict) -> dict:
                out = _target_eval_one(
                    dict(preset or {}),
                    base_tc=base_tc,
                    hc_f_arr=hc_f_arr,
                    hc_m_arr=hc_m_arr,
                )
                out = dict(out or {})
                out["idx"] = int(idx)
                return out

            def _consume_one(idx: int, out: dict) -> bool:
                out_by_idx[int(idx)] = dict(out or {})
                return False

            target_tel = _auto_run_optuna_eval_loop(
                optuna_mod=optuna_mod,
                cfg=cfg,
                n_total=int(n_total),
                seed=int(seed_target + sum(ord(ch) for ch in str(target_name)) * 31 + sum(ord(ch) for ch in str(phase_tag)) * 17),
                base_data=dict(base_tc or {}),
                seed_presets=list(seed_presets or []),
                build_preset=optuna_builder,
                eval_one=_eval_one,
                consume_one=_consume_one,
                objective_value=lambda out: _auto_optuna_objective_value(
                    dict((out or {}).get("metrics", {}) or {}),
                    use_refine_tiebreak=False,
                ),
                workers=int(workers),
                seed_to_params=seed_to_params,
                study_name=study_name,
                study_scope=raw_scope,
                phase_label=f"target {str(target_name)} {str(phase_tag)}",
                phase_kind=phase_kind,
            )
            _ = dict(target_tel or {})
            out = []
            for idx in range(1, int(n_total) + 1):
                out.append(
                    dict(
                        out_by_idx.get(
                            int(idx),
                            {"idx": int(idx), "ok": False, "error": "missing worker result"},
                        )
                        or {}
                    )
                )
            return out

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
        use_optuna_curve_trials = bool(
            str(optimizer_backend) == "optuna" and _auto_optuna_module_ready(optuna_mod)
        )
        candidates = []
        phase1_seed_presets = _build_auto_mode_candidates(
            base_tc,
            n_trials=1,
            seed=seed_tc,
            optimize_mag_low=False,
        )
        if not bool(use_optuna_curve_trials):
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
        curve_scored = []
        phase1_trial_total = int(max(1, trials_eff if bool(use_optuna_curve_trials) else len(candidates)))
        trials_total_count = int(phase1_trial_total)
        cb = status_cb if bool(emit_status) else None

        phase1_out = _run_target_trials(
            candidates,
            base_tc=base_tc,
            hc_f_arr=hc_f,
            hc_m_arr=hc_m,
            phase_tag="phase1",
            target_name=hc_name,
            phase_kind="target",
            n_total_override=int(phase1_trial_total),
            seed_presets=list(phase1_seed_presets or []),
            optuna_builder=(
                (lambda tr, _base_tc=dict(base_tc): _suggest_auto_mode_candidate_optuna(
                    _base_tc,
                    tr,
                    optimize_mag_low=False,
                ))
                if bool(use_optuna_curve_trials)
                else None
            ),
            seed_to_params=(
                (lambda preset, _base_tc=dict(base_tc): _seed_auto_mode_candidate_optuna_params(
                    _base_tc,
                    preset,
                    optimize_mag_low=False,
                ))
                if bool(use_optuna_curve_trials)
                else None
            ),
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
                curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
                if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                    best_metrics = dict(met)
                    best_preset = dict(trial_preset)
                    improved = True
            else:
                logger.warning(
                    f"Automatic mode target trial failed: target={hc_name} "
                    f"{c_idx}/{int(phase1_trial_total)} ({str(out.get('error', 'unknown error') or 'unknown error')})"
                )

            if callable(cb) and bool(improved):
                rank_now = _auto_safe_float((best_metrics or {}).get("rank_score"), 0.0)
                avg_now = _auto_safe_float((best_metrics or {}).get("avg_score"), 0.0)
                cb(
                    "CamillaFIR automatic mode: target trials best improved "
                    f"(target {t_idx}/{len(shortlisted)} {hc_name}, "
                    f"trial {c_idx}/{int(phase1_trial_total)}{f6_txt}, goal {goal}, "
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
            phase1_best = dict(_auto_select_best_scored(top_list) or top_list[0])
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

                local_trial_total = int(AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
                local_shrink = float(
                    _auto_adaptive_shrink_factor(
                        top_list,
                        base_shrink=float(AUTO_MODE_LOCAL_REFINEMENT_SHRINK),
                        plateau_hit=False,
                    )
                )

                def _target_local_candidate_clip(cand_in: dict) -> dict:
                    cand = dict(cand_in or {})
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
                    return dict(cand)

                local_seed_presets = [
                    _target_local_candidate_clip(c)
                    for c in _build_auto_mode_candidates_local(
                        base_tc,
                        center,
                        1,
                        int(seed_tc + li * 100003),
                        shrink=float(local_shrink),
                        optimize_mag_low=False,
                    )
                ]
                local_candidates = []
                if not bool(use_optuna_curve_trials):
                    local_candidates = [
                        _target_local_candidate_clip(c)
                        for c in _build_auto_mode_candidates_local(
                            base_tc,
                            center,
                            int(local_trial_total),
                            int(seed_tc + li * 100003),
                            shrink=float(local_shrink),
                            optimize_mag_low=False,
                        )
                    ]
                trials_total_count += int(local_trial_total)
                local_out = _run_target_trials(
                    local_candidates,
                    base_tc=base_tc,
                    hc_f_arr=hc_f,
                    hc_m_arr=hc_m,
                    phase_tag=_auto_optuna_scope_with_context(
                        f"local_center_{li}_u1",
                        center=dict(center or {}),
                        shrink=float(local_shrink),
                        extra={
                            "filter_key": str(filter_key),
                            "target_name": str(hc_name),
                        },
                    ),
                    target_name=hc_name,
                    phase_kind="local",
                    n_total_override=int(local_trial_total),
                    seed_presets=list(local_seed_presets or []),
                    optuna_builder=(
                        (
                            lambda tr,
                            _base_tc=dict(base_tc),
                            _center=dict(center),
                            _shrink=float(local_shrink): _target_local_candidate_clip(
                                _suggest_auto_mode_candidate_local_optuna(
                                    _base_tc,
                                    _center,
                                    tr,
                                    shrink=float(_shrink),
                                    optimize_mag_low=False,
                                )
                            )
                        )
                        if bool(use_optuna_curve_trials)
                        else None
                    ),
                    seed_to_params=(
                        (
                            lambda preset,
                            _base_tc=dict(base_tc),
                            _center=dict(center),
                            _shrink=float(local_shrink): _seed_auto_mode_candidate_local_optuna_params(
                                _base_tc,
                                _center,
                                preset,
                                shrink=float(_shrink),
                                optimize_mag_low=False,
                            )
                        )
                        if bool(use_optuna_curve_trials)
                        else None
                    ),
                )
                for lc_idx, out in enumerate(local_out, start=1):
                    if bool(out.get("ok", False)):
                        met = dict(out.get("metrics", {}) or {})
                        trial_preset = dict(out.get("preset", {}) or {})
                        ok_n += 1
                        rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                        avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
                        curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
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
                            f"center={li} {lc_idx}/{int(local_trial_total)} "
                            f"({str(out.get('error', 'unknown error') or 'unknown error')})"
                        )

        if ok_n <= 0 or not isinstance(best_metrics, dict):
            return None

        final_best = _auto_select_best_scored(curve_scored)
        if isinstance(final_best, dict):
            best_metrics = dict(final_best.get("metrics", {}) or {})
            best_preset = dict(final_best.get("preset", {}) or {})

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
        return _auto_target_result_rank_key(item)

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
    evaluated = sorted(evaluated, key=_auto_target_result_rank_key)
    rank_tie_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_BEST_RANK_TIE_EPS, 0.05)))
    target_scored = [
        {
            **dict(it or {}),
            "_auto_select_kind": "target_curve",
            "_target_rank_tie_eps": float(rank_tie_eps),
        }
        for it in evaluated
    ]
    rank_winner = _auto_select_best_scored(target_scored)
    winner = dict(rank_winner or evaluated[0])
    selection_method = str(winner.pop("_auto_selection_method", "top3x10_trials") or "top3x10_trials")
    if selection_method == "top3x10_trials_rank_tie_composite":
        old_winner = dict(evaluated[0])
        logger.info(
            "Automatic mode target select: rank tie-break by avg/mode/boost "
            f"(eps={rank_tie_eps:.3f}) "
            f"{str(old_winner.get('hc_mode', 'n/a'))} -> {str(winner.get('hc_mode', 'n/a'))}, "
            f"avg_rank={_auto_safe_float(old_winner.get('avg_rank_score'), 0.0):.3f}"
            f" -> {_auto_safe_float(winner.get('avg_rank_score'), 0.0):.3f}, "
            f"mode_ripple={_auto_target_result_mode_ripple(old_winner):.4f}"
            f" -> {_auto_target_result_mode_ripple(winner):.4f}, "
            f"boost_penalty={_auto_safe_float(old_winner.get('boost_penalty', 0.0), 0.0):.3f}"
            f" -> {_auto_safe_float(winner.get('boost_penalty', 0.0), 0.0):.3f}"
        )
    if bool(cache_wildcard_participated) and bool(winner.get("from_cache_wildcard", False)):
        selection_method = "trial_with_cache_wildcard"

    winner_mode_ripple = _auto_target_result_mode_ripple(winner)
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

# --------------------------------------------------------------------
# Auto-mode scoring + ranking helpers
# --------------------------------------------------------------------

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
        and (float(auto_exc_hz_now) + 1e-6) >= float(auto_exc_zero_penalty_hz)
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

# --------------------------------------------------------------------
# Auto-mode search state + orchestration
# --------------------------------------------------------------------


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
    cache_base_data = dict(base_data or {})
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
    optuna_mod = _auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
    def _cache_ready_preset(
        preset: dict | None,
        *,
        best_metrics: dict | None = None,
    ) -> dict:
        out = dict(preset or {})
        auto_exc_hz = _auto_safe_float(
            out.get(
                "_auto_exc_freq_hz",
                out.get(
                    "best_auto_exc_freq_hz",
                    out.get(
                        "exc_freq",
                        dict(best_metrics or {}).get("auto_exc_zero_penalty_hz", float("nan")),
                    ),
                ),
            ),
            float("nan"),
        )
        if np.isfinite(auto_exc_hz):
            auto_exc_hz = float(
                np.clip(
                    float(auto_exc_hz),
                    float(_auto_safe_float(cfg.exc_min_hz, AUTO_MODE_EXC_MIN_HZ)),
                    float(_auto_safe_float(cfg.exc_max_hz, AUTO_MODE_EXC_MAX_HZ)),
                )
            )
            auto_exc_hz = float(round(auto_exc_hz, 1))
            out["_auto_exc_freq_hz"] = float(auto_exc_hz)
            out["best_auto_exc_freq_hz"] = float(auto_exc_hz)
            out["exc_freq"] = float(auto_exc_hz)
        return dict(out)

    def _materialize_preset_result(
        preset: dict | None,
        *,
        include_response_arrays: bool,
        summarize: bool,
    ) -> tuple[object, dict, dict]:
        ready_preset = _cache_ready_preset(
            preset,
            best_metrics=exact_cached_metrics if isinstance(exact_cached_metrics, dict) else None,
        )
        final_data = dict(cache_base_data or {})
        final_data.update(dict(ready_preset or {}))
        if str(filter_key) in ("linear", "asym"):
            final_data["phase_limit"] = round(
                float(
                    _auto_phase_limit_clip(
                        final_data.get("phase_limit", cache_base_data.get("phase_limit", 400.0)),
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

        result = run_pipeline(
            cfg_final,
            final_measurements,
            include_response_arrays=bool(include_response_arrays),
        )
        if bool(summarize):
            result.metrics["summary"] = summarize_run(result)
        metrics = _auto_score_result(
            result,
            auto_exc_freq_hz=_auto_safe_float(
                final_data.get("_auto_exc_freq_hz", float("nan")),
                float("nan"),
            ),
            base_data=final_data,
        )
        return result, dict(metrics or {}), dict(final_data or {})

    def _save_cached_best(
        *,
        best_preset: dict,
        best_metrics: dict | None,
        best_hc_mode: str | None,
    ) -> None:
        if not bool(cfg.cache_enabled):
            return
        best_hc_mode_builtin = _auto_builtin_target_name(best_hc_mode)
        measurement_sig = _auto_measurement_signature(measurements)
        sig = _auto_signature(
            base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=best_hc_mode,
            include_hc_mode=True,
        )
        sig_target = _auto_signature(
            base_data=cache_base_data,
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
            best_preset=dict(best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=best_hc_mode,
            measurement_sig=measurement_sig,
            goal=goal,
            filter_key=filter_key,
            program_version=program_version,
        )
        _auto_cache_put_best(
            sig_target,
            best_preset=dict(best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=best_hc_mode_builtin,
            measurement_sig=measurement_sig,
            goal=goal,
            filter_key=filter_key,
            program_version=program_version,
        )
        _auto_cache_put_target_for_measurements(
            measurements=measurements,
            best_hc_mode=best_hc_mode_builtin,
            best_preset=dict(best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            goal=goal,
            filter_key=filter_key,
            program_version=program_version,
        )
        _auto_cache_put_last_used_best(
            best_preset=dict(best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=best_hc_mode,
            measurement_sig=measurement_sig,
            goal=goal,
            filter_key=filter_key,
            program_version=program_version,
        )

    exact_cached_preset = {}
    exact_cached_metrics = {}
    exact_cache_sig = None
    seed = int(20260302 + int(fs_v) * 17 + int(taps_v))
    optuna_search_sig = _auto_signature(
        base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
        include_hc_mode=True,
    )
    if bool(cfg.cache_enabled):
        try:
            exact_cache_sig = _auto_signature(
                base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
                include_hc_mode=True,
            )
            exact_cached_entry = _auto_cache_get_entry(
                exact_cache_sig,
                filter_key=filter_key,
                program_version=program_version,
            ) or {}
            exact_cached_preset = _auto_cache_get_best(
                exact_cache_sig,
                filter_key=filter_key,
                program_version=program_version,
            ) or {}
            exact_cached_metrics = dict((exact_cached_entry or {}).get("best_metrics", {}) or {})
        except Exception:
            exact_cached_preset = {}
            exact_cached_metrics = {}

    if isinstance(exact_cached_preset, dict) and exact_cached_preset:
        try:
            cache_target_name = str(cache_base_data.get("hc_mode", "n/a") or "n/a").strip() or "n/a"
            exact_cached_preset = _cache_ready_preset(
                exact_cached_preset,
                best_metrics=exact_cached_metrics,
            )
            logger.info(
                "Automatic mode: exact preset cache hit for same measurements + settings, "
                "using cached target=%s and running up to %d x %d extra micro-trials around cached winner.",
                cache_target_name,
                int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS),
                int(AUTO_MODE_CACHE_REFINE_MICRO_TRIALS),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: preset loaded from cache "
                    f"(same measurements + settings, target {cache_target_name}, "
                    f"running up to {int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)} x "
                    f"{int(AUTO_MODE_CACHE_REFINE_MICRO_TRIALS)} extra micro-trials)"
                )
            best_result = None
            best_preset = dict(exact_cached_preset or {})
            if isinstance(exact_cached_metrics, dict) and exact_cached_metrics:
                best_metrics = dict(exact_cached_metrics or {})
            else:
                best_result, best_metrics, _best_data = _materialize_preset_result(
                    best_preset,
                    include_response_arrays=False,
                    summarize=False,
                )

            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: cache refine init "
                    f"(rounds up to {int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)}, "
                    f"{int(AUTO_MODE_CACHE_REFINE_MICRO_TRIALS)} trials/round, "
                    f"min rank improvement {float(AUTO_MODE_CACHE_REFINE_MIN_RANK_IMPROVEMENT):.2f})"
                )
            micro_trials = int(max(1, AUTO_MODE_CACHE_REFINE_MICRO_TRIALS))
            improved_any = False
            improved_count_total = 0
            executed_micro_trials_total = 0
            initial_best_preset = dict(best_preset or {})
            rounds_executed = 0
            stop_reason = "max_rounds"
            cache_refine_optuna_tels = []
            min_round_improvement = float(max(0.0, _auto_safe_float(AUTO_MODE_CACHE_REFINE_MIN_RANK_IMPROVEMENT, 0.02)))
            for round_idx in range(1, int(max(1, AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)) + 1):
                rounds_executed = int(round_idx)
                round_start_metrics = dict(best_metrics or {})
                round_start_rank = _auto_safe_float(round_start_metrics.get("rank_score"), 0.0)
                round_start_preset = dict(best_preset or {})
                round_improved_count = 0
                round_executed = 0
                round_tel = {}
                raw_scope = _auto_optuna_scope_with_context(
                    "phase3-micro-u1-cache",
                    center=dict(round_start_preset or {}),
                    shrink=1.0,
                    extra={
                        "filter_key": str(filter_key),
                        "round": int(round_idx),
                    },
                )
                round_seed_presets = _build_auto_mode_candidates_micro(
                    cache_base_data,
                    dict(round_start_preset or {}),
                    n_trials=1,
                    shrink=1.0,
                )
                if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
                    _auto_optuna_remember_result(
                        optuna_mod,
                        base_data=dict(cache_base_data or {}),
                        study_name=_auto_optuna_study_name(
                            study_sig=optuna_search_sig,
                            scope=_auto_optuna_effective_scope(cache_base_data, raw_scope, phase_kind="micro"),
                        ),
                        study_scope=raw_scope,
                        phase_kind="micro",
                        seed=int(seed + 700000 + round_idx * 1009),
                        preset=dict(round_start_preset or {}),
                        metrics=dict(round_start_metrics or {}),
                        seed_to_params=(
                            lambda preset,
                            _base=dict(cache_base_data),
                            _center=dict(round_start_preset or {}),
                            _shrink=1.0: _seed_auto_mode_candidate_micro_optuna_params(
                                _base,
                                _center,
                                preset,
                                shrink=float(_shrink),
                            )
                        ),
                        use_refine_tiebreak=True,
                        out_payload={
                            "idx": 0,
                            "ok": True,
                            "metrics": dict(round_start_metrics or {}),
                            "trial_preset": dict(round_start_preset or {}),
                            "phase": "exact_cache_micro_refine_seed",
                            "round": int(round_idx),
                        },
                    )
                if bool(str(optimizer_backend) == "optuna" and _auto_optuna_module_ready(optuna_mod)):
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: cache refine round "
                            f"{int(round_idx)}/{int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)} "
                            f"(optuna {int(micro_trials)} trials)"
                        )

                    def _cache_eval_one(idx: int, preset: dict) -> dict:
                        _res_i, met_i, _data_i = _materialize_preset_result(
                            preset,
                            include_response_arrays=False,
                            summarize=False,
                        )
                        return {
                            "idx": int(idx),
                            "ok": True,
                            "metrics": dict(met_i or {}),
                            "trial_preset": dict(preset or {}),
                            "phase": "exact_cache_micro_refine",
                            "round": int(round_idx),
                        }

                    def _cache_consume_one(idx: int, out: dict) -> bool:
                        nonlocal best_metrics, best_preset, improved_any, improved_count_total, round_improved_count, round_executed, executed_micro_trials_total
                        round_executed += 1
                        executed_micro_trials_total += 1
                        if not bool(dict(out or {}).get("ok", False)):
                            err_txt = str((out or {}).get("error", "unknown error") or "unknown error")
                            logger.warning(
                                "Automatic mode cache refine round %d trial %d/%d failed: %s",
                                int(round_idx),
                                int(idx),
                                int(micro_trials),
                                str(err_txt),
                            )
                            return False

                        met_i = dict((out or {}).get("metrics", {}) or {})
                        cand = dict((out or {}).get("trial_preset", {}) or {})
                        better, _reason = _auto_is_better_refine(
                            dict(met_i or {}),
                            dict(best_metrics or {}),
                            goal,
                            return_reason=True,
                        )
                        if not better:
                            return False
                        prev_best = dict(best_metrics or {})
                        best_metrics = dict(met_i or {})
                        best_preset = _cache_ready_preset(dict(cand or {}), best_metrics=best_metrics)
                        improved_any = True
                        improved_count_total += 1
                        round_improved_count += 1
                        logger.info(
                            "Automatic mode cache refine improved: round %d trial %d/%d, "
                            "rank_score %.3f -> %.3f, avg_score %.3f -> %.3f",
                            int(round_idx),
                            int(idx),
                            int(micro_trials),
                            _auto_safe_float(prev_best.get("rank_score"), 0.0),
                            _auto_safe_float(best_metrics.get("rank_score"), 0.0),
                            _auto_safe_float(prev_best.get("avg_score"), 0.0),
                            _auto_safe_float(best_metrics.get("avg_score"), 0.0),
                        )
                        if callable(status_cb):
                            status_cb(
                                "CamillaFIR automatic mode: cache refine best improved "
                                f"(round {int(round_idx)}, {int(idx)}/{int(micro_trials)}, "
                                f"rank {_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}, "
                                f"avg {_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f})"
                            )
                        return False

                    round_tel = dict(
                        _auto_run_optuna_eval_loop(
                            optuna_mod=optuna_mod,
                            cfg=cfg,
                            n_total=int(micro_trials),
                            seed=int(seed + 700000 + round_idx * 1009),
                            base_data=dict(cache_base_data or {}),
                            seed_presets=list(round_seed_presets or []),
                            build_preset=(
                                lambda tr,
                                _base=dict(cache_base_data),
                                _center=dict(round_start_preset or {}),
                                _shrink=1.0: _suggest_auto_mode_candidate_micro_optuna(
                                    _base,
                                    _center,
                                    tr,
                                    shrink=float(_shrink),
                                )
                            ),
                            eval_one=_cache_eval_one,
                            consume_one=_cache_consume_one,
                            objective_value=lambda out: _auto_optuna_objective_value(
                                dict((out or {}).get("metrics", {}) or {}),
                                use_refine_tiebreak=True,
                            ),
                            workers=int(_auto_trial_workers(cache_base_data, int(micro_trials))),
                            seed_to_params=(
                                lambda preset,
                                _base=dict(cache_base_data),
                                _center=dict(round_start_preset or {}),
                                _shrink=1.0: _seed_auto_mode_candidate_micro_optuna_params(
                                    _base,
                                    _center,
                                    preset,
                                    shrink=float(_shrink),
                                )
                            ),
                            study_name=_auto_optuna_study_name(
                                study_sig=optuna_search_sig,
                                scope=_auto_optuna_effective_scope(cache_base_data, raw_scope, phase_kind="micro"),
                            ),
                            study_scope=raw_scope,
                            phase_label=f"cache refine round {int(round_idx)}/{int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)}",
                            phase_kind="micro",
                        )
                        or {}
                    )
                    if round_tel:
                        cache_refine_optuna_tels.append(dict(round_tel))
                else:
                    micro_candidates = _build_auto_mode_candidates_micro(
                        cache_base_data,
                        dict(best_preset or {}),
                        n_trials=int(micro_trials + 1),
                        shrink=1.0,
                    )
                    micro_candidates = [
                        dict(cand or {})
                        for cand in list(micro_candidates or [])
                        if isinstance(cand, dict) and dict(cand or {}) != dict(best_preset or {})
                    ][: int(micro_trials)]
                    if len(micro_candidates) < int(micro_trials):
                        logger.info(
                            "Automatic mode cache refine round %d: generated %d/%d micro candidates.",
                            int(round_idx),
                            int(len(micro_candidates)),
                            int(micro_trials),
                        )
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: cache refine round "
                            f"{int(round_idx)}/{int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)} "
                            f"({int(len(micro_candidates))}/{int(micro_trials)} candidates)"
                        )

                    for idx, cand in enumerate(micro_candidates, start=1):
                        _res_i, met_i, _data_i = _materialize_preset_result(
                            cand,
                            include_response_arrays=False,
                            summarize=False,
                        )
                        if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
                            _auto_optuna_remember_result(
                                optuna_mod,
                                base_data=dict(cache_base_data or {}),
                                study_name=_auto_optuna_study_name(
                                    study_sig=optuna_search_sig,
                                    scope=_auto_optuna_effective_scope(cache_base_data, raw_scope, phase_kind="micro"),
                                ),
                                study_scope=raw_scope,
                                phase_kind="micro",
                                seed=int(seed + 700000 + round_idx * 1009 + idx),
                                preset=dict(cand or {}),
                                metrics=dict(met_i or {}),
                                seed_to_params=(
                                    lambda preset,
                                    _base=dict(cache_base_data),
                                    _center=dict(round_start_preset or {}),
                                    _shrink=1.0: _seed_auto_mode_candidate_micro_optuna_params(
                                        _base,
                                        _center,
                                        preset,
                                        shrink=float(_shrink),
                                    )
                                ),
                                use_refine_tiebreak=True,
                                out_payload={
                                    "idx": int(idx),
                                    "ok": True,
                                    "metrics": dict(met_i or {}),
                                    "trial_preset": dict(cand or {}),
                                    "phase": "exact_cache_micro_refine",
                                    "round": int(round_idx),
                                },
                            )
                        round_executed += 1
                        executed_micro_trials_total += 1
                        better, _reason = _auto_is_better_refine(
                            dict(met_i or {}),
                            dict(best_metrics or {}),
                            goal,
                            return_reason=True,
                        )
                        if better:
                            prev_best = dict(best_metrics or {})
                            best_metrics = dict(met_i or {})
                            best_preset = _cache_ready_preset(dict(cand or {}), best_metrics=best_metrics)
                            improved_any = True
                            improved_count_total += 1
                            round_improved_count += 1
                            logger.info(
                                "Automatic mode cache refine improved: round %d trial %d/%d, "
                                "rank_score %.3f -> %.3f, avg_score %.3f -> %.3f",
                                int(round_idx),
                                int(idx),
                                int(len(micro_candidates)),
                                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                                _auto_safe_float(best_metrics.get("rank_score"), 0.0),
                                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                                _auto_safe_float(best_metrics.get("avg_score"), 0.0),
                            )
                            if callable(status_cb):
                                status_cb(
                                    "CamillaFIR automatic mode: cache refine best improved "
                                    f"(round {int(round_idx)}, {int(idx)}/{int(len(micro_candidates))}, "
                                    f"rank {_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}, "
                                    f"avg {_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f})"
                                )

                round_end_rank = _auto_safe_float(dict(best_metrics or {}).get("rank_score"), 0.0)
                round_delta = float(round_end_rank - round_start_rank)
                round_winner_changed = bool(dict(best_preset or {}) != dict(round_start_preset or {}))
                logger.info(
                    "Automatic mode cache refine round %d summary: executed %d/%d, "
                    "improvements=%d, winner_changed=%s, rank_delta=%.3f, final_rank=%.3f%s",
                    int(round_idx),
                    int(round_executed),
                    int(micro_trials),
                    int(round_improved_count),
                    str(bool(round_winner_changed)).lower(),
                    float(round_delta),
                    float(round_end_rank),
                    "" if not round_tel else f", {_auto_optuna_telemetry_text(round_tel)}",
                )
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: cache refine round summary "
                        f"(round {int(round_idx)}, executed {int(round_executed)}/{int(micro_trials)}, "
                        f"improvements {int(round_improved_count)}, delta {float(round_delta):.3f}"
                        f"{'' if not round_tel else f', {_auto_optuna_telemetry_text(round_tel)}'})"
                    )
                if round_improved_count <= 0:
                    stop_reason = "no_improvement"
                    break
                if float(round_delta) < float(min_round_improvement):
                    stop_reason = "below_threshold"
                    break

            winner_changed = bool(dict(best_preset or {}) != dict(initial_best_preset or {}))
            cache_refine_rollup_tel = _auto_optuna_telemetry_rollup(cache_refine_optuna_tels)
            logger.info(
                "Automatic mode cache refine summary: rounds=%d/%d, executed %d/%d micro-trials, "
                "improvements=%d, winner_changed=%s, stop_reason=%s, final_rank=%.3f, final_avg=%.3f",
                int(rounds_executed),
                int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS),
                int(executed_micro_trials_total),
                int(micro_trials * max(1, rounds_executed)),
                int(improved_count_total),
                str(bool(winner_changed)).lower(),
                str(stop_reason),
                _auto_safe_float(dict(best_metrics or {}).get("rank_score"), 0.0),
                _auto_safe_float(dict(best_metrics or {}).get("avg_score"), 0.0),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: cache refine summary "
                    f"(rounds {int(rounds_executed)}/{int(AUTO_MODE_CACHE_REFINE_MAX_ROUNDS)}, "
                    f"executed {int(executed_micro_trials_total)} trials, "
                    f"improvements {int(improved_count_total)}, "
                    f"winner {'changed' if bool(winner_changed) else 'unchanged'}, "
                    f"stop {str(stop_reason)})"
                )

            best_result, best_metrics_recalc, best_data = _materialize_preset_result(
                best_preset,
                include_response_arrays=True,
                summarize=True,
            )
            best_metrics = dict(best_metrics_recalc or best_metrics or {})
            best_preset = dict(best_data or best_preset or {})
            if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
                raw_scope = "phase1"
                scope_eff = _auto_optuna_effective_scope(cache_base_data, raw_scope, phase_kind="phase1")
                _auto_optuna_remember_result(
                    optuna_mod,
                    base_data=dict(cache_base_data or {}),
                    study_name=_auto_optuna_study_name(
                        study_sig=optuna_search_sig,
                        scope=scope_eff,
                    ),
                    study_scope=scope_eff,
                    phase_kind="phase1",
                    seed=int(seed + 500001),
                    preset=dict(best_preset or {}),
                    metrics=dict(best_metrics or {}),
                    seed_to_params=(
                        lambda preset,
                        _base=dict(cache_base_data): _seed_auto_mode_candidate_optuna_params(
                            _base,
                            preset,
                        )
                    ),
                    use_refine_tiebreak=False,
                    out_payload={
                        "idx": 1,
                        "ok": True,
                        "metrics": dict(best_metrics or {}),
                        "trial_preset": dict(best_preset or {}),
                        "phase": "exact_cache_replay",
                    },
                )
            cached_best_auto_exc_hz = _auto_safe_float(
                best_data.get("_auto_exc_freq_hz", best_data.get("best_auto_exc_freq_hz", float("nan"))),
                float("nan"),
            )
            _save_cached_best(
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                best_hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
            )
            return {
                "best_result": best_result,
                "best_metrics": dict(best_metrics or {}),
                "best_preset": dict(best_preset or {}),
                "winner_explanation": {
                    "summary": (
                        "Loaded exact cached preset and ran extra cache-refine micro-trials."
                        if bool(improved_any)
                        else "Loaded exact cached preset and verified it with cache-refine micro-trials."
                    ),
                    "reasons": [],
                    "deltas": {},
                    "phase_label": "exact cache hit + micro refine",
                    "target_name": str(cache_target_name),
                },
                "best_auto_exc_freq_hz": (
                    float(cached_best_auto_exc_hz)
                    if np.isfinite(cached_best_auto_exc_hz)
                    else float("nan")
                ),
                "auto_goal": str(goal),
                "selection_basis": str(rank_basis),
                "top": [],
                "trials_total": int(executed_micro_trials_total),
                "trials_ok": int(executed_micro_trials_total),
                "trials_phase1_total": 0,
                "trials_phase1_ok": 0,
                "trials_phase2_total": int(executed_micro_trials_total),
                "trials_phase2_ok": int(executed_micro_trials_total),
                "optuna_phase1_telemetry": {},
                "optuna_phase2_local_telemetry": [],
                "optuna_phase3_micro_telemetry": dict(cache_refine_rollup_tel or {}),
                "optuna_phase2_rollup_telemetry": dict(cache_refine_rollup_tel or {}),
                "phase1_plateau_hit": False,
                "phase2_plateau_hit": bool(str(stop_reason) in ("no_improvement", "below_threshold")),
                "search_fs": int(fs_v),
                "search_taps": int(taps_v),
            }
        except Exception as exc:
            logger.warning(
                "Automatic mode: exact preset cache materialization failed, "
                f"falling back to search ({type(exc).__name__}: {exc})"
            )

    try:
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
    except Exception:
        seed_preset = {}
    try:
        prior_seed_preset = dict(
            get_auto_mode_filter_seed_preset(
                search_base_data.get("filter_type", cache_base_data.get("filter_type", ""))
            )
            or {}
        )
    except Exception:
        prior_seed_preset = {}
    if seed_preset:
        search_base_data.update(seed_preset)

    # --- Auto-mode cache: load best preset for this measurement+settings signature ---
    if bool(cfg.cache_enabled) and not seed_preset:
        try:
            sig = _auto_signature(
                base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
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

    use_optuna_trials = bool(
        str(optimizer_backend) == "optuna" and _auto_optuna_module_ready(optuna_mod)
    )
    candidates = []
    if not bool(use_optuna_trials):
        candidates = _build_auto_mode_candidates(search_base_data, n_trials=int(n_trials_eff), seed=seed)
    elif int(n_trials_eff) > 0:
        logger.info(
            "Automatic mode optimizer backend: optuna "
            f"(trials={int(n_trials_eff)}, "
            f"startup={int(_auto_optuna_startup_for_phase_kind(cfg, phase_kind='phase1', total=int(n_trials_eff)))})"
        )
    try:
        target_label = str(search_base_data.get("hc_mode", "") or "").strip()
    except Exception:
        target_label = ""
    winner_target_name = str(target_label or "").strip() or None
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
    phase1_optuna_tel = {}
    phase2_local_optuna_tels = []
    phase3_micro_optuna_tel = {}

    def _eval_candidates(
        cands: list[dict],
        *,
        phase_label: str,
        phase_kind: str | None = None,
        plateau_after_no_improve: int = 0,
        use_refine_tiebreak: bool = False,
        focus_lo_hz: float | None = None,
        focus_hi_hz: float | None = None,
        n_total_override: int | None = None,
        seed_presets: list[dict] | None = None,
        optuna_builder=None,
        seed_to_params=None,
        study_scope: str | None = None,
    ) -> dict:
        phase_state = _AutoModePhaseState()
        use_optuna_phase = bool(
            str(optimizer_backend) == "optuna"
            and _auto_optuna_module_ready(optuna_mod)
            and callable(optuna_builder)
        )
        n_total = int(n_total_override) if n_total_override is not None else int(len(cands))
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
                    _auto_set_search_winner(
                        search_state,
                        metrics,
                        trial_preset,
                        prev_metrics=prev_best,
                        phase_label=phase_label,
                        target_name=winner_target_name,
                    )
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

        if bool(use_optuna_phase):
            raw_scope = str(study_scope or phase_label)
            scope_eff = _auto_optuna_effective_scope(search_base_data, raw_scope, phase_kind=phase_kind)
            study_name = _auto_optuna_study_name(
                study_sig=optuna_search_sig,
                scope=scope_eff,
            )
            phase_tel = dict(
                _auto_run_optuna_eval_loop(
                optuna_mod=optuna_mod,
                cfg=cfg,
                n_total=int(n_total),
                seed=int(seed + sum(ord(ch) for ch in str(phase_label)) * 31),
                base_data=dict(search_base_data or {}),
                seed_presets=list(seed_presets or []),
                build_preset=optuna_builder,
                eval_one=_eval_one,
                consume_one=_consume_one,
                objective_value=lambda out: _auto_optuna_objective_value(
                    dict((out or {}).get("metrics", {}) or {}),
                    use_refine_tiebreak=bool(use_refine_tiebreak),
                ),
                workers=int(workers),
                seed_to_params=seed_to_params,
                study_name=study_name,
                study_scope=raw_scope,
                phase_label=str(phase_label),
                phase_kind=phase_kind,
            )
                or {}
            )
            if _auto_optuna_needs_zero_feasible_rescue(
                base_data=search_base_data,
                phase_kind=phase_kind,
                telemetry=phase_tel,
            ):
                logger.warning(
                    "Automatic mode Optuna rescue fallback: phase=%s scope=%s rerunning without constraints",
                    str(phase_kind or ""),
                    str(raw_scope),
                )
                rescue_base_data = _auto_optuna_base_data_without_constraints(search_base_data)
                rescue_scope = f"{str(raw_scope)}-zf0"
                rescue_scope_eff = _auto_optuna_effective_scope(rescue_base_data, rescue_scope, phase_kind=phase_kind)
                rescue_tel = dict(
                    _auto_run_optuna_eval_loop(
                        optuna_mod=optuna_mod,
                        cfg=cfg,
                        n_total=int(n_total),
                        seed=int(seed + sum(ord(ch) for ch in str(phase_label)) * 31),
                        base_data=dict(rescue_base_data or {}),
                        seed_presets=list(seed_presets or []),
                        build_preset=optuna_builder,
                        eval_one=_eval_one,
                        consume_one=_consume_one,
                        objective_value=lambda out: _auto_optuna_objective_value(
                            dict((out or {}).get("metrics", {}) or {}),
                            use_refine_tiebreak=bool(use_refine_tiebreak),
                        ),
                        workers=int(workers),
                        seed_to_params=seed_to_params,
                        study_name=_auto_optuna_study_name(
                            study_sig=optuna_search_sig,
                            scope=rescue_scope_eff,
                        ),
                        study_scope=rescue_scope,
                        phase_label=f"{str(phase_label)} rescue",
                        phase_kind=phase_kind,
                    )
                    or {}
                )
                logger.info(
                    "Automatic mode Optuna rescue result [%s]: run=%d ok=%d startup=%d model=%d best=%s",
                    str(phase_label),
                    int((rescue_tel or {}).get("run_trials", 0) or 0),
                    int((rescue_tel or {}).get("complete_trials", 0) or 0),
                    int((rescue_tel or {}).get("startup_complete", 0) or 0),
                    int((rescue_tel or {}).get("model_complete", 0) or 0),
                    "n/a"
                    if (rescue_tel or {}).get("best_raw_value", None) is None
                    else f"{float(rescue_tel['best_raw_value']):.3f}",
                )
                phase_tel = {
                    **dict(phase_tel or {}),
                    "zero_feasible_fallback_used": True,
                    "fallback_reason": "zero_feasible",
                    "fallback_telemetry": dict(rescue_tel or {}),
                }
            else:
                phase_tel = {
                    **dict(phase_tel or {}),
                    "zero_feasible_fallback_used": False,
                }
            return {
                "ok": int(phase_state.ok_n),
                "tried": int(phase_state.tried_n),
                "plateau_hit": bool(phase_state.plateau_hit),
                "improved_any": bool(phase_state.improved_any),
                "optuna_telemetry": dict(phase_tel or {}),
                "optuna_zero_feasible_fallback_used": bool(phase_tel.get("zero_feasible_fallback_used", False)),
                "optuna_zero_feasible_fallback_telemetry": dict(phase_tel.get("fallback_telemetry", {}) or {}),
            }

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
            "optuna_telemetry": {},
            "optuna_zero_feasible_fallback_used": False,
            "optuna_zero_feasible_fallback_telemetry": {},
        }

    phase1_seed_presets = []
    if isinstance(prior_seed_preset, dict) and prior_seed_preset:
        phase1_seed_presets.append(dict(prior_seed_preset))
        logger.info(
            "Automatic mode: loaded built-in prior seed preset for %s filter.",
            str(filter_key),
        )
    phase1_seed_presets.extend(
        _build_auto_mode_candidates(
            search_base_data,
            n_trials=1,
            seed=int(seed),
        )
    )
    phase1_stats = _eval_candidates(
        candidates,
        phase_label="phase 1/2",
        phase_kind="phase1",
        plateau_after_no_improve=int(cfg.phase1_plateau_rounds),
        use_refine_tiebreak=False,
        n_total_override=int(n_trials_eff),
        seed_presets=list(phase1_seed_presets or []),
        optuna_builder=(
            (lambda tr, _base=dict(search_base_data): _suggest_auto_mode_candidate_optuna(_base, tr))
            if bool(use_optuna_trials)
            else None
        ),
        seed_to_params=(
            (lambda preset, _base=dict(search_base_data): _seed_auto_mode_candidate_optuna_params(_base, preset))
            if bool(use_optuna_trials)
            else None
        ),
        study_scope="phase1",
    )
    phase1_ok = int(phase1_stats.get("ok", 0) or 0)
    phase1_tried = int(phase1_stats.get("tried", 0) or 0)
    phase1_plateau_hit = bool(phase1_stats.get("plateau_hit", False))
    phase1_optuna_tel = dict(phase1_stats.get("optuna_telemetry", {}) or {})

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
        phase1_top_best = dict(_auto_select_best_scored(phase1_top) or phase1_top[0])
        p1m = dict(phase1_top_best.get("metrics", {}) or {})
        p1p = dict(phase1_top_best.get("preset", {}) or {})
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
        p1_optuna_txt = _auto_optuna_telemetry_text(phase1_optuna_tel)
        p1_status_suffix = f", {p1_optuna_txt}" if p1_optuna_txt else ""
        logger.info(
            "Automatic mode Phase1 done: "
            f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
            f"{p1_detail}{p1_status_suffix}"
        )
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: Phase1 done "
                f"rank={_auto_safe_float(p1m.get('rank_score'), 0.0):.3f}, "
                f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                f"mode_ripple={p1_mode_txt}, "
                f"boost={p1_boost_txt}, "
                f"{p1_detail}{p1_status_suffix}"
            )

    phase1_best_item = _auto_select_best_scored(phase1_top) if phase1_top else None
    phase1_best_metrics = dict((phase1_best_item or {}).get("metrics", {}) or {}) if phase1_best_item else None
    phase1_best_preset = dict((phase1_best_item or {}).get("preset", {}) or {}) if phase1_best_item else None
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
            local_candidates = []
            local_seed_presets = _build_auto_mode_candidates_local(
                search_base_data,
                center,
                1,
                int(local_seed),
                shrink=float(local_shrink),
            )
            if not bool(use_optuna_trials):
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
                phase_kind="local",
                plateau_after_no_improve=0,
                use_refine_tiebreak=True,
                focus_lo_hz=float(phase2_focus_lo) if np.isfinite(phase2_focus_lo) else None,
                focus_hi_hz=float(phase2_focus_hi) if np.isfinite(phase2_focus_hi) else None,
                n_total_override=int(cfg.local_refine_trials_per_top),
                seed_presets=list(local_seed_presets or []),
                optuna_builder=(
                    (
                        lambda tr,
                        _base=dict(search_base_data),
                        _center=dict(center),
                        _shrink=float(local_shrink): _suggest_auto_mode_candidate_local_optuna(
                            _base,
                            _center,
                            tr,
                            shrink=float(_shrink),
                        )
                    )
                    if bool(use_optuna_trials)
                    else None
                ),
                seed_to_params=(
                    (
                        lambda preset,
                        _base=dict(search_base_data),
                        _center=dict(center),
                        _shrink=float(local_shrink): _seed_auto_mode_candidate_local_optuna_params(
                            _base,
                            _center,
                            preset,
                            shrink=float(_shrink),
                        )
                    )
                    if bool(use_optuna_trials)
                    else None
                ),
                study_scope=_auto_optuna_scope_with_context(
                    f"phase2-local-center-{int(ci)}-u1",
                    center=dict(center or {}),
                    shrink=float(local_shrink),
                    extra={
                        "filter_key": str(filter_key),
                        "target": str(winner_target_name or ""),
                    },
                ),
            )
            phase2_ok += int(stats.get("ok", 0) or 0)
            phase2_tried += int(stats.get("tried", 0) or 0)
            local_tel = dict(stats.get("optuna_telemetry", {}) or {})
            if local_tel:
                phase2_local_optuna_tels.append(
                    {
                        "center_index": int(ci),
                        "phase_label": f"phase 2/2 local center#{ci}",
                        "telemetry": dict(local_tel),
                    }
                )
            local_tel_txt = _auto_optuna_telemetry_text(local_tel)
            local_rescue_suffix = ", zero-feasible fallback used" if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
            local_fallback_txt = _auto_optuna_fallback_summary_text(local_tel) if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
            local_best_metrics = dict(search_state.best_metrics or {})
            local_rank_txt = _auto_optuna_fmt_value(local_best_metrics.get("rank_score"), 3)
            local_avg_txt = _auto_optuna_fmt_value(local_best_metrics.get("avg_score"), 3)
            logger.info(
                "Automatic mode Local refine summary: center #%d, rank=%s, avg=%s%s%s",
                int(ci),
                str(local_rank_txt),
                str(local_avg_txt),
                "" if not local_tel_txt else f", {local_tel_txt}",
                str(local_rescue_suffix),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: Local refine summary "
                    f"center #{int(ci)}, rank={local_rank_txt}, avg_score={local_avg_txt}"
                    f"{'' if not local_tel_txt else f', {local_tel_txt}'}"
                    f"{local_rescue_suffix}"
                )
            if local_fallback_txt:
                logger.info(
                    "Automatic mode Local refine fallback detail: center #%d, %s",
                    int(ci),
                    str(local_fallback_txt),
                )
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: Local refine fallback "
                        f"center #{int(ci)}, {local_fallback_txt}"
                    )
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
            prev_best = dict(search_state.best_metrics or {})
            _auto_set_search_winner(
                search_state,
                phase1_best_metrics,
                phase1_best_preset or {},
                prev_metrics=prev_best,
                phase_label="phase 1 carry-forward",
                target_name=winner_target_name,
            )

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
        micro_shrink = float(
            np.clip(
                micro_shrink * 0.70,
                AUTO_MODE_ADAPTIVE_SHRINK_MIN,
                1.0,
            )
        )
        micro_center = dict(search_state.best_preset or {})
        micro_candidates = []
        micro_seed_presets = _build_auto_mode_candidates_micro(
            search_base_data,
            dict(micro_center),
            n_trials=1,
            shrink=float(micro_shrink),
        )
        if not bool(use_optuna_trials):
            micro_candidates = _build_auto_mode_candidates_micro(
                search_base_data,
                dict(micro_center),
                n_trials=int(cfg.phase3_micro_trials),
                shrink=float(micro_shrink),
            )
        logger.info(
            "Phase3 micro size: %d%s",
            int(cfg.phase3_micro_trials),
            " (optuna)" if bool(use_optuna_trials) else "",
        )
        if callable(status_cb):
            status_cb(
                f"CamillaFIR automatic mode: Phase3 micro "
                f"{int(cfg.phase3_micro_trials)} trials around current best"
            )
        before_micro = dict(search_state.best_metrics or {})
        micro_stats = _eval_candidates(
            micro_candidates,
            phase_label="phase 3/3 micro",
            phase_kind="micro",
            plateau_after_no_improve=0,
            use_refine_tiebreak=True,
            focus_lo_hz=float(phase2_focus_lo) if np.isfinite(phase2_focus_lo) else None,
            focus_hi_hz=float(phase2_focus_hi) if np.isfinite(phase2_focus_hi) else None,
            n_total_override=int(cfg.phase3_micro_trials),
            seed_presets=list(micro_seed_presets or []),
            optuna_builder=(
                (
                    lambda tr,
                    _base=dict(search_base_data),
                    _center=dict(micro_center),
                    _shrink=float(micro_shrink): _suggest_auto_mode_candidate_micro_optuna(
                        _base,
                        _center,
                        tr,
                        shrink=float(_shrink),
                    )
                )
                if bool(use_optuna_trials)
                else None
            ),
            seed_to_params=(
                (
                    lambda preset,
                    _base=dict(search_base_data),
                    _center=dict(micro_center),
                    _shrink=float(micro_shrink): _seed_auto_mode_candidate_micro_optuna_params(
                        _base,
                        _center,
                        preset,
                        shrink=float(_shrink),
                    )
                )
                if bool(use_optuna_trials)
                else None
            ),
            study_scope=_auto_optuna_scope_with_context(
                "phase3-micro-u1",
                center=dict(micro_center or {}),
                shrink=float(micro_shrink),
                extra={
                    "filter_key": str(filter_key),
                    "target": str(winner_target_name or ""),
                },
            ),
        )
        phase2_ok += int(micro_stats.get("ok", 0) or 0)
        phase2_tried += int(micro_stats.get("tried", 0) or 0)
        phase3_micro_optuna_tel = dict(micro_stats.get("optuna_telemetry", {}) or {})
        if bool(micro_stats.get("improved_any", False)):
            logger.info(
                "Automatic mode Phase3 micro improved: "
                f"avg_score {_auto_safe_float(before_micro.get('avg_score'), 0.0):.3f}"
                f" -> {_auto_safe_float((search_state.best_metrics or {}).get('avg_score'), 0.0):.3f}, "
                f"rank_score {_auto_safe_float(before_micro.get('rank_score'), 0.0):.3f}"
                f" -> {_auto_safe_float((search_state.best_metrics or {}).get('rank_score'), 0.0):.3f}"
            )
        micro_tel_txt = _auto_optuna_telemetry_text(phase3_micro_optuna_tel)
        micro_rescue_suffix = ", zero-feasible fallback used" if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        micro_fallback_txt = _auto_optuna_fallback_summary_text(phase3_micro_optuna_tel) if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        micro_best_metrics = dict(search_state.best_metrics or {})
        micro_rank_txt = _auto_optuna_fmt_value(micro_best_metrics.get("rank_score"), 3)
        micro_avg_txt = _auto_optuna_fmt_value(micro_best_metrics.get("avg_score"), 3)
        logger.info(
            "Automatic mode Phase3 micro summary: rank=%s, avg=%s%s%s",
            str(micro_rank_txt),
            str(micro_avg_txt),
            "" if not micro_tel_txt else f", {micro_tel_txt}",
            str(micro_rescue_suffix),
        )
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: Phase3 micro summary "
                f"rank={micro_rank_txt}, avg_score={micro_avg_txt}"
                f"{'' if not micro_tel_txt else f', {micro_tel_txt}'}"
                f"{micro_rescue_suffix}"
            )
        if micro_fallback_txt:
            logger.info(
                "Automatic mode Phase3 micro fallback detail: %s",
                str(micro_fallback_txt),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: Phase3 micro fallback "
                    f"{micro_fallback_txt}"
                )

    phase2_roll_items = [dict((it or {}).get("telemetry", {}) or {}) for it in phase2_local_optuna_tels]
    if phase3_micro_optuna_tel:
        phase2_roll_items.append(dict(phase3_micro_optuna_tel))
    phase2_rollup_tel = _auto_optuna_telemetry_rollup(phase2_roll_items)
    phase2_rollup_txt = _auto_optuna_telemetry_text(phase2_rollup_tel)
    if phase2_rollup_txt:
        logger.info("Automatic mode Phase2 summary: %s", str(phase2_rollup_txt))
        if callable(status_cb):
            status_cb(f"CamillaFIR automatic mode: Phase2 summary {phase2_rollup_txt}")

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
            rank_best = dict(_auto_select_best_scored(phase2_kept) or phase2_kept[0])
            pareto_pool = [
                {
                    **dict(it or {}),
                    "_auto_select_kind": "phase2_pareto",
                    "_phase2_pareto_acoustic_drop": float(
                        _auto_safe_float(
                            cfg.phase2_pareto_acoustic_drop,
                            AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
                        )
                    ),
                }
                for it in phase2_kept
            ]
            pareto_winner = _auto_select_best_scored(pareto_pool)
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
                prev_best = dict(search_state.best_metrics or {})
                _auto_set_search_winner(
                    search_state,
                    w_metrics,
                    w_preset,
                    prev_metrics=prev_best,
                    phase_label="phase 2 pareto",
                    target_name=winner_target_name,
                )
        else:
            logger.info(
                "Pareto front skipped: "
                f"phase2 kept pool too small ({int(len(phase2_kept))} < {int(pareto_min_n)})"
            )

    if search_state.best_metrics is None or not isinstance(search_state.best_preset, dict):
        return None

    # Materialize full output only once for the final winner.
    try:
        final_best_preset = dict(search_state.best_preset or {})
        final_auto_exc_hz = _auto_safe_float(
            dict(search_state.best_metrics or {}).get("auto_exc_zero_penalty_hz", float("nan")),
            float("nan"),
        )
        if np.isfinite(final_auto_exc_hz):
            final_auto_exc_hz = float(
                np.clip(
                    float(final_auto_exc_hz),
                    float(_auto_safe_float(cfg.exc_min_hz, AUTO_MODE_EXC_MIN_HZ)),
                    float(_auto_safe_float(cfg.exc_max_hz, AUTO_MODE_EXC_MAX_HZ)),
                )
            )
            final_auto_exc_hz = float(round(final_auto_exc_hz, 1))
            final_best_preset["_auto_exc_freq_hz"] = float(final_auto_exc_hz)
            final_best_preset["best_auto_exc_freq_hz"] = float(final_auto_exc_hz)
            final_best_preset["exc_freq"] = float(final_auto_exc_hz)
        final_data = dict(search_base_data or {})
        final_data.update(dict(final_best_preset or {}))
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
        search_state.best_metrics = _auto_score_result(
            search_state.best_result,
            auto_exc_freq_hz=_auto_safe_float(
                final_data.get("_auto_exc_freq_hz", float("nan")),
                float("nan"),
            ),
            base_data=final_data,
        )
        search_state.best_preset = dict(final_best_preset or {})
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
    cached_best_preset = _cache_ready_preset(
        search_state.best_preset,
        best_metrics=search_state.best_metrics,
    )

    # --- Auto-mode cache: save best preset for this signature ---
    if bool(cfg.cache_enabled):
        try:
            _save_cached_best(
                best_preset=dict(cached_best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                best_hc_mode=str(search_base_data.get("hc_mode", "") or "").strip() or None,
            )
            logger.info("Automatic mode: saved best preset to cache.")
        except Exception:
            pass

    return {
        "best_result": search_state.best_result,
        "best_metrics": dict(search_state.best_metrics),
        "best_preset": dict(cached_best_preset or {}),
        "winner_explanation": dict(search_state.winner_explanation or {}),
        "optimizer_backend": str(optimizer_backend or "builtin"),
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
        "optuna_phase1_telemetry": dict(phase1_optuna_tel or {}),
        "optuna_phase2_local_telemetry": list(phase2_local_optuna_tels or []),
        "optuna_phase3_micro_telemetry": dict(phase3_micro_optuna_tel or {}),
        "optuna_phase2_rollup_telemetry": dict(phase2_rollup_tel or {}),
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
