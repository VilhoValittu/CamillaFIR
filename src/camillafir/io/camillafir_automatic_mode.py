"""Automatic-mode entrypoints and high-level orchestration for CamillaFIR.

This module coordinates auto-mode target selection, refine stages, and final
result assembly while delegating stage-specific logic to the dedicated
orchestrator modules under ``auto_mode``.
"""

import logging
import json
import os
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..common.house_curves import get_house_curve_by_name
from ..engine import build_config, run_pipeline, summarize_run
from ..app_paths import camillafir_data_dir, program_version_token
from .auto_mode.cache_signature import (
    _auto_cache_get_best,
    _auto_cache_get_best_target,
    _auto_cache_get_entry,
    _auto_cache_get_last_used_best,
    _auto_cache_get_target_for_measurements,
    _auto_compat_version,
    _auto_signature,
    get_auto_mode_cache_path,
)
from .auto_mode.candidate_generation import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _build_auto_mode_candidates_micro,
    _build_auto_mode_candidates_optuna,
    _suggest_auto_mode_candidate_optuna,
)
from .auto_mode.filter_priors import get_auto_mode_filter_seed_preset
from .auto_mode.materialize import AutoModeMaterializeContext, build_materialize_helpers
from .auto_mode.protection_seed import (
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
)
from .auto_mode.scoring_metrics import (
    _auto_score_result,
    _auto_exc_penalty_bins_from_dbg,
    _auto_exc_zero_penalty_freq_hz_from_stats,
)
from .auto_mode import orchestrator_finalize, orchestrator_refine, orchestrator_target
from .auto_mode.scoring_ranking import (
    _auto_build_winner_explanation,
    _auto_hybrid_mixed_freq_penalty,
    _auto_is_better_refine,
    _auto_rank_key,
    _auto_rank_key_goal,
    _auto_reject,
    _auto_ripple_metric_for_gate,
)
from .auto_mode.search_state import (
    _AutoModeSearchState,
    _auto_set_search_winner,
)
from .auto_mode.target_preselection import _auto_select_builtin_target_curve
from .auto_mode.shared import (
    AutoModeConfig,
    AUTO_MODE_COMPAT_VERSION,
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
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
    AUTO_MODE_OPTUNA_PRUNING_ENABLED,
    AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
    AUTO_MODE_OPTUNA_TELEMETRY,
    AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    _auto_filter_cache_key,
    _auto_goal,
    _auto_goal_basis_text,
    _auto_goal_norm,
    _auto_optimizer_backend,
    _auto_optuna_sampler_kwargs,
    _auto_phase_limit_center,
    _auto_safe_bool,
    _auto_safe_float,
    _auto_safe_int,
    _auto_trial_chunk_size,
    _auto_trial_workers,
)
from ..dsp._pruning import (
    clear_pruning_hook as _clear_pruning_hook,
    set_pruning_hook as _set_pruning_hook,
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
AUTO_MODE_PRESET_TRANSIENT_KEYS = frozenset({
    "program_version",
    "auto_mode_compat_version",
})


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
AUTO_MODE_RESIDUAL_TIEBREAK_ENABLED = True
AUTO_MODE_RESIDUAL_TIEBREAK_TOP_K = 3
AUTO_MODE_RESIDUAL_TIEBREAK_RANK_EPS = 0.35
AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_ENABLED = True
AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_OFFSETS_HZ = (-20.0, -10.0, 10.0, 20.0)
AUTO_MODE_MAG_C_MIN_WINNER_POLISH_ENABLED = True
AUTO_MODE_MAG_C_MIN_WINNER_POLISH_STEP_HZ = 1.0
AUTO_MODE_MAG_C_MIN_WINNER_POLISH_MAX_DOWN_HZ = 15.0

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


# --------------------------------------------------------------------
# Auto-mode Optuna backend + search entrypoints
# --------------------------------------------------------------------

from .auto_mode import optuna_backend as _optuna_backend

_auto_import_optuna = _optuna_backend._auto_import_optuna
_auto_optuna_module_ready = _optuna_backend._auto_optuna_module_ready
_auto_optuna_storage_filename = _optuna_backend._auto_optuna_storage_filename
_auto_optuna_storage_path = _optuna_backend._auto_optuna_storage_path
_auto_optuna_study_name = _optuna_backend._auto_optuna_study_name
_auto_optuna_create_storage = _optuna_backend._auto_optuna_create_storage
_auto_optuna_create_study = _optuna_backend._auto_optuna_create_study
_auto_optuna_cross_study_best_params = _optuna_backend._auto_optuna_cross_study_best_params
_auto_optuna_jsonable = _optuna_backend._auto_optuna_jsonable
_auto_optuna_scope_context_hash = _optuna_backend._auto_optuna_scope_context_hash
_auto_optuna_scope_with_context = _optuna_backend._auto_optuna_scope_with_context
_auto_optuna_param_signature = _optuna_backend._auto_optuna_param_signature
_auto_optuna_trial_params = _optuna_backend._auto_optuna_trial_params
_auto_optuna_trial_payload_preset = _optuna_backend._auto_optuna_trial_payload_preset
_auto_optuna_tdc_min = _optuna_backend._auto_optuna_tdc_min
_auto_optuna_trial_distributions = _optuna_backend._auto_optuna_trial_distributions
_auto_optuna_build_completed_trial = _optuna_backend._auto_optuna_build_completed_trial
_auto_optuna_startup_for_phase_kind = _optuna_backend._auto_optuna_startup_for_phase_kind
_auto_optuna_is_refine_phase_kind = _optuna_backend._auto_optuna_is_refine_phase_kind
_auto_optuna_constraint_scope_kind = _optuna_backend._auto_optuna_constraint_scope_kind
_auto_optuna_constraints_enabled_for_scope = _optuna_backend._auto_optuna_constraints_enabled_for_scope
_auto_optuna_effective_scope = _optuna_backend._auto_optuna_effective_scope
_auto_optuna_constraint_thresholds = _optuna_backend._auto_optuna_constraint_thresholds
_auto_optuna_trial_out_payload = _optuna_backend._auto_optuna_trial_out_payload
_auto_optuna_constraint_vector_from_metrics = _optuna_backend._auto_optuna_constraint_vector_from_metrics
_auto_optuna_use_events_constraint = _optuna_backend._auto_optuna_use_events_constraint
_auto_optuna_constraints_func = _optuna_backend._auto_optuna_constraints_func
_auto_optuna_run_token = _optuna_backend._auto_optuna_run_token
_auto_optuna_constraint_info_for_metrics = _optuna_backend._auto_optuna_constraint_info_for_metrics
_auto_optuna_trial_objective_value = _optuna_backend._auto_optuna_trial_objective_value
_auto_optuna_attach_out_telemetry = _optuna_backend._auto_optuna_attach_out_telemetry
_auto_optuna_base_data_without_constraints = _optuna_backend._auto_optuna_base_data_without_constraints
_auto_optuna_needs_zero_feasible_rescue = _optuna_backend._auto_optuna_needs_zero_feasible_rescue
_auto_optuna_build_run_telemetry = _optuna_backend._auto_optuna_build_run_telemetry
_auto_optuna_log_run_telemetry = _optuna_backend._auto_optuna_log_run_telemetry
_auto_optuna_fmt_value = _optuna_backend._auto_optuna_fmt_value
_auto_optuna_telemetry_text = _optuna_backend._auto_optuna_telemetry_text
_auto_optuna_telemetry_text_ex = _optuna_backend._auto_optuna_telemetry_text_ex
_auto_optuna_events_debug_text = _optuna_backend._auto_optuna_events_debug_text
_auto_optuna_fallback_summary_text = _optuna_backend._auto_optuna_fallback_summary_text
_auto_optuna_telemetry_rollup = _optuna_backend._auto_optuna_telemetry_rollup
_auto_optuna_study_records = _optuna_backend._auto_optuna_study_records
_auto_optuna_remember_result = _optuna_backend._auto_optuna_remember_result
_auto_optuna_objective_value = _optuna_backend._auto_optuna_objective_value
_auto_run_optuna_eval_loop = _optuna_backend._auto_run_optuna_eval_loop

from .auto_mode import search_entrypoints as _search_entrypoints

_build_auto_mode_orchestrator_runtime = _search_entrypoints._build_auto_mode_orchestrator_runtime
_run_auto_mode_search_impl = _search_entrypoints._run_auto_mode_search_impl
_run_auto_mode_search = _search_entrypoints._run_auto_mode_search


def _auto_trial_workers(base_data: dict | None, n_trials: int) -> int:
    """Facade wrapper kept for monkeypatch-compatible worker budgeting tests."""
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
    return orchestrator_target.select_target_curve_with_trials(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        pin_obj=pin_obj,
        status_cb=status_cb,
        top_n=int(top_n),
        trials_per_curve=int(trials_per_curve),
        runtime=_build_auto_mode_orchestrator_runtime(),
    )
