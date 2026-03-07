import hashlib
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("CamillaFIR")
MAX_SAFE_BOOST = 8.0
AUTO_MODE_TRIALS = 100
AUTO_MODE_REFINE_TRIALS = 50
AUTO_MODE_GOAL_DEFAULT = "balanced"
AUTO_MODE_GOAL_ROOM_SAFE = "room-safe"
AUTO_MODE_GOAL_LOW_RIPPLE = "low-ripple"
AUTO_MODE_GOAL_FLAT = "flat"
AUTO_MODE_GOAL_ACOUSTIC = "acoustic"
AUTO_MODE_GOAL_HYBRID = "hybrid"
AUTO_MODE_CACHE_ENABLED = True
AUTO_MODE_CACHE_MAX_ITEMS = 64
AUTO_MODE_CACHE_FILENAME = "camillafir_auto_mode_cache.json"
AUTO_MODE_CACHE_FILTER_KEYS = ("linear", "mixed", "minimum", "asym")
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
AUTO_MODE_TARGET_PRESELECT_BASS_SHAPE_W = 0.34
AUTO_MODE_TARGET_PRESELECT_TREBLE_SHAPE_W = 0.28
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
AUTO_MODE_PARALLEL_ENABLED = True
AUTO_MODE_PARALLEL_MIN_TRIALS = 6
AUTO_MODE_PARALLEL_MAX_WORKERS = 0
AUTO_MODE_PARALLEL_BATCH_MULTIPLIER = 2
AUTO_MODE_OPTUNA_PILOT_ENABLED = True
AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS = 24
AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS = 12
AUTO_MODE_OPTUNA_MULTIVARIATE = True
AUTO_MODE_OPTUNA_GROUP = False
AUTO_MODE_OPTUNA_CONSTANT_LIAR = True
AUTO_MODE_DUAL_MODE_ENABLED = True
AUTO_MODE_DUAL_MODE_TOP_N = 2
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
AUTO_MODE_PHASE2_HARD_GATE_ENABLED = True
AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP = 8
AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION = 0.70
AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION = 0.75
AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK = True
AUTO_MODE_ADAPTIVE_SHRINK_ENABLED = True
AUTO_MODE_ADAPTIVE_SHRINK_MIN = 0.20
AUTO_MODE_ADAPTIVE_SHRINK_MAX = 0.55
AUTO_MODE_PHASE3_MICRO_ENABLED = True
AUTO_MODE_PHASE3_MICRO_TRIALS = 12
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ = 110.0
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ = 40.0
AUTO_MODE_HYBRID_LOCAL_TOP_N = 2
AUTO_MODE_HYBRID_LOCAL_PER_ANCHOR = 10
AUTO_MODE_RANK_SCORE_GAIN = 1.0
AUTO_MODE_RANK_SCORE_BIAS = 18.7
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
AUTO_MODE_HPF_SMOOTH_OCT = 1.00
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


def _auto_builtin_target_name(hc_mode: str | None) -> str | None:
    key = str(hc_mode or "").strip().lower()
    if not key:
        return None
    return AUTO_MODE_BUILTIN_TARGET_LOOKUP.get(key)


def _auto_hash_array(a: np.ndarray, *, decimals: int = 4, max_len: int = 1200) -> str:
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


def _auto_optuna_sampler_kwargs(base_data: dict | None, *, workers: int = 1) -> dict:
    data = dict(base_data or {})
    multivariate = _auto_safe_bool(
        data.get("auto_mode_optuna_multivariate", AUTO_MODE_OPTUNA_MULTIVARIATE),
        AUTO_MODE_OPTUNA_MULTIVARIATE,
    )
    group = bool(multivariate) and _auto_safe_bool(
        data.get("auto_mode_optuna_group", AUTO_MODE_OPTUNA_GROUP),
        AUTO_MODE_OPTUNA_GROUP,
    )
    constant_liar = bool(int(max(1, _auto_safe_int(workers, 1))) > 1) and _auto_safe_bool(
        data.get("auto_mode_optuna_constant_liar", AUTO_MODE_OPTUNA_CONSTANT_LIAR),
        AUTO_MODE_OPTUNA_CONSTANT_LIAR,
    )
    return {
        "multivariate": bool(multivariate),
        "group": bool(group),
        "constant_liar": bool(constant_liar),
    }


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
    optuna_multivariate: bool = AUTO_MODE_OPTUNA_MULTIVARIATE
    optuna_group: bool = AUTO_MODE_OPTUNA_GROUP
    optuna_constant_liar: bool = AUTO_MODE_OPTUNA_CONSTANT_LIAR

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
            optuna_multivariate=_auto_safe_bool(
                data.get("auto_mode_optuna_multivariate", AUTO_MODE_OPTUNA_MULTIVARIATE),
                AUTO_MODE_OPTUNA_MULTIVARIATE,
            ),
            optuna_group=_auto_safe_bool(
                data.get("auto_mode_optuna_group", AUTO_MODE_OPTUNA_GROUP),
                AUTO_MODE_OPTUNA_GROUP,
            ),
            optuna_constant_liar=_auto_safe_bool(
                data.get("auto_mode_optuna_constant_liar", AUTO_MODE_OPTUNA_CONSTANT_LIAR),
                AUTO_MODE_OPTUNA_CONSTANT_LIAR,
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


def _m(metrics: dict | None, key: str, default=float("nan")) -> float:
    try:
        v = float((metrics or {}).get(key, default))
    except Exception:
        v = float(default)
    return float(v)
