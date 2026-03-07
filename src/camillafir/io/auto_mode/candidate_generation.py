import numpy as np

from .shared import (
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_LOCAL_REFINE_SHRINK,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
    AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ,
    AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_FRAC,
    AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ,
    AUTO_MODE_PHASE_LIMIT_EXPLORE_UNIFORM_FRAC,
    AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ,
    AUTO_MODE_PHASE_LIMIT_SIGMA_HZ,
    AUTO_MODE_PHASE3_MICRO_TRIALS,
    _auto_goal,
    _auto_goal_norm,
    _auto_is_phase_search_filter,
    _auto_phase_limit_center,
    _auto_safe_float,
    _auto_sample_mag_low_pair,
    _clip,
    _jitter,
)


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

    s = float(np.clip(_auto_safe_float(shrink, 1.0), 0.25, 1.0))
    mixed_steps = [0.0, -16.0 * s, -8.0 * s, +8.0 * s, +16.0 * s]
    phase_steps = [0.0, -28.0 * s, -14.0 * s, +14.0 * s, +28.0 * s]
    tdc_steps = [0.0, -8.0 * s, -4.0 * s, +4.0 * s, +8.0 * s]
    fdw_steps = [0.0, -1.0 * s, +1.0 * s]
    reg_steps = [0.0, -6.0 * s, +6.0 * s]
    tw_steps = [0.0, -15.0 * s, +15.0 * s]
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
