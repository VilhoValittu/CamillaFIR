"""Automatic-mode search entrypoints and runtime assembly."""

from __future__ import annotations

import logging

import numpy as np

from . import api as auto_api
from . import orchestrator_finalize, orchestrator_refine
from .filter_priors import get_auto_mode_filter_seed_preset
from .materialize import AutoModeMaterializeContext, build_materialize_helpers

logger = logging.getLogger("CamillaFIR")


def _build_auto_mode_orchestrator_runtime() -> dict:
    return {
        "auto_cache_get_best": auto_api._auto_cache_get_best,
        "auto_cache_get_best_target": auto_api._auto_cache_get_best_target,
        "auto_cache_get_entry": auto_api._auto_cache_get_entry,
        "auto_cache_get_target_for_measurements": auto_api._auto_cache_get_target_for_measurements,
        "auto_score_result": auto_api._auto_score_result,
        "auto_select_builtin_target_curve": auto_api._auto_select_builtin_target_curve,
        "auto_trial_workers": auto_api._auto_trial_workers,
        "auto_optuna_base_data_without_constraints": auto_api._auto_optuna_base_data_without_constraints,
        "auto_import_optuna": auto_api._auto_import_optuna,
        "auto_optuna_effective_scope": auto_api._auto_optuna_effective_scope,
        "auto_optuna_fallback_summary_text": auto_api._auto_optuna_fallback_summary_text,
        "auto_optuna_fmt_value": auto_api._auto_optuna_fmt_value,
        "auto_optuna_module_ready": auto_api._auto_optuna_module_ready,
        "auto_optuna_needs_zero_feasible_rescue": auto_api._auto_optuna_needs_zero_feasible_rescue,
        "auto_optuna_objective_value": auto_api._auto_optuna_objective_value,
        "auto_optuna_remember_result": auto_api._auto_optuna_remember_result,
        "auto_optuna_scope_with_context": auto_api._auto_optuna_scope_with_context,
        "auto_optuna_study_name": auto_api._auto_optuna_study_name,
        "auto_optuna_telemetry_rollup": auto_api._auto_optuna_telemetry_rollup,
        "auto_optuna_telemetry_text": auto_api._auto_optuna_telemetry_text,
        "auto_run_optuna_eval_loop": auto_api._auto_run_optuna_eval_loop,
        "build_config": auto_api.build_config,
        "build_auto_mode_candidates_micro": auto_api._build_auto_mode_candidates_micro,
        "cache_refine_max_rounds": auto_api.AUTO_MODE_CACHE_REFINE_MAX_ROUNDS,
        "cache_refine_micro_trials": auto_api.AUTO_MODE_CACHE_REFINE_MICRO_TRIALS,
        "cache_refine_min_rank_improvement": auto_api.AUTO_MODE_CACHE_REFINE_MIN_RANK_IMPROVEMENT,
        "get_house_curve_by_name": auto_api.get_house_curve_by_name,
        "mag_c_min_winner_polish_enabled": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_ENABLED,
        "mag_c_min_winner_polish_max_down_hz": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_MAX_DOWN_HZ,
        "mag_c_min_winner_polish_step_hz": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_STEP_HZ,
        "phase_limit_winner_polish_enabled": auto_api.AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_ENABLED,
        "phase_limit_winner_polish_offsets_hz": auto_api.AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_OFFSETS_HZ,
        "run_pipeline": auto_api.run_pipeline,
        "suggest_auto_mode_candidate_optuna": auto_api._suggest_auto_mode_candidate_optuna,
        "summarize_run": auto_api.summarize_run,
    }

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
    n_trials: int = auto_api.AUTO_MODE_TRIALS,
) -> dict | None:
    cache_base_data = dict(base_data or {})
    search_base_data = dict(base_data or {})
    cfg = auto_api.AutoModeConfig.from_base_data(search_base_data)
    n_trials_eff = int(max(1, auto_api._auto_safe_int(n_trials, cfg.trials)))
    compat_version = auto_api._auto_compat_version(search_base_data)
    goal = auto_api._auto_goal(search_base_data)
    filter_key = auto_api._auto_filter_cache_key(search_base_data)
    rank_basis = auto_api._auto_goal_basis_text(goal)
    optimizer_backend = auto_api._auto_optimizer_backend(
        search_base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled),
    )
    optuna_mod = auto_api._auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
    exact_cached_metrics = {}
    (
        _cache_ready_preset,
        _materialize_preset_result,
        _preset_signature_ignoring_residual,
        _maybe_apply_residual_tiebreak,
    ) = build_materialize_helpers(
        AutoModeMaterializeContext(
            cfg=cfg,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin_obj=pin_obj,
            filter_key=str(filter_key),
            max_safe_boost=float(auto_api.MAX_SAFE_BOOST),
            goal=str(goal),
            status_cb=status_cb,
            exact_cached_metrics_getter=lambda: exact_cached_metrics,
            auto_score_result_fn=auto_api._auto_score_result,
            auto_optuna_jsonable_fn=auto_api._auto_optuna_jsonable,
            auto_rank_key_fn=auto_api._auto_rank_key,
            auto_is_better_refine_fn=auto_api._auto_is_better_refine,
            build_config_fn=auto_api.build_config,
            run_pipeline_fn=auto_api.run_pipeline,
            summarize_run_fn=auto_api.summarize_run,
            preset_transient_keys=tuple(auto_api.AUTO_MODE_PRESET_TRANSIENT_KEYS),
            residual_tiebreak_enabled=bool(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_ENABLED),
            residual_top_k=int(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_TOP_K),
            residual_rank_eps=float(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_RANK_EPS),
        )
    )
    seed = int(20260302 + int(fs_v) * 17 + int(taps_v))
    optuna_search_sig = auto_api._auto_signature(
        base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
        include_hc_mode=True,
    )

    exact_cache_result = orchestrator_refine.run_exact_cache_micro_refine(
        cache_base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        status_cb=status_cb,
        cfg=cfg,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=optuna_search_sig,
        _cache_ready_preset=_cache_ready_preset,
        _materialize_preset_result=_materialize_preset_result,
        runtime=_build_auto_mode_orchestrator_runtime(),
    )
    if isinstance(exact_cache_result, dict):
        exact_cache_final = orchestrator_finalize.finalize_search_result(
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin_obj=pin_obj,
            cfg=cfg,
            goal=goal,
            rank_basis=rank_basis,
            filter_key=filter_key,
            compat_version=compat_version,
            optimizer_backend=optimizer_backend,
            status_cb=status_cb,
            optuna_mod=optuna_mod,
            optuna_search_sig=optuna_search_sig,
            seed=int(seed),
            search_state=None,
            winner_target_name=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
            phase1_ok=0,
            phase2_ok=0,
            phase1_tried=0,
            phase2_tried=0,
            phase1_plateau_hit=False,
            phase2_plateau_hit=False,
            phase1_optuna_tel={},
            phase2_local_optuna_tels=[],
            phase3_micro_optuna_tel={},
            phase2_rollup_tel={},
            _cache_ready_preset=_cache_ready_preset,
            _materialize_preset_result=_materialize_preset_result,
            _maybe_apply_residual_tiebreak=_maybe_apply_residual_tiebreak,
            cache_refine_result=exact_cache_result,
            runtime=_build_auto_mode_orchestrator_runtime(),
        )
        if isinstance(exact_cache_final, dict):
            return exact_cache_final

    try:
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
    except (TypeError, ValueError):
        seed_preset = {}
    try:
        prior_seed_preset = dict(
            get_auto_mode_filter_seed_preset(
                search_base_data.get("filter_type", cache_base_data.get("filter_type", ""))
            )
            or {}
        )
    except (TypeError, ValueError):
        logger.debug("Failed to read automatic-mode prior seed preset", exc_info=True)
        prior_seed_preset = {}
    if seed_preset:
        search_base_data.update(seed_preset)

    if bool(cfg.cache_enabled) and not seed_preset:
        try:
            sig = auto_api._auto_signature(
                base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
                include_hc_mode=True,
            )
            cached_entry_sig = auto_api._auto_cache_get_entry(sig, filter_key=filter_key, compat_version=compat_version)
            cached = dict((cached_entry_sig or {}).get("best_preset", {}) or {}) if isinstance(cached_entry_sig, dict) else {}
            cached_metrics_seed = dict((cached_entry_sig or {}).get("best_metrics", {}) or {}) if isinstance(cached_entry_sig, dict) else {}
            cache_seed_source = "signature"
            if not (isinstance(cached, dict) and cached):
                cache_seed_source = "last_used"
                cached_entry_last = auto_api._auto_cache_get_last_used_best(
                    goal=goal,
                    filter_key=filter_key,
                    compat_version=compat_version,
                )
                cached = dict((cached_entry_last or {}).get("best_preset", {}) or {})
                cached_metrics_seed = dict((cached_entry_last or {}).get("best_metrics", {}) or {})
            if isinstance(cached, dict) and cached:
                search_base_data["_auto_target_seed_preset"] = dict(cached)
                search_base_data.update(dict(cached))
                if isinstance(cached_metrics_seed, dict) and cached_metrics_seed:
                    search_base_data["_auto_target_seed_metrics"] = dict(cached_metrics_seed)
                if str(cache_seed_source) == "last_used":
                    logger.info(
                        "Automatic mode: loaded filter-specific last-used preset seed."
                    )
                else:
                    logger.info("Automatic mode: loaded cached best preset seed.")
        except Exception:
            # Cache seed loading is opportunistic and must not block a fresh search.
            pass

    if str(filter_key) in ("linear", "asym"):
        prev_phase_limit = auto_api._auto_safe_float(search_base_data.get("phase_limit", float("nan")), float("nan"))
        clamped_phase_limit = round(
            float(auto_api._auto_phase_limit_center(search_base_data.get("phase_limit", None))),
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
        str(optimizer_backend) == "optuna" and auto_api._auto_optuna_module_ready(optuna_mod)
    )
    candidates = []
    if not bool(use_optuna_trials):
        candidates = auto_api._build_auto_mode_candidates(search_base_data, n_trials=int(n_trials_eff), seed=seed)
    elif int(n_trials_eff) > 0:
        logger.info(
            "Automatic mode optimizer backend: optuna "
            f"(trials={int(n_trials_eff)}, "
            f"startup={int(auto_api._auto_optuna_startup_for_phase_kind(cfg, phase_kind='phase1', total=int(n_trials_eff)))})"
        )
    try:
        target_label = str(search_base_data.get("hc_mode", "") or "").strip()
    except Exception:
        # Target labels come from user/cache data; stringify defensively for status text only.
        target_label = ""
    winner_target_name = str(target_label or "").strip() or None
    if not target_label:
        target_label = "n/a"
    f6_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_mag_c_min_hz", search_base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    low_bass_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_low_bass_cut_hz", search_base_data.get("low_bass_cut_hz", float("nan"))),
        float("nan"),
    )
    exc_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_exc_freq_hz", search_base_data.get("exc_freq", float("nan"))),
        float("nan"),
    )
    hpf_enabled = bool(search_base_data.get("hpf_enable", False))
    hpf_freq = auto_api._auto_safe_float(search_base_data.get("hpf_freq", float("nan")), float("nan"))
    hpf_slope = auto_api._auto_safe_float(search_base_data.get("hpf_slope", float("nan")), float("nan"))
    hpf_meta = dict(search_base_data.get("_auto_hpf_meta", {}) or {})
    if isinstance(hpf_meta, dict):
        hpf_meta_enabled = bool(hpf_meta.get("applied", hpf_meta.get("enabled", False)))
        if not hpf_enabled:
            hpf_enabled = bool(hpf_meta_enabled)
        if not np.isfinite(hpf_freq):
            hpf_freq = auto_api._auto_safe_float(hpf_meta.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_slope = auto_api._auto_safe_float(hpf_meta.get("slope_db_oct", float("nan")), float("nan"))
    if isinstance(hpf, dict):
        if not hpf_enabled:
            hpf_enabled = bool(hpf.get("enabled", False))
        if not np.isfinite(hpf_freq):
            hpf_freq = auto_api._auto_safe_float(hpf.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_order = auto_api._auto_safe_float(hpf.get("order", float("nan")), float("nan"))
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

    search_state = auto_api._AutoModeSearchState()
    try:
        _csm = dict(search_base_data.get("_auto_target_seed_metrics", {}) or {})
        _csp = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
        if _csm and _csp:
            auto_api._auto_set_search_winner(
                search_state,
                _csm,
                _csp,
                phase_label="cache_seed",
                target_name=winner_target_name,
            )
            logger.info(
                "Automatic mode: baseline seeded from cache (rank %.3f).",
                auto_api._auto_safe_float(_csm.get("rank_score"), 0.0),
            )
    except Exception:
        pass
    phase_stats = orchestrator_refine.run_search_refine_stages(
        search_base_data=search_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        cfg=cfg,
        goal=goal,
        filter_key=filter_key,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=optuna_search_sig,
        status_prefix=status_prefix,
        winner_target_name=winner_target_name,
        search_state=search_state,
        n_trials_eff=int(n_trials_eff),
        candidates=list(candidates or []),
        prior_seed_preset=dict(prior_seed_preset or {}),
        use_optuna_trials=bool(use_optuna_trials),
        runtime=_build_auto_mode_orchestrator_runtime(),
    )
    return orchestrator_finalize.finalize_search_result(
        search_base_data=search_base_data,
        cache_base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        cfg=cfg,
        goal=goal,
        rank_basis=rank_basis,
        filter_key=filter_key,
        compat_version=compat_version,
        optimizer_backend=optimizer_backend,
        status_cb=status_cb,
        optuna_mod=optuna_mod,
        optuna_search_sig=optuna_search_sig,
        seed=int(seed),
        search_state=search_state,
        winner_target_name=winner_target_name,
        phase1_ok=int(dict(phase_stats or {}).get('phase1_ok', 0) or 0),
        phase2_ok=int(dict(phase_stats or {}).get('phase2_ok', 0) or 0),
        phase1_tried=int(dict(phase_stats or {}).get('phase1_tried', 0) or 0),
        phase2_tried=int(dict(phase_stats or {}).get('phase2_tried', 0) or 0),
        phase1_plateau_hit=bool(dict(phase_stats or {}).get('phase1_plateau_hit', False)),
        phase2_plateau_hit=bool(dict(phase_stats or {}).get('phase2_plateau_hit', False)),
        phase1_optuna_tel=dict(dict(phase_stats or {}).get('phase1_optuna_tel', {}) or {}),
        phase2_local_optuna_tels=list(dict(phase_stats or {}).get('phase2_local_optuna_tels', []) or []),
        phase3_micro_optuna_tel=dict(dict(phase_stats or {}).get('phase3_micro_optuna_tel', {}) or {}),
        phase2_rollup_tel=dict(dict(phase_stats or {}).get('phase2_rollup_tel', {}) or {}),
        _cache_ready_preset=_cache_ready_preset,
        _materialize_preset_result=_materialize_preset_result,
        _maybe_apply_residual_tiebreak=_maybe_apply_residual_tiebreak,
        runtime=_build_auto_mode_orchestrator_runtime(),
    )


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
        n_trials: int = auto_api.AUTO_MODE_TRIALS,
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
    n_trials: int = auto_api.AUTO_MODE_TRIALS,
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
