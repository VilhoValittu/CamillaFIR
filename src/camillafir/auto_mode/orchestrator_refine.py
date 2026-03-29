"""Refine-stage orchestration for automatic mode."""

from __future__ import annotations

import logging

import numpy as np

from .cache_signature import _auto_signature
from .candidate_generation import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_micro_optuna_params,
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
    _suggest_auto_mode_candidate_micro_optuna,
)
from .refine_eval import (
    RefineEvalContext,
    build_phase2_rollup_telemetry,
    run_candidate_phase,
)
from .runtime_context import coerce_orchestrator_runtime
from .scoring_ranking import (
    _auto_adaptive_shrink_factor,
    _auto_build_refine_profile,
    _auto_goal_uses_local_refine,
    _auto_is_better_refine,
    _auto_rank_key,
    _auto_select_best_scored,
)
from .shared import (
    AUTO_MODE_ADAPTIVE_SHRINK_MIN,
    _auto_safe_float,
)

logger = logging.getLogger("CamillaFIR")
__all__ = ["run_exact_cache_micro_refine", "run_search_refine_stages"]


def run_exact_cache_micro_refine(
    *,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    _cache_ready_preset,
    _materialize_preset_result,
    runtime=None,
) -> dict | None:
    runtime = coerce_orchestrator_runtime(runtime)
    exact_cached_preset = {}
    exact_cached_metrics = {}
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
            exact_cached_entry = runtime.auto_cache_get_entry(
                exact_cache_sig,
                filter_key=filter_key,
                compat_version=compat_version,
            ) or {}
            exact_cached_preset = runtime.auto_cache_get_best(
                exact_cache_sig,
                filter_key=filter_key,
                compat_version=compat_version,
            ) or {}
            exact_cached_metrics = dict((exact_cached_entry or {}).get("best_metrics", {}) or {})
        except Exception as exc:
            # Cache reads are opportunistic; a bad entry should simply disable the fast path.
            logger.debug("Auto-mode cache read failed, disabling fast path: %s: %s", type(exc).__name__, exc)
            exact_cached_preset = {}
            exact_cached_metrics = {}

    if not (isinstance(exact_cached_preset, dict) and exact_cached_preset):
        return None
    try:
        cache_target_name = str(cache_base_data.get("hc_mode", "n/a") or "n/a").strip() or "n/a"
        exact_cached_preset = _cache_ready_preset(
            exact_cached_preset,
            best_metrics=exact_cached_metrics,
        )
        logger.info(
            "Automatic mode: exact preset cache hit for same measurements + settings, using cached target=%s and running up to %d x %d extra micro-trials around cached winner.",
            cache_target_name,
            int(runtime.cache_refine_max_rounds),
            int(runtime.cache_refine_micro_trials),
        )
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: preset loaded from cache "
                f"(same measurements + settings, target {cache_target_name}, "
                f"running up to {int(runtime.cache_refine_max_rounds)} x "
                f"{int(runtime.cache_refine_micro_trials)} extra micro-trials)"
            )
        best_preset = dict(exact_cached_preset or {})
        if isinstance(exact_cached_metrics, dict) and exact_cached_metrics:
            best_metrics = dict(exact_cached_metrics or {})
        else:
            _best_result, best_metrics, _best_data = _materialize_preset_result(
                best_preset,
                include_response_arrays=False,
                summarize=False,
            )

        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: cache refine init "
                f"(rounds up to {int(runtime.cache_refine_max_rounds)}, "
                f"{int(runtime.cache_refine_micro_trials)} trials/round, "
                f"min rank improvement {float(runtime.cache_refine_min_rank_improvement):.2f})"
            )
        micro_trials = int(max(1, runtime.cache_refine_micro_trials))
        improved_any = False
        improved_count_total = 0
        executed_micro_trials_total = 0
        initial_best_preset = dict(best_preset or {})
        rounds_executed = 0
        stop_reason = "max_rounds"
        cache_refine_optuna_tels = []
        min_round_improvement = float(
            max(0.0, _auto_safe_float(runtime.cache_refine_min_rank_improvement, 0.02))
        )
        for round_idx in range(1, int(max(1, runtime.cache_refine_max_rounds)) + 1):
            rounds_executed = int(round_idx)
            round_start_metrics = dict(best_metrics or {})
            round_start_rank = _auto_safe_float(round_start_metrics.get("rank_score"), 0.0)
            round_start_preset = dict(best_preset or {})
            round_improved_count = 0
            round_executed = 0
            round_tel = {}
            raw_scope = runtime.auto_optuna_scope_with_context(
                "phase3-micro-u1-cache",
                center=dict(round_start_preset or {}),
                shrink=1.0,
                extra={
                    "filter_key": str(filter_key),
                    "round": int(round_idx),
                },
            )
            round_seed_presets = runtime.build_auto_mode_candidates_micro(
                cache_base_data,
                dict(round_start_preset or {}),
                n_trials=1,
                shrink=1.0,
            )
            if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
                runtime.auto_optuna_remember_result(
                    optuna_mod,
                    base_data=dict(cache_base_data or {}),
                    study_name=runtime.auto_optuna_study_name(
                        study_sig=optuna_search_sig,
                        scope=runtime.auto_optuna_effective_scope(
                            cache_base_data,
                            raw_scope,
                            phase_kind="micro",
                        ),
                    ),
                    study_scope=raw_scope,
                    phase_kind="micro",
                    seed=int(seed + 700000 + round_idx * 1009),
                    preset=dict(round_start_preset or {}),
                    metrics=dict(round_start_metrics or {}),
                    seed_to_params=(
                        lambda preset,
                        _base=dict(cache_base_data),
                        _center=dict(round_start_preset or {}): _seed_auto_mode_candidate_micro_optuna_params(
                            _base,
                            _center,
                            preset,
                            shrink=1.0,
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
            if bool(str(optimizer_backend) == "optuna" and runtime.auto_optuna_module_ready(optuna_mod)):
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: cache refine round "
                        f"{int(round_idx)}/{int(runtime.cache_refine_max_rounds)} "
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
                        "Automatic mode cache refine improved: round %d trial %d/%d, rank_score %.3f -> %.3f, avg_score %.3f -> %.3f",
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
                    runtime.auto_run_optuna_eval_loop(
                        optuna_mod=optuna_mod,
                        cfg=cfg,
                        n_total=int(micro_trials),
                        seed=int(seed + 700000 + round_idx * 1009),
                        base_data=dict(cache_base_data or {}),
                        seed_presets=list(round_seed_presets or []),
                        build_preset=(
                            lambda tr,
                            _base=dict(cache_base_data),
                            _center=dict(round_start_preset or {}): _suggest_auto_mode_candidate_micro_optuna(
                                _base,
                                _center,
                                tr,
                                shrink=1.0,
                            )
                        ),
                        eval_one=_cache_eval_one,
                        consume_one=_cache_consume_one,
                        objective_value=lambda out: runtime.auto_optuna_objective_value(
                            dict((out or {}).get("metrics", {}) or {}),
                            use_refine_tiebreak=True,
                        ),
                        workers=int(runtime.auto_trial_workers(cache_base_data, int(micro_trials))),
                        seed_to_params=(
                            lambda preset,
                            _base=dict(cache_base_data),
                            _center=dict(round_start_preset or {}): _seed_auto_mode_candidate_micro_optuna_params(
                                _base,
                                _center,
                                preset,
                                shrink=1.0,
                            )
                        ),
                        study_name=runtime.auto_optuna_study_name(
                            study_sig=optuna_search_sig,
                            scope=runtime.auto_optuna_effective_scope(
                                cache_base_data,
                                raw_scope,
                                phase_kind="micro",
                            ),
                        ),
                        study_scope=raw_scope,
                        phase_label=f"cache refine round {int(round_idx)}/{int(runtime.cache_refine_max_rounds)}",
                        phase_kind="micro",
                    )
                    or {}
                )
                if round_tel:
                    cache_refine_optuna_tels.append(dict(round_tel))
            else:
                micro_candidates = runtime.build_auto_mode_candidates_micro(
                    cache_base_data,
                    dict(best_preset or {}),
                    n_trials=int(micro_trials + 1),
                    shrink=1.0,
                )
                micro_candidates = [
                    dict(cand or {})
                    for cand in list(micro_candidates or [])
                    if isinstance(cand, dict)
                    and dict(cand or {}) != dict(best_preset or {})
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
                        f"{int(round_idx)}/{int(runtime.cache_refine_max_rounds)} "
                        f"({int(len(micro_candidates))}/{int(micro_trials)} candidates)"
                    )

                for idx, cand in enumerate(micro_candidates, start=1):
                    _res_i, met_i, _data_i = _materialize_preset_result(
                        cand,
                        include_response_arrays=False,
                        summarize=False,
                    )
                    if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
                        runtime.auto_optuna_remember_result(
                            optuna_mod,
                            base_data=dict(cache_base_data or {}),
                            study_name=runtime.auto_optuna_study_name(
                                study_sig=optuna_search_sig,
                                scope=runtime.auto_optuna_effective_scope(
                                    cache_base_data,
                                    raw_scope,
                                    phase_kind="micro",
                                ),
                            ),
                            study_scope=raw_scope,
                            phase_kind="micro",
                            seed=int(seed + 700000 + round_idx * 1009 + idx),
                            preset=dict(cand or {}),
                            metrics=dict(met_i or {}),
                            seed_to_params=(
                                lambda preset,
                                _base=dict(cache_base_data),
                                _center=dict(round_start_preset or {}): _seed_auto_mode_candidate_micro_optuna_params(
                                    _base,
                                    _center,
                                    preset,
                                    shrink=1.0,
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
                        best_preset = _cache_ready_preset(
                            dict(cand or {}),
                            best_metrics=best_metrics,
                        )
                        improved_any = True
                        improved_count_total += 1
                        round_improved_count += 1
                        logger.info(
                            "Automatic mode cache refine improved: round %d trial %d/%d, rank_score %.3f -> %.3f, avg_score %.3f -> %.3f",
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
                "Automatic mode cache refine round %d summary: executed %d/%d, improvements=%d, winner_changed=%s, rank_delta=%.3f, final_rank=%.3f%s",
                int(round_idx),
                int(round_executed),
                int(micro_trials),
                int(round_improved_count),
                str(bool(round_winner_changed)).lower(),
                float(round_delta),
                float(round_end_rank),
                "" if not round_tel else f", {runtime.auto_optuna_telemetry_text(round_tel)}",
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: cache refine round summary "
                    f"(round {int(round_idx)}, executed {int(round_executed)}/{int(micro_trials)}, "
                    f"improvements {int(round_improved_count)}, delta {float(round_delta):.3f}"
                    f"{'' if not round_tel else f', {runtime.auto_optuna_telemetry_text(round_tel)}'})"
                )
            if round_improved_count <= 0:
                stop_reason = "no_improvement"
                break
            if float(round_delta) < float(min_round_improvement):
                stop_reason = "below_threshold"
                break

        winner_changed = bool(dict(best_preset or {}) != dict(initial_best_preset or {}))
        cache_refine_rollup_tel = runtime.auto_optuna_telemetry_rollup(cache_refine_optuna_tels)
        logger.info(
            "Automatic mode cache refine summary: rounds=%d/%d, executed %d/%d micro-trials, improvements=%d, winner_changed=%s, stop_reason=%s, final_rank=%.3f, final_avg=%.3f",
            int(rounds_executed),
            int(runtime.cache_refine_max_rounds),
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
                f"(rounds {int(rounds_executed)}/{int(runtime.cache_refine_max_rounds)}, "
                f"executed {int(executed_micro_trials_total)} trials, "
                f"improvements {int(improved_count_total)}, "
                f"winner {'changed' if bool(winner_changed) else 'unchanged'}, "
                f"stop {str(stop_reason)})"
            )
        return {
            "cache_target_name": str(cache_target_name),
            "best_preset": dict(best_preset or {}),
            "best_metrics": dict(best_metrics or {}),
            "improved_any": bool(improved_any),
            "improved_count_total": int(improved_count_total),
            "executed_micro_trials_total": int(executed_micro_trials_total),
            "cache_refine_rollup_tel": dict(cache_refine_rollup_tel or {}),
            "stop_reason": str(stop_reason),
        }
    except Exception as exc:
        # Cache-refine failures should fall back to the normal search path instead of aborting auto mode.
        logger.warning(
            "Automatic mode: exact preset cache materialization failed, "
            f"falling back to search ({type(exc).__name__}: {exc})"
        )
    return None


def run_search_refine_stages(
    *,
    search_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    status_prefix: str,
    winner_target_name: str | None,
    search_state,
    n_trials_eff: int,
    candidates: list[dict],
    prior_seed_preset: dict | None,
    use_optuna_trials: bool,
    runtime=None,
) -> dict:
    runtime = coerce_orchestrator_runtime(runtime)
    phase1_ok = 0
    phase2_ok = 0
    phase1_tried = 0
    phase2_tried = 0
    phase1_plateau_hit = False
    phase2_plateau_hit = False
    phase1_optuna_tel = {}
    phase2_local_optuna_tels = []
    phase3_micro_optuna_tel = {}

    ctx = RefineEvalContext(
        search_base_data=dict(search_base_data or {}),
        measurements=dict(measurements or {}),
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        cfg=cfg,
        goal=str(goal),
        filter_key=str(filter_key),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=str(optuna_search_sig),
        status_cb=status_cb,
        status_prefix=str(status_prefix),
        winner_target_name=winner_target_name,
        search_state=search_state,
        runtime=runtime,
    )

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
    phase1_stats = run_candidate_phase(
        ctx,
        candidates,
        phase_label="phase 1/2",
        phase_kind="phase1",
        plateau_after_no_improve=int(cfg.phase1_plateau_rounds),
        use_refine_tiebreak=False,
        n_total_override=int(n_trials_eff),
        seed_presets=list(phase1_seed_presets or []),
        optuna_builder=(
            (lambda tr, _base=dict(search_base_data): runtime.suggest_auto_mode_candidate_optuna(_base, tr))
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
        p1_optuna_txt = runtime.auto_optuna_telemetry_text(phase1_optuna_tel)
        p1_status_suffix = f", {p1_optuna_txt}" if p1_optuna_txt else ""
        logger.info(
            "Automatic mode Phase1 done: avg_score=%.3f, %s%s",
            _auto_safe_float(p1m.get("avg_score"), 0.0),
            str(p1_detail),
            str(p1_status_suffix),
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
                logger.info("Automatic mode Local refine: center #%d %s", int(ci), str(local_detail))
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
            stats = run_candidate_phase(
                ctx,
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
                study_scope=runtime.auto_optuna_scope_with_context(
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
            local_tel_txt = runtime.auto_optuna_telemetry_text(local_tel)
            local_rescue_suffix = ", zero-feasible fallback used" if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
            local_fallback_txt = runtime.auto_optuna_fallback_summary_text(local_tel) if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
            local_best_metrics = dict(search_state.best_metrics or {})
            local_rank_txt = runtime.auto_optuna_fmt_value(local_best_metrics.get("rank_score"), 3)
            local_avg_txt = runtime.auto_optuna_fmt_value(local_best_metrics.get("avg_score"), 3)
            logger.info(
                "Automatic mode Local refine summary: center #%d, current_best_rank=%s, avg=%s%s%s",
                int(ci),
                str(local_rank_txt),
                str(local_avg_txt),
                "" if not local_tel_txt else f", {local_tel_txt}",
                str(local_rescue_suffix),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: Local refine summary "
                    f"center #{int(ci)}, current_best_rank={local_rank_txt}, avg_score={local_avg_txt}"
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
                    "Automatic mode Local refine winner improved: avg_score %.3f -> %.3f, rank_score %.3f -> %.3f",
                    _auto_safe_float(before.get("avg_score"), 0.0),
                    _auto_safe_float((search_state.best_metrics or {}).get("avg_score"), 0.0),
                    _auto_safe_float(before.get("rank_score"), 0.0),
                    _auto_safe_float((search_state.best_metrics or {}).get("rank_score"), 0.0),
                )

    if bool(cfg.local_refine_keep_best_phase1) and isinstance(phase1_best_metrics, dict):
        if search_state.best_metrics is None or _auto_rank_key(search_state.best_metrics) > _auto_rank_key(phase1_best_metrics):
            from .search_state import _auto_set_search_winner

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
        micro_shrink = float(np.clip(micro_shrink * 0.70, AUTO_MODE_ADAPTIVE_SHRINK_MIN, 1.0))
        micro_center = dict(search_state.best_preset or {})
        micro_candidates = []
        micro_seed_presets = runtime.build_auto_mode_candidates_micro(
            search_base_data,
            dict(micro_center),
            n_trials=1,
            shrink=float(micro_shrink),
        )
        if not bool(use_optuna_trials):
            micro_candidates = runtime.build_auto_mode_candidates_micro(
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
                f"CamillaFIR automatic mode: Phase3 micro {int(cfg.phase3_micro_trials)} trials around current best"
            )
        before_micro = dict(search_state.best_metrics or {})
        micro_stats = run_candidate_phase(
            ctx,
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
            study_scope=runtime.auto_optuna_scope_with_context(
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
                "Automatic mode Phase3 micro improved: avg_score %.3f -> %.3f, rank_score %.3f -> %.3f",
                _auto_safe_float(before_micro.get("avg_score"), 0.0),
                _auto_safe_float((search_state.best_metrics or {}).get("avg_score"), 0.0),
                _auto_safe_float(before_micro.get("rank_score"), 0.0),
                _auto_safe_float((search_state.best_metrics or {}).get("rank_score"), 0.0),
            )
        micro_tel_txt = runtime.auto_optuna_telemetry_text(phase3_micro_optuna_tel)
        micro_rescue_suffix = ", zero-feasible fallback used" if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        micro_fallback_txt = runtime.auto_optuna_fallback_summary_text(phase3_micro_optuna_tel) if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        micro_best_metrics = dict(search_state.best_metrics or {})
        micro_rank_txt = runtime.auto_optuna_fmt_value(micro_best_metrics.get("rank_score"), 3)
        micro_avg_txt = runtime.auto_optuna_fmt_value(micro_best_metrics.get("avg_score"), 3)
        logger.info(
            "Automatic mode Phase3 micro summary: current_best_rank=%s, avg=%s%s%s",
            str(micro_rank_txt),
            str(micro_avg_txt),
            "" if not micro_tel_txt else f", {micro_tel_txt}",
            str(micro_rescue_suffix),
        )
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: Phase3 micro summary "
                f"current_best_rank={micro_rank_txt}, avg_score={micro_avg_txt}"
                f"{'' if not micro_tel_txt else f', {micro_tel_txt}'}"
                f"{micro_rescue_suffix}"
            )
        if micro_fallback_txt:
            logger.info("Automatic mode Phase3 micro fallback detail: %s", str(micro_fallback_txt))
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: Phase3 micro fallback "
                    f"{micro_fallback_txt}"
                )

    phase2_rollup_tel = build_phase2_rollup_telemetry(
        phase2_local_optuna_tels=phase2_local_optuna_tels,
        phase3_micro_optuna_tel=phase3_micro_optuna_tel,
        telemetry_rollup_fn=runtime.auto_optuna_telemetry_rollup,
    )
    phase2_rollup_txt = runtime.auto_optuna_telemetry_text(phase2_rollup_tel)
    if phase2_rollup_txt:
        logger.info("Automatic mode Phase2 summary: %s", str(phase2_rollup_txt))
        if callable(status_cb):
            status_cb(f"CamillaFIR automatic mode: Phase2 summary {phase2_rollup_txt}")
    return {
        "phase1_ok": int(phase1_ok),
        "phase2_ok": int(phase2_ok),
        "phase1_tried": int(phase1_tried),
        "phase2_tried": int(phase2_tried),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "phase1_optuna_tel": dict(phase1_optuna_tel or {}),
        "phase2_local_optuna_tels": list(phase2_local_optuna_tels or []),
        "phase3_micro_optuna_tel": dict(phase3_micro_optuna_tel or {}),
        "phase2_rollup_tel": dict(phase2_rollup_tel or {}),
    }


__all__ = [
    "run_exact_cache_micro_refine",
    "run_search_refine_stages",
]
