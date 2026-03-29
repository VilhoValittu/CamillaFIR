"""Finalize-stage orchestration for automatic mode."""

from __future__ import annotations

import logging

import numpy as np

from .cache_signature import (
    _auto_cache_put_best,
    _auto_cache_put_last_used_best,
    _auto_cache_put_target_for_measurements,
    _auto_measurement_signature,
    _auto_signature,
)
from .candidate_generation import _seed_auto_mode_candidate_optuna_params
from .rank_score import attach_official_rank_score
from .runtime_context import coerce_orchestrator_runtime
from .scoring_ranking import (
    _auto_is_better_refine,
    _auto_mode_ripple_for_pareto,
    _auto_phase2_hard_gate_pool,
    _auto_phase2_pareto_front,
    _auto_prepost_for_pareto,
    _auto_prepost_lr_for_pareto,
    _auto_rank_key,
    _auto_realized_rms_20_200_for_pareto,
    _auto_select_best_scored,
)
from .shared import _auto_builtin_target_name, _auto_safe_float, _m
from .winner_polish import (
    apply_mag_c_min_winner_polish,
    apply_phase_limit_winner_polish,
)

logger = logging.getLogger("CamillaFIR")
__all__ = ["finalize_search_result"]


def _save_cached_best(
    *,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    cfg,
    goal: str,
    filter_key: str,
    compat_version: str,
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
        compat_version=compat_version,
    )
    _auto_cache_put_best(
        sig_target,
        best_preset=dict(best_preset or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=best_hc_mode_builtin,
        measurement_sig=measurement_sig,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )
    _auto_cache_put_target_for_measurements(
        measurements=measurements,
        best_hc_mode=best_hc_mode_builtin,
        best_preset=dict(best_preset or {}),
        best_metrics=dict(best_metrics or {}),
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )
    _auto_cache_put_last_used_best(
        best_preset=dict(best_preset or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=best_hc_mode,
        measurement_sig=measurement_sig,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )


def _finalize_cached_result(
    *,
    runtime,
    search_base_data: dict,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    cfg,
    goal: str,
    rank_basis: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    status_cb,
    optuna_mod,
    optuna_search_sig: str,
    seed: int,
    _cache_ready_preset,
    _materialize_preset_result,
    _maybe_apply_residual_tiebreak,
    cache_refine_result: dict,
) -> dict | None:
    cache_target_name = str(cache_refine_result.get("cache_target_name", "n/a") or "n/a")
    best_preset = dict(cache_refine_result.get("best_preset", {}) or {})
    best_metrics = dict(cache_refine_result.get("best_metrics", {}) or {})
    improved_any = bool(cache_refine_result.get("improved_any", False))
    improved_count_total = int(cache_refine_result.get("improved_count_total", 0) or 0)
    executed_micro_trials_total = int(cache_refine_result.get("executed_micro_trials_total", 0) or 0)
    cache_refine_rollup_tel = dict(cache_refine_result.get("cache_refine_rollup_tel", {}) or {})
    stop_reason = str(cache_refine_result.get("stop_reason", "max_rounds") or "max_rounds")
    try:
        best_preset, best_metrics, residual_cache_improved = _maybe_apply_residual_tiebreak(
            best_preset=best_preset,
            best_metrics=best_metrics,
            candidate_items=None,
            base_data_ref=cache_base_data,
            phase_label="cache residual tie-break",
        )
        if bool(residual_cache_improved):
            improved_any = True
            improved_count_total += 1

        best_preset, best_metrics, phase_limit_cache_improved, phase_limit_cache_meta = apply_phase_limit_winner_polish(
            best_preset=best_preset,
            best_metrics=best_metrics,
            base_data_ref=cache_base_data,
            phase_label="cache phase_limit winner polish",
            goal=goal,
            filter_key=filter_key,
            enabled=bool(runtime.phase_limit_winner_polish_enabled),
            offsets_hz=tuple(runtime.phase_limit_winner_polish_offsets_hz),
            status_cb=status_cb,
            materialize_preset_result=_materialize_preset_result,
            cache_ready_preset=_cache_ready_preset,
            auto_is_better_refine=_auto_is_better_refine,
        )
        if bool(phase_limit_cache_improved):
            improved_any = True
            improved_count_total += 1

        best_preset, best_metrics, mag_c_min_cache_improved, mag_c_min_cache_meta = apply_mag_c_min_winner_polish(
            best_preset=best_preset,
            best_metrics=best_metrics,
            base_data_ref=cache_base_data,
            phase_label="cache mag_c_min winner polish",
            goal=goal,
            enabled=bool(runtime.mag_c_min_winner_polish_enabled),
            step_hz=float(runtime.mag_c_min_winner_polish_step_hz),
            max_down_hz=float(runtime.mag_c_min_winner_polish_max_down_hz),
            status_cb=status_cb,
            materialize_preset_result=_materialize_preset_result,
            cache_ready_preset=_cache_ready_preset,
            auto_is_better_refine=_auto_is_better_refine,
        )
        if bool(mag_c_min_cache_improved):
            improved_any = True
            improved_count_total += 1

        best_result, best_metrics_recalc, best_data = _materialize_preset_result(
            best_preset,
            include_response_arrays=True,
            summarize=True,
            base_data_override=cache_base_data,
        )
        best_metrics = attach_official_rank_score(best_metrics_recalc or best_metrics)
        best_applied_preset = dict(best_data or best_preset or {})
        best_cache_preset = _cache_ready_preset(best_preset, best_metrics=best_metrics)
        if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
            raw_scope = "phase1"
            scope_eff = runtime.auto_optuna_effective_scope(
                cache_base_data,
                raw_scope,
                phase_kind="phase1",
            )
            runtime.auto_optuna_remember_result(
                optuna_mod,
                base_data=dict(cache_base_data or {}),
                study_name=runtime.auto_optuna_study_name(
                    study_sig=optuna_search_sig,
                    scope=scope_eff,
                ),
                study_scope=scope_eff,
                phase_kind="phase1",
                seed=int(seed + 500001),
                preset=dict(best_preset or {}),
                metrics=dict(best_metrics or {}),
                seed_to_params=(
                    lambda preset, _base=dict(cache_base_data): _seed_auto_mode_candidate_optuna_params(
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
            best_applied_preset.get(
                "_auto_exc_freq_hz",
                best_applied_preset.get("best_auto_exc_freq_hz", float("nan")),
            ),
            float("nan"),
        )
        _save_cached_best(
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            cfg=cfg,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
            best_preset=dict(best_cache_preset or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
        )
        winner_rank = float(
            _auto_safe_float(
                best_metrics.get("rank_score_official", best_metrics.get("rank_score", float("nan"))),
                float("nan"),
            )
        )
        winner_components = dict(best_metrics.get("rank_score_components", {}) or {})
        return {
            "best_result": best_result,
            "best_metrics": dict(best_metrics or {}),
            "best_preset": dict(best_cache_preset or {}),
            "best_applied_preset": dict(best_applied_preset or {}),
            "winner": {
                "rank_score_official": float(winner_rank) if np.isfinite(winner_rank) else float("nan"),
                "rank_score_components": dict(winner_components),
            },
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
            "best_auto_exc_freq_hz": float(cached_best_auto_exc_hz) if np.isfinite(cached_best_auto_exc_hz) else float("nan"),
            "phase_limit_winner_polish": dict(phase_limit_cache_meta or {}),
            "mag_c_min_winner_polish": dict(mag_c_min_cache_meta or {}),
            "optimizer_backend": str(optimizer_backend or "builtin"),
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
        # Exact-cache finalize is a best-effort fast path; search fallback remains authoritative.
        logger.warning(
            "Automatic mode: exact preset cache materialization failed, "
            f"falling back to search ({type(exc).__name__}: {exc})"
        )
        return None


def finalize_search_result(
    *,
    search_base_data: dict,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    cfg,
    goal: str,
    rank_basis: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    status_cb,
    optuna_mod,
    optuna_search_sig: str,
    seed: int,
    search_state,
    winner_target_name: str | None,
    phase1_ok: int,
    phase2_ok: int,
    phase1_tried: int,
    phase2_tried: int,
    phase1_plateau_hit: bool,
    phase2_plateau_hit: bool,
    phase1_optuna_tel: dict,
    phase2_local_optuna_tels: list,
    phase3_micro_optuna_tel: dict,
    phase2_rollup_tel: dict,
    _cache_ready_preset,
    _materialize_preset_result,
    _maybe_apply_residual_tiebreak,
    cache_refine_result: dict | None = None,
    runtime=None,
) -> dict | None:
    runtime = coerce_orchestrator_runtime(runtime)
    if isinstance(cache_refine_result, dict):
        return _finalize_cached_result(
            runtime=runtime,
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
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
            _cache_ready_preset=_cache_ready_preset,
            _materialize_preset_result=_materialize_preset_result,
            _maybe_apply_residual_tiebreak=_maybe_apply_residual_tiebreak,
            cache_refine_result=dict(cache_refine_result or {}),
        )
    if search_state is None:
        return None

    phase2_pool_raw = [dict(it or {}) for it in (search_state.phase2_pool or []) if isinstance(it, dict)]
    if phase2_pool_raw:
        phase2_rank_vals = [
            _m(dict(it.get("metrics", {}) or {}), "rank_score", float("nan"))
            for it in phase2_pool_raw
        ]
        phase2_rank_vals = [float(v) for v in phase2_rank_vals if np.isfinite(v)]
        phase2_best_rank = max(phase2_rank_vals) if phase2_rank_vals else float("nan")
        rank_win = float(max(0.0, _auto_safe_float(cfg.phase2_pareto_rank_window, 2.0)))
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
            "Phase2 pool size: %d (kept %d)",
            int(len(phase2_pool_raw)),
            int(len(phase2_kept)),
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
            logger.info("Pareto front size: %d", int(len(front)))
            rank_best = dict(_auto_select_best_scored(phase2_kept) or phase2_kept[0])
            pareto_pool = [
                {
                    **dict(it or {}),
                    "_auto_select_kind": "phase2_pareto",
                    "_phase2_pareto_acoustic_drop": float(
                        _auto_safe_float(cfg.phase2_pareto_acoustic_drop, 0.35)
                    ),
                }
                for it in phase2_kept
            ]
            pareto_winner = _auto_select_best_scored(pareto_pool)
            if isinstance(pareto_winner, dict):
                from .search_state import _auto_set_search_winner

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
                    "Pareto winner: avg=%.3f, prepost=%.4f, mode_ripple=%.3f, rms20_200=%.3f, net_boost=%.3f",
                    _m(w_metrics, "avg_score", 0.0),
                    w_prepost if np.isfinite(w_prepost) else float("nan"),
                    w_mode_ripple if np.isfinite(w_mode_ripple) else float("nan"),
                    w_rms20 if np.isfinite(w_rms20) else float("nan"),
                    w_boost if np.isfinite(w_boost) else float("nan"),
                )
                rb_metrics = dict(rank_best.get("metrics", {}) or {})
                rb_prepost = _auto_prepost_for_pareto(rb_metrics)
                rb_mode_ripple = _auto_mode_ripple_for_pareto(rb_metrics)
                logger.info(
                    "Best-by-rank would have been: avg=%.3f, prepost=%.4f, mode_ripple=%.3f, rms20_200=%.3f, net_boost=%.3f",
                    _m(rb_metrics, "avg_score", 0.0),
                    rb_prepost if np.isfinite(rb_prepost) else float("nan"),
                    rb_mode_ripple,
                    _auto_realized_rms_20_200_for_pareto(rb_metrics),
                    _m(rb_metrics, "max_net_boost_db", float("nan")),
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
                "Pareto front skipped: phase2 kept pool too small (%d < %d)",
                int(len(phase2_kept)),
                int(pareto_min_n),
            )

    if search_state.best_metrics is None or not isinstance(search_state.best_preset, dict):
        return None

    residual_candidate_items = list(search_state.phase2_pool or search_state.scored or [])
    residual_best_preset, residual_best_metrics, residual_improved = _maybe_apply_residual_tiebreak(
        best_preset=search_state.best_preset,
        best_metrics=search_state.best_metrics,
        candidate_items=residual_candidate_items,
        base_data_ref=search_base_data,
        phase_label="residual tie-break",
    )
    if bool(residual_improved):
        from .search_state import _auto_set_search_winner

        prev_best = dict(search_state.best_metrics or {})
        _auto_set_search_winner(
            search_state,
            residual_best_metrics,
            residual_best_preset,
            prev_metrics=prev_best,
            phase_label="residual tie-break",
            target_name=winner_target_name,
        )

    polished_best_preset, polished_best_metrics, phase_limit_polish_improved, phase_limit_polish_meta = apply_phase_limit_winner_polish(
        best_preset=search_state.best_preset,
        best_metrics=search_state.best_metrics,
        base_data_ref=search_base_data,
        phase_label="phase_limit winner polish",
        goal=goal,
        filter_key=filter_key,
        enabled=bool(runtime.phase_limit_winner_polish_enabled),
        offsets_hz=tuple(runtime.phase_limit_winner_polish_offsets_hz),
        status_cb=status_cb,
        materialize_preset_result=_materialize_preset_result,
        cache_ready_preset=_cache_ready_preset,
        auto_is_better_refine=_auto_is_better_refine,
    )
    if bool(phase_limit_polish_improved):
        from .search_state import _auto_set_search_winner

        prev_best = dict(search_state.best_metrics or {})
        _auto_set_search_winner(
            search_state,
            polished_best_metrics,
            polished_best_preset,
            prev_metrics=prev_best,
            phase_label="phase_limit winner polish",
            target_name=winner_target_name,
        )

    mag_c_min_best_preset, mag_c_min_best_metrics, mag_c_min_polish_improved, mag_c_min_polish_meta = apply_mag_c_min_winner_polish(
        best_preset=search_state.best_preset,
        best_metrics=search_state.best_metrics,
        base_data_ref=search_base_data,
        phase_label="mag_c_min winner polish",
        goal=goal,
        enabled=bool(runtime.mag_c_min_winner_polish_enabled),
        step_hz=float(runtime.mag_c_min_winner_polish_step_hz),
        max_down_hz=float(runtime.mag_c_min_winner_polish_max_down_hz),
        status_cb=status_cb,
        materialize_preset_result=_materialize_preset_result,
        cache_ready_preset=_cache_ready_preset,
        auto_is_better_refine=_auto_is_better_refine,
    )
    if bool(mag_c_min_polish_improved):
        from .search_state import _auto_set_search_winner

        prev_best = dict(search_state.best_metrics or {})
        _auto_set_search_winner(
            search_state,
            mag_c_min_best_metrics,
            mag_c_min_best_preset,
            prev_metrics=prev_best,
            phase_label="mag_c_min winner polish",
            target_name=winner_target_name,
        )

    try:
        final_best_preset = dict(search_state.best_preset or {})
        best_result, best_metrics_recalc, best_data = _materialize_preset_result(
            final_best_preset,
            include_response_arrays=True,
            summarize=True,
            base_data_override=search_base_data,
        )
        search_state.best_result = best_result
        search_state.best_metrics = dict(best_metrics_recalc or {})
        search_state.best_preset = dict(best_data or final_best_preset or {})
    except Exception as exc:
        # Materialization can fail on late-stage result packaging; keep the last known winner if available.
        logger.warning(
            "Automatic mode final materialization failed: %s",
            f"{type(exc).__name__}: {exc}",
        )
        if search_state.best_result is None:
            return None

    top = sorted(
        search_state.scored,
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[:5]
    logger.info(
        "Automatic mode search result: goal=%s, basis=%s, rank=%.3f",
        str(goal),
        str(rank_basis),
        _auto_safe_float(search_state.best_metrics.get("rank_score"), 0.0),
    )

    best_auto_exc_hz = _auto_safe_float(
        dict(search_state.best_metrics or {}).get("auto_exc_zero_penalty_hz", float("nan")),
        float("nan"),
    )
    materialized_best_preset = dict(search_state.best_preset or {})
    cached_best_preset = _cache_ready_preset(
        final_best_preset if "final_best_preset" in locals() else materialized_best_preset,
        best_metrics=search_state.best_metrics,
    )

    if bool(cfg.cache_enabled):
        try:
            _save_cached_best(
                cache_base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                cfg=cfg,
                goal=goal,
                filter_key=filter_key,
                compat_version=compat_version,
                best_preset=dict(cached_best_preset or {}),
                best_metrics=dict(search_state.best_metrics or {}),
                best_hc_mode=str(search_base_data.get("hc_mode", "") or "").strip() or None,
            )
            logger.info("Automatic mode: saved best preset to cache.")
        except Exception:
            # Cache persistence must not change the selected winner when the cache backend misbehaves.
            pass

    winner_rank = float(
        _auto_safe_float(
            search_state.best_metrics.get("rank_score_official", search_state.best_metrics.get("rank_score", float("nan"))),
            float("nan"),
        )
    )
    winner_components = dict(search_state.best_metrics.get("rank_score_components", {}) or {})
    return {
        "best_result": search_state.best_result,
        "best_metrics": dict(search_state.best_metrics),
        "best_preset": dict(cached_best_preset or {}),
        "best_applied_preset": dict(materialized_best_preset or {}),
        "winner": {
            "rank_score_official": float(winner_rank) if np.isfinite(winner_rank) else float("nan"),
            "rank_score_components": dict(winner_components),
        },
        "winner_explanation": dict(search_state.winner_explanation or {}),
        "phase_limit_winner_polish": dict(phase_limit_polish_meta or {}),
        "mag_c_min_winner_polish": dict(mag_c_min_polish_meta or {}),
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


__all__ = [
    "finalize_search_result",
]
