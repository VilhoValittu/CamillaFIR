"""Pure helpers for auto-mode refine candidate evaluation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .search_state import _AutoModePhaseState, _AutoModeSearchState, _auto_set_search_winner
from .scoring_ranking import _auto_is_better_refine, _auto_rank_key
from .shared import (
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_MODE_SOFT_K,
    MAX_SAFE_BOOST,
    _auto_metric_text,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _auto_trial_chunk_size,
    _auto_trial_workers,
    _clip,
)

logger = logging.getLogger("CamillaFIR")
__all__ = [
    "RefineEvalContext",
    "apply_refine_mode_soft_penalty",
    "build_phase2_rollup_telemetry",
    "evaluate_search_candidate",
    "normalize_trial_preset",
    "run_candidate_phase",
]


@dataclass
class RefineEvalContext:
    search_base_data: dict
    measurements: dict
    fs_v: int
    taps_v: int
    xos: list
    hpf: dict | None
    hc_f: Any
    hc_m: Any
    pin_obj: Any
    cfg: Any
    goal: str
    filter_key: str
    optimizer_backend: str
    optuna_mod: Any
    seed: int
    optuna_search_sig: str
    status_cb: Callable[[str], None] | None
    status_prefix: str
    winner_target_name: str | None
    search_state: _AutoModeSearchState
    runtime: Any

    def __post_init__(self) -> None:
        if self.cfg is None:
            raise ValueError("RefineEvalContext: cfg must not be None")
        if self.runtime is None:
            raise ValueError("RefineEvalContext: runtime must not be None")
        if self.search_state is None:
            raise ValueError("RefineEvalContext: search_state must not be None")


def apply_refine_mode_soft_penalty(
    metrics: dict | None,
    *,
    refine_mode_soft_k,
) -> dict:
    out = dict(metrics or {})
    soft_k = float(
        max(
            0.0,
            _auto_safe_float(refine_mode_soft_k, AUTO_MODE_REFINE_MODE_SOFT_K),
        )
    )
    mode_ripple = _auto_safe_float(out.get("mode_ripple_db"), float("nan"))
    rank_base = _auto_safe_float(out.get("rank_score"), 0.0)
    soft_pen = (
        float(soft_k) * max(0.0, float(mode_ripple))
        if np.isfinite(mode_ripple)
        else 0.0
    )
    out["mode_ripple_soft_penalty"] = float(soft_pen)
    out["rank_score_refine"] = float(rank_base - float(soft_pen))
    return out


def normalize_trial_preset(
    preset: dict | None,
    *,
    trial_data: dict | None,
    base_data: dict | None,
    filter_key: str | None,
) -> dict:
    trial_preset = dict(preset or {})
    trial_data = dict(trial_data or {})
    base_data = dict(base_data or {})
    filter_name = str(filter_key or "")
    if filter_name == "mixed":
        trial_preset["mixed_freq"] = round(
            _clip(
                trial_data.get("mixed_freq", base_data.get("mixed_freq", 180.0)),
                80.0,
                320.0,
            ),
            1,
        )
    elif filter_name in ("linear", "asym"):
        trial_preset["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_data.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    return dict(trial_preset)


def build_phase2_rollup_telemetry(
    *,
    phase2_local_optuna_tels: list | None,
    phase3_micro_optuna_tel: dict | None,
    telemetry_rollup_fn,
) -> dict:
    phase2_roll_items = [
        dict((it or {}).get("telemetry", {}) or {})
        for it in list(phase2_local_optuna_tels or [])
        if isinstance(it, dict)
    ]
    if isinstance(phase3_micro_optuna_tel, dict) and phase3_micro_optuna_tel:
        phase2_roll_items.append(dict(phase3_micro_optuna_tel))
    return dict(telemetry_rollup_fn(phase2_roll_items) or {})


def evaluate_search_candidate(
    ctx: RefineEvalContext,
    *,
    preset: dict,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
) -> dict:
    trial_data = dict(ctx.search_base_data or {})
    trial_data.update(dict(preset or {}))
    if str(ctx.filter_key) in ("linear", "asym"):
        trial_data["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get(
                        "phase_limit",
                        ctx.search_base_data.get("phase_limit", 400.0),
                    ),
                    default=400.0,
                )
            ),
            1,
        )
    trial_data["comparison_mode"] = True
    trial_measurements = dict(ctx.measurements or {})
    trial_measurements["ui_data"] = trial_data
    cfg_trial = ctx.runtime.build_config(
        trial_data,
        fs_v=int(ctx.fs_v),
        taps_v=int(ctx.taps_v),
        xos=ctx.xos,
        hpf=ctx.hpf,
        hc_f=ctx.hc_f,
        hc_m=ctx.hc_m,
        max_safe_boost=float(MAX_SAFE_BOOST),
    )
    try:
        setattr(
            cfg_trial,
            "bass_smooth_w_gamma",
            float(trial_data.get("bass_smooth_w_gamma", 2.40)),
        )
        setattr(
            cfg_trial,
            "bass_smooth_w_max",
            float(trial_data.get("bass_smooth_w_max", 0.45)),
        )
    except Exception:
        # Trial-only tuning attributes are optional across config variants.
        pass
    result = ctx.runtime.run_pipeline(
        cfg_trial,
        trial_measurements,
        include_response_arrays=False,
    )
    metrics = ctx.runtime.auto_score_result(
        result,
        auto_exc_freq_hz=_auto_safe_float(
            trial_data.get("_auto_exc_freq_hz", float("nan")),
            float("nan"),
        ),
        focus_lo_hz=focus_lo_hz if bool(use_refine_tiebreak) else None,
        focus_hi_hz=focus_hi_hz if bool(use_refine_tiebreak) else None,
        base_data=trial_data,
    )
    if bool(use_refine_tiebreak):
        metrics = apply_refine_mode_soft_penalty(
            metrics,
            refine_mode_soft_k=getattr(
                ctx.cfg,
                "refine_mode_soft_k",
                AUTO_MODE_REFINE_MODE_SOFT_K,
            ),
        )
    trial_preset = normalize_trial_preset(
        preset,
        trial_data=trial_data,
        base_data=ctx.search_base_data,
        filter_key=ctx.filter_key,
    )
    return {
        "ok": True,
        "metrics": dict(metrics or {}),
        "trial_preset": dict(trial_preset),
    }


def _consume_phase_result(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    idx: int,
    n_total: int,
    out: dict,
    phase_label: str,
    plateau_after_no_improve: int,
    use_refine_tiebreak: bool,
) -> bool:
    phase_state.tried_n = int(idx)
    improved = False

    if bool(out.get("ok", False)):
        metrics = dict(out.get("metrics", {}) or {})
        metrics["trial"] = int(len(ctx.search_state.scored) + 1)
        metrics["phase"] = str(phase_label)
        trial_preset = dict(out.get("trial_preset", {}) or {})
        ctx.search_state.scored.append(
            {"metrics": dict(metrics), "preset": dict(trial_preset)}
        )
        if bool(use_refine_tiebreak):
            ctx.search_state.phase2_pool.append(
                {
                    "preset": dict(trial_preset),
                    "metrics": dict(metrics or {}),
                }
            )
        phase_state.ok_n += 1

        better = False
        refine_reason = "rank"
        if ctx.search_state.best_metrics is None:
            better = True
        elif bool(use_refine_tiebreak):
            better, refine_reason = _auto_is_better_refine(
                metrics,
                ctx.search_state.best_metrics,
                ctx.goal,
                return_reason=True,
            )
        else:
            better = bool(_auto_rank_key(metrics) < _auto_rank_key(ctx.search_state.best_metrics))

        if better:
            prev_best = dict(ctx.search_state.best_metrics or {})
            _auto_set_search_winner(
                ctx.search_state,
                metrics,
                trial_preset,
                prev_metrics=prev_best,
                phase_label=phase_label,
                target_name=ctx.winner_target_name,
            )
            improved = True
            phase_state.improved_any = True
            if bool(use_refine_tiebreak) and str(refine_reason) == "mode_ripple":
                rank_prev = _auto_safe_float(prev_best.get("rank_score"), 0.0)
                rank_new = _auto_safe_float(metrics.get("rank_score"), 0.0)
                rank_eps = float(
                    max(
                        0.0,
                        _auto_safe_float(
                            ctx.cfg.refine_tiebreak_rank_eps,
                            AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
                        ),
                    )
                )
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
            "Automatic mode trial %d/%d failed (%s): %s",
            int(idx),
            int(n_total),
            str(phase_label),
            str(err_txt),
        )

    if callable(ctx.status_cb):
        rank_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("rank_score"), 0.0)
        if bool(improved):
            avg_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("avg_score"), 0.0)
            mode_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("mode_ripple_db"), float("nan"))
            boost_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("max_net_boost_db"), float("nan"))
            ctx.status_cb(
                f"{ctx.status_prefix}: {phase_label} best improved {idx}/{n_total} "
                f"(goal {ctx.goal}, rank {rank_now:.3f}, avg {avg_now:.3f}, "
                f"mode {'n/a' if not np.isfinite(mode_now) else f'{mode_now:.3f} dB'}, "
                f"boost {'n/a' if not np.isfinite(boost_now) else f'{boost_now:.2f} dB'}, "
                f"ok {int(phase_state.ok_n)}/{int(phase_state.tried_n)})"
            )
        elif ctx.search_state.best_metrics is not None:
            ctx.status_cb(
                f"{ctx.status_prefix}: {phase_label} {idx}/{n_total} "
                f"(rank {rank_now:.3f}, ok {int(phase_state.ok_n)}/{int(phase_state.tried_n)})"
            )

    if int(plateau_after_no_improve) > 0:
        if improved:
            phase_state.no_improve_streak = 0
        else:
            phase_state.no_improve_streak += 1
        if phase_state.no_improve_streak >= int(plateau_after_no_improve):
            phase_state.plateau_hit = True
            best_now = (
                "n/a"
                if not ctx.search_state.best_metrics
                else _auto_metric_text(ctx.search_state.best_metrics, ctx.goal)
            )
            move_txt = "plateau -> phase 2" if "1/2" in str(phase_label) else "plateau -> stop"
            logger.info(
                "Automatic mode %s: no-improve plateau detected (%d rounds), %s.",
                str(phase_label),
                int(plateau_after_no_improve),
                str(move_txt),
            )
            if callable(ctx.status_cb):
                ctx.status_cb(
                    f"{ctx.status_prefix}: {phase_label} {idx}/{n_total} "
                    f"(best {best_now}, {move_txt})"
                )
            return True
    return False


def run_candidate_phase(
    ctx: RefineEvalContext,
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
        str(ctx.optimizer_backend) == "optuna"
        and ctx.runtime.auto_optuna_module_ready(ctx.optuna_mod)
        and callable(optuna_builder)
    )
    n_total = int(n_total_override) if n_total_override is not None else int(len(cands))
    workers = int(_auto_trial_workers(ctx.search_base_data, n_total))
    if workers > 1:
        logger.info(
            "Automatic mode %s: parallel trial run enabled (%d workers)",
            str(phase_label),
            int(workers),
        )

    def _eval_one(idx: int, preset: dict) -> dict:
        out = evaluate_search_candidate(
            ctx,
            preset=dict(preset or {}),
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
        )
        return {
            "idx": int(idx),
            **dict(out or {}),
        }

    def _consume_one(idx: int, out: dict) -> bool:
        return _consume_phase_result(
            ctx,
            phase_state,
            idx=int(idx),
            n_total=int(n_total),
            out=dict(out or {}),
            phase_label=phase_label,
            plateau_after_no_improve=int(plateau_after_no_improve),
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )

    if bool(use_optuna_phase):
        raw_scope = str(study_scope or phase_label)
        scope_eff = ctx.runtime.auto_optuna_effective_scope(
            ctx.search_base_data,
            raw_scope,
            phase_kind=phase_kind,
        )
        study_name = ctx.runtime.auto_optuna_study_name(
            study_sig=ctx.optuna_search_sig,
            scope=scope_eff,
        )
        phase_tel = dict(
            ctx.runtime.auto_run_optuna_eval_loop(
                optuna_mod=ctx.optuna_mod,
                cfg=ctx.cfg,
                n_total=int(n_total),
                seed=int(ctx.seed + sum(ord(ch) for ch in str(phase_label)) * 31),
                base_data=dict(ctx.search_base_data or {}),
                seed_presets=list(seed_presets or []),
                build_preset=optuna_builder,
                eval_one=_eval_one,
                consume_one=_consume_one,
                objective_value=lambda out: ctx.runtime.auto_optuna_objective_value(
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
        if ctx.runtime.auto_optuna_needs_zero_feasible_rescue(
            base_data=ctx.search_base_data,
            phase_kind=phase_kind,
            telemetry=phase_tel,
        ):
            logger.warning(
                "Automatic mode Optuna rescue fallback: phase=%s scope=%s rerunning without constraints",
                str(phase_kind or ""),
                str(raw_scope),
            )
            rescue_base_data = ctx.runtime.auto_optuna_base_data_without_constraints(
                ctx.search_base_data
            )
            rescue_scope = f"{str(raw_scope)}-zf0"
            rescue_scope_eff = ctx.runtime.auto_optuna_effective_scope(
                rescue_base_data,
                rescue_scope,
                phase_kind=phase_kind,
            )
            rescue_tel = dict(
                ctx.runtime.auto_run_optuna_eval_loop(
                    optuna_mod=ctx.optuna_mod,
                    cfg=ctx.cfg,
                    n_total=int(n_total),
                    seed=int(ctx.seed + sum(ord(ch) for ch in str(phase_label)) * 31),
                    base_data=dict(rescue_base_data or {}),
                    seed_presets=list(seed_presets or []),
                    build_preset=optuna_builder,
                    eval_one=_eval_one,
                    consume_one=_consume_one,
                    objective_value=lambda out: ctx.runtime.auto_optuna_objective_value(
                        dict((out or {}).get("metrics", {}) or {}),
                        use_refine_tiebreak=bool(use_refine_tiebreak),
                    ),
                    workers=int(workers),
                    seed_to_params=seed_to_params,
                    study_name=ctx.runtime.auto_optuna_study_name(
                        study_sig=ctx.optuna_search_sig,
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
            "optuna_zero_feasible_fallback_used": bool(
                phase_tel.get("zero_feasible_fallback_used", False)
            ),
            "optuna_zero_feasible_fallback_telemetry": dict(
                phase_tel.get("fallback_telemetry", {}) or {}
            ),
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
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
                chunk = idx_presets[c0 : c0 + int(chunk_size)]
                future_map = {
                    executor.submit(_eval_one, int(idx), dict(preset or {})): int(idx)
                    for idx, preset in chunk
                }
                chunk_out: dict[int, dict] = {}
                for future in as_completed(list(future_map.keys())):
                    idx = int(future_map.get(future, 0))
                    try:
                        out = future.result()
                        if not isinstance(out, dict):
                            out = {
                                "idx": int(idx),
                                "ok": False,
                                "error": "invalid worker result",
                            }
                    except Exception as exc:
                        out = {
                            "idx": int(idx),
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    chunk_out[int(idx)] = dict(out)
                for idx, _preset in chunk:
                    out = dict(
                        chunk_out.get(
                            int(idx),
                            {
                                "idx": int(idx),
                                "ok": False,
                                "error": "missing worker result",
                            },
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


__all__ = [
    "apply_refine_mode_soft_penalty",
    "RefineEvalContext",
    "build_phase2_rollup_telemetry",
    "evaluate_search_candidate",
    "normalize_trial_preset",
    "run_candidate_phase",
]
