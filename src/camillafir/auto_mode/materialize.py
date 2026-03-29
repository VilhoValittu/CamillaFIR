from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from .shared import (
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _auto_safe_int,
)

logger = logging.getLogger("CamillaFIR")


@dataclass
class AutoModeMaterializeContext:
    cfg: Any
    cache_base_data: dict
    measurements: dict
    fs_v: int
    taps_v: int
    xos: list
    hpf: dict | None
    hc_f: Any
    hc_m: Any
    pin_obj: Any
    filter_key: str
    max_safe_boost: float
    goal: str
    status_cb: Callable[[str], None] | None
    exact_cached_metrics_getter: Callable[[], dict | None] | None
    auto_score_result_fn: Callable[..., dict]
    auto_optuna_jsonable_fn: Callable[[Any], Any]
    auto_rank_key_fn: Callable[[dict], Any]
    auto_is_better_refine_fn: Callable[..., Any]
    build_config_fn: Callable[..., Any]
    run_pipeline_fn: Callable[..., Any]
    summarize_run_fn: Callable[..., Any]
    preset_transient_keys: tuple[str, ...]
    residual_tiebreak_enabled: bool
    residual_top_k: int
    residual_rank_eps: float


def build_materialize_helpers(ctx: AutoModeMaterializeContext):
    cfg = ctx.cfg
    cache_base_data = dict(ctx.cache_base_data or {})
    measurements = dict(ctx.measurements or {})
    filter_key = str(ctx.filter_key or "")
    goal = str(ctx.goal or "")
    status_cb = ctx.status_cb
    transient_keys = tuple(str(key) for key in tuple(ctx.preset_transient_keys or ()))

    def _current_exact_cached_metrics() -> dict | None:
        getter = ctx.exact_cached_metrics_getter
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception as exc:
            logger.debug("Exact cached metrics getter failed: %s: %s", type(exc).__name__, exc)
            return None
        return dict(value or {}) if isinstance(value, dict) else None

    def _cache_ready_preset(
        preset: dict | None,
        *,
        best_metrics: dict | None = None,
    ) -> dict:
        out = dict(preset or {})
        for key in transient_keys:
            out.pop(str(key), None)
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
                    float(_auto_safe_float(getattr(cfg, "exc_min_hz", AUTO_MODE_EXC_MIN_HZ), AUTO_MODE_EXC_MIN_HZ)),
                    float(_auto_safe_float(getattr(cfg, "exc_max_hz", AUTO_MODE_EXC_MAX_HZ), AUTO_MODE_EXC_MAX_HZ)),
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
        base_data_override: dict | None = None,
        best_metrics_override: dict | None = None,
    ) -> tuple[object, dict, dict]:
        ready_preset = _cache_ready_preset(
            preset,
            best_metrics=(
                dict(best_metrics_override or {})
                if isinstance(best_metrics_override, dict)
                else _current_exact_cached_metrics()
            ),
        )
        final_data = dict(base_data_override or cache_base_data or {})
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

        cfg_final = ctx.build_config_fn(
            final_data,
            fs_v=int(ctx.fs_v),
            taps_v=int(ctx.taps_v),
            xos=ctx.xos,
            hpf=ctx.hpf,
            hc_f=ctx.hc_f,
            hc_m=ctx.hc_m,
            max_safe_boost=float(ctx.max_safe_boost),
        )
        try:
            setattr(cfg_final, "bass_smooth_w_gamma", float(final_data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg_final, "bass_smooth_w_max", float(final_data.get("bass_smooth_w_max", 0.45)))
        except Exception as exc:
            logger.debug("Could not set bass_smooth attrs on cfg: %s: %s", type(exc).__name__, exc)

        result = ctx.run_pipeline_fn(
            cfg_final,
            final_measurements,
            include_response_arrays=bool(include_response_arrays),
        )
        if bool(summarize):
            result.metrics["summary"] = ctx.summarize_run_fn(result)
        metrics = ctx.auto_score_result_fn(
            result,
            auto_exc_freq_hz=_auto_safe_float(
                final_data.get("_auto_exc_freq_hz", float("nan")),
                float("nan"),
            ),
            base_data=final_data,
        )
        return result, dict(metrics or {}), dict(final_data or {})

    def _preset_signature_ignoring_residual(preset: dict | None) -> str:
        base_preset = dict(preset or {})
        base_preset.pop("enable_residual_pass", None)
        try:
            payload = json.dumps(
                ctx.auto_optuna_jsonable_fn(_cache_ready_preset(base_preset)),
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception as exc:
            logger.debug("JSON serialization failed for preset signature, using str fallback: %s: %s", type(exc).__name__, exc)
            payload = str(sorted(base_preset.items()))
        return str(payload)

    def _maybe_apply_residual_tiebreak(
        *,
        best_preset: dict | None,
        best_metrics: dict | None,
        candidate_items: list[dict] | None,
        base_data_ref: dict | None,
        phase_label: str,
    ) -> tuple[dict, dict, bool]:
        cur_best_preset = dict(best_preset or {})
        cur_best_metrics = dict(best_metrics or {})
        if not bool(ctx.residual_tiebreak_enabled):
            return cur_best_preset, cur_best_metrics, False
        if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
            return cur_best_preset, cur_best_metrics, False
        if bool(cur_best_preset.get("enable_residual_pass", False)):
            return cur_best_preset, cur_best_metrics, False

        logger.debug("Automatic mode residual tie-break starting (%s)", phase_label)
        top_k = int(max(1, _auto_safe_int(ctx.residual_top_k, 3)))
        rank_eps = float(max(0.0, _auto_safe_float(ctx.residual_rank_eps, 0.35)))
        best_rank = _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
        seen: set[str] = set()
        shortlist: list[dict] = []

        def _maybe_add_candidate(preset: dict | None, metrics: dict | None, *, source: str) -> None:
            if len(shortlist) >= int(top_k):
                return
            cand_preset = _cache_ready_preset(
                dict(preset or {}),
                best_metrics=dict(metrics or {}),
            )
            if bool(cand_preset.get("enable_residual_pass", False)):
                return
            sig = _preset_signature_ignoring_residual(cand_preset)
            if sig in seen:
                return
            rank_v = _auto_safe_float(dict(metrics or {}).get("rank_score"), float("nan"))
            if np.isfinite(best_rank) and np.isfinite(rank_v):
                if float(best_rank - rank_v) > float(rank_eps):
                    return
            seen.add(sig)
            shortlist.append(
                {
                    "preset": dict(cand_preset or {}),
                    "metrics": dict(metrics or {}),
                    "source": str(source),
                }
            )

        _maybe_add_candidate(cur_best_preset, cur_best_metrics, source="current_best")
        ranked_items = sorted(
            [dict(it or {}) for it in list(candidate_items or []) if isinstance(it, dict)],
            key=lambda it: ctx.auto_rank_key_fn(dict(it.get("metrics", {}) or {})),
        )
        for item in ranked_items:
            _maybe_add_candidate(
                dict(item.get("preset", {}) or {}),
                dict(item.get("metrics", {}) or {}),
                source=str(item.get("phase", item.get("source", "candidate")) or "candidate"),
            )

        if not shortlist:
            return cur_best_preset, cur_best_metrics, False

        improved = False
        logger.info(
            "Automatic mode residual tie-break: testing %d finalist preset(s) within %.2f rank window.",
            int(len(shortlist)),
            float(rank_eps),
        )
        for idx, item in enumerate(shortlist, start=1):
            cand_base = dict(item.get("preset", {}) or {})
            cand_test = dict(cand_base or {})
            cand_test["enable_residual_pass"] = True
            try:
                _residual_result, residual_metrics, _residual_data = _materialize_preset_result(
                    cand_test,
                    include_response_arrays=False,
                    summarize=False,
                    base_data_override=base_data_ref,
                )
            except Exception as exc:
                logger.warning(
                    "Automatic mode residual tie-break failed for finalist %d/%d (%s): %s",
                    int(idx),
                    int(len(shortlist)),
                    str(item.get("source", "candidate")),
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            residual_metrics = dict(residual_metrics or {})
            decision = ctx.auto_is_better_refine_fn(
                residual_metrics,
                cur_best_metrics,
                goal,
                return_reason=True,
            )
            if isinstance(decision, tuple):
                better, reason = decision
            else:
                better, reason = bool(decision), ""
            # Guard: reject residual if avg_score drops significantly
            if bool(better):
                prev_avg = _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
                new_avg = _auto_safe_float(residual_metrics.get("avg_score"), float("nan"))
                if np.isfinite(prev_avg) and np.isfinite(new_avg):
                    avg_drop = float(prev_avg - new_avg)
                    if avg_drop > 3.0:
                        better = False
                        reason = "avg_score_guard"
                        logger.info(
                            "Automatic mode residual tie-break: rejected finalist %d/%d due to "
                            "avg_score drop %.1f (%.1f -> %.1f)",
                            int(idx), int(len(shortlist)),
                            float(avg_drop), float(prev_avg), float(new_avg),
                        )
                        continue
            base_rank = _auto_safe_float(dict(item.get("metrics", {}) or {}).get("rank_score"), float("nan"))
            new_rank = _auto_safe_float(residual_metrics.get("rank_score"), float("nan"))
            logger.info(
                "Automatic mode residual tie-break finalist %d/%d (%s): base_rank=%.3f -> residual_rank=%.3f, decision=%s (%s)",
                int(idx),
                int(len(shortlist)),
                str(item.get("source", "candidate")),
                float(base_rank) if np.isfinite(base_rank) else float("nan"),
                float(new_rank) if np.isfinite(new_rank) else float("nan"),
                "accept" if bool(better) else "reject",
                str(reason),
            )
            if not bool(better):
                continue

            prev_best = dict(cur_best_metrics or {})
            cur_best_metrics = dict(residual_metrics or {})
            cur_best_preset = _cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            logger.info(
                "Automatic mode residual tie-break accepted finalist %d/%d: rank %.3f -> %.3f, avg %.3f -> %.3f",
                int(idx),
                int(len(shortlist)),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: residual tie-break improved "
                    f"(rank {_auto_safe_float(cur_best_metrics.get('rank_score'), 0.0):.3f}, "
                    f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

        return cur_best_preset, cur_best_metrics, bool(improved)

    return (
        _cache_ready_preset,
        _materialize_preset_result,
        _preset_signature_ignoring_residual,
        _maybe_apply_residual_tiebreak,
    )


__all__ = [
    "AutoModeMaterializeContext",
    "build_materialize_helpers",
]
