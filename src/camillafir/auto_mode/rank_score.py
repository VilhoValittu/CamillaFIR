from __future__ import annotations

from typing import Any

import numpy as np

from .shared import _auto_safe_float

OFFICIAL_RANK_SCORE_CONTEXT = "preset_objective_score"
RUN_RANKING_SCORE_CONTEXT = "run_ranking_score"


def compute_rank_score_components(
    *,
    avg_score: Any,
    boost_penalty: Any = 0.0,
    event_penalty: Any = 0.0,
    lr_delta_penalty: Any = 0.0,
    dsp_penalty: Any = 0.0,
    exc_penalty: Any = 0.0,
    mode_penalty: Any = 0.0,
    phase_limit_penalty: Any = 0.0,
    gain: Any = 1.0,
    bias: Any = 0.0,
    score_min: Any = 0.0,
    score_max: Any = 100.0,
    context: str | None = None,
) -> dict[str, Any]:
    avg = float(_auto_safe_float(avg_score, 0.0))
    boost = float(max(0.0, _auto_safe_float(boost_penalty, 0.0)))
    event = float(max(0.0, _auto_safe_float(event_penalty, 0.0)))
    lr_pen = float(max(0.0, _auto_safe_float(lr_delta_penalty, 0.0)))
    dsp_pen = float(max(0.0, _auto_safe_float(dsp_penalty, 0.0)))
    exc_pen = float(max(0.0, _auto_safe_float(exc_penalty, 0.0)))
    mode_pen = float(max(0.0, _auto_safe_float(mode_penalty, 0.0)))
    phase_pen = float(max(0.0, _auto_safe_float(phase_limit_penalty, 0.0)))
    g = float(_auto_safe_float(gain, 1.0))
    b = float(_auto_safe_float(bias, 0.0))
    lo = float(_auto_safe_float(score_min, 0.0))
    hi = float(_auto_safe_float(score_max, 100.0))
    if hi < lo:
        lo, hi = hi, lo

    rank_raw = float(avg - boost - event - lr_pen - dsp_pen - exc_pen - mode_pen - phase_pen)
    rank_score = float(np.clip((g * rank_raw) + b, lo, hi))
    score_kind = str(context or OFFICIAL_RANK_SCORE_CONTEXT).strip() or OFFICIAL_RANK_SCORE_CONTEXT
    score_label = "Best rank score" if score_kind == OFFICIAL_RANK_SCORE_CONTEXT else "Run ranking score"

    return {
        "rank_score": float(rank_score),
        "avg_score": float(avg),
        "boost_penalty": float(boost),
        "event_penalty": float(event),
        "lr_delta_penalty": float(lr_pen),
        "dsp_penalty": float(dsp_pen),
        "exc_penalty": float(exc_pen),
        "mode_penalty": float(mode_pen),
        "phase_limit_penalty": float(phase_pen),
        "rank_score_raw": float(rank_raw),
        "rank_score_gain": float(g),
        "rank_score_bias": float(b),
        "context": {
            "score_kind": str(score_kind),
            "score_label": str(score_label),
            "score_min": float(lo),
            "score_max": float(hi),
        },
    }


def attach_official_rank_score(
    metrics: dict[str, Any] | None,
    *,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(metrics or {})
    comp = dict(components or out.get("rank_score_components", {}) or {})
    if not comp:
        rank_score = _auto_safe_float(out.get("rank_score", float("nan")), float("nan"))
        if np.isfinite(rank_score):
            comp = {
                "rank_score": float(rank_score),
                "avg_score": float(_auto_safe_float(out.get("avg_score", float("nan")), float("nan"))),
                "boost_penalty": float(_auto_safe_float(out.get("boost_penalty", 0.0), 0.0)),
                "event_penalty": float(_auto_safe_float(out.get("event_penalty", 0.0), 0.0)),
                "lr_delta_penalty": float(_auto_safe_float(out.get("lr_delta_penalty", 0.0), 0.0)),
                "dsp_penalty": float(_auto_safe_float(out.get("dsp_penalty", 0.0), 0.0)),
                "exc_penalty": float(_auto_safe_float(out.get("exc_penalty", 0.0), 0.0)),
                "mode_penalty": float(_auto_safe_float(out.get("mode_penalty", 0.0), 0.0)),
                "phase_limit_penalty": float(_auto_safe_float(out.get("phase_limit_penalty", 0.0), 0.0)),
                "context": {
                    "score_kind": str(OFFICIAL_RANK_SCORE_CONTEXT),
                    "score_label": "Best rank score",
                },
            }
    official = float(
        _auto_safe_float(
            out.get(
                "rank_score_official",
                comp.get("rank_score", out.get("rank_score", float("nan"))),
            ),
            float("nan"),
        )
    )
    if np.isfinite(official):
        out["rank_score_official"] = float(official)
        out["rank_score_components"] = dict(comp)
    return out


def compute_run_ranking_score_components(
    *,
    avg_score: Any,
    boost_penalty: Any = 0.0,
    event_penalty: Any = 0.0,
    lr_delta_penalty: Any = 0.0,
    dsp_penalty: Any = 0.0,
) -> dict[str, Any]:
    components = compute_rank_score_components(
        avg_score=avg_score,
        boost_penalty=boost_penalty,
        event_penalty=event_penalty,
        lr_delta_penalty=lr_delta_penalty,
        dsp_penalty=dsp_penalty,
        context=RUN_RANKING_SCORE_CONTEXT,
    )
    out = dict(components)
    out["run_ranking_score"] = float(components.get("rank_score", 0.0))
    out["run_ranking_score_components"] = dict(components)
    return out


def official_rank_score(metrics: dict[str, Any] | None) -> float:
    m = dict(metrics or {})
    score = _auto_safe_float(m.get("rank_score_official", float("nan")), float("nan"))
    if np.isfinite(score):
        return float(score)
    comp = dict(m.get("rank_score_components", {}) or {})
    score = _auto_safe_float(comp.get("rank_score", float("nan")), float("nan"))
    if np.isfinite(score):
        return float(score)
    return float(_auto_safe_float(m.get("rank_score", float("nan")), float("nan")))
