import math
from typing import Any

from . import camillafir_plot as plots
from ..config.results import FilterResult
from ..auto_mode.rank_score import (
    attach_official_rank_score,
    compute_run_ranking_score_components,
    official_rank_score,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if x == x and abs(x) != float("inf"):
            return x
    except Exception:
        pass
    return float(default)


def _collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []


def _pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _safe_float(st.get(k, float("nan")), float("nan"))
        if not (v == v) or abs(v) == float("inf"):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _dsp_quality_penalty(st: dict | None) -> float:
    st = st or {}
    penalty = 0.0

    real_rms = _pick_metric(
        st,
        (
            "real_mag_error_rms",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)

    ripple_rms = _pick_metric(
        st,
        (
            "ripple_rms",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)

    gd_grad_max = _pick_metric(
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

    if not bool(st.get("pre_energy_metric_suspect", False)):
        pre_ringing_db = _pick_metric(
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

        pre_post_ratio = _pick_metric(
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

    phase_boundary_mdb = _pick_metric(
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

    return float(max(0.0, penalty))


def _score_export_result(result: FilterResult) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = plots.calc_ai_summary_from_stats(l_st)
    r_ai = plots.calc_ai_summary_from_stats(r_st)

    l_score = l_ai.get("score")
    r_score = r_ai.get("score")
    l_score_f = _safe_float(l_score, 0.0)
    r_score_f = _safe_float(r_score, 0.0)
    avg_score = (l_score_f + r_score_f) / 2.0
    lr_delta = abs(l_score_f - r_score_f)

    net_boost_max = max(
        _safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        _safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    reflections_n = len(_collect_reflections(l_st)) + len(_collect_reflections(r_st))
    dsp_penalty = 0.5 * (_dsp_quality_penalty(l_st) + _dsp_quality_penalty(r_st))

    boost_pen = 1.5 * max(0.0, net_boost_max - 1.0)
    event_pen = 0.5 * float(reflections_n)
    lr_pen = 0.25 * lr_delta
    run_score = compute_run_ranking_score_components(
        avg_score=avg_score,
        boost_penalty=boost_pen,
        event_penalty=event_pen,
        lr_delta_penalty=lr_pen,
        dsp_penalty=dsp_penalty,
    )

    return {
        "fs": int(getattr(result, "fs", 0) or 0),
        "run_ranking_score": float(run_score.get("run_ranking_score", 0.0)),
        "run_ranking_score_components": dict(run_score.get("run_ranking_score_components", {}) or {}),
        "avg_score": float(avg_score),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "events_total": int(reflections_n),
        "boost_penalty": float(boost_pen),
        "event_penalty": float(event_pen),
        "lr_delta_penalty": float(lr_pen),
        "dsp_penalty": float(dsp_penalty),
    }


def _build_export_ranking(results: list[FilterResult]) -> dict:
    entries = [_score_export_result(r) for r in list(results or [])]
    entries = sorted(
        entries,
        key=lambda e: (
            -_safe_float(e.get("run_ranking_score"), 0.0),
            -_safe_float(e.get("avg_score"), 0.0),
            _safe_float(e.get("max_net_boost_db"), 0.0),
            _safe_float(e.get("events_total"), 0.0),
            _safe_float(e.get("lr_delta_score"), 0.0),
        ),
    )
    for idx, e in enumerate(entries, start=1):
        e["rank"] = int(idx)
    best = entries[0] if entries else None
    return {
        "entries": entries,
        "best_fs": None if best is None else int(best.get("fs", 0)),
    }


def _append_export_ranking(summary_content: str, fs_v: int, ranking_context: dict | None) -> str:
    ctx = ranking_context or {}
    entries = list(ctx.get("entries") or [])
    if not entries:
        return summary_content

    cur = None
    for e in entries:
        if int(e.get("fs", -1)) == int(fs_v):
            cur = e
            break
    if cur is None:
        return summary_content

    best_fs = ctx.get("best_fs", None)
    summary_content += "\n=== RUN RANKING (SAMPLE-RATE COMPARISON) ===\n"
    summary_content += (
        "Method: run_ranking_score = avg_acoustic_score - boost_penalty - event_penalty - L/R_delta_penalty - dsp_penalty\n"
    )
    summary_content += (
        "Penalties: boost=1.5*dB over +1dB net boost max, event=0.5/event, L/R delta=0.25/score-point, dsp=quality-derived\n"
    )
    summary_content += (
        f"Current run: {int(fs_v)} Hz | rank #{int(cur.get('rank', 0))}/{len(entries)} | "
        f"run_ranking_score={_safe_float(cur.get('run_ranking_score'), 0.0):.3f}/100\n"
    )
    if best_fs is not None:
        summary_content += f"Recommended run: {int(best_fs)} Hz\n"
    summary_content += "Note: Run ranking score compares sample-rate outputs and is separate from automatic-mode Best rank score.\n"
    summary_content += (
        f"{'Rank':<6}{'FS (Hz)':<10}{'RunScore':<12}{'AvgScore':<10}{'DSPpen':<9}"
        f"{'MaxNetBoost':<13}{'Events':<8}{'L/R Delta':<10}\n"
    )
    summary_content += "-" * 78 + "\n"
    for e in entries:
        summary_content += (
            f"#{int(e.get('rank', 0)):<5}"
            f"{int(e.get('fs', 0)):<10}"
            f"{_safe_float(e.get('run_ranking_score'), 0.0):<12.3f}"
            f"{_safe_float(e.get('avg_score'), 0.0):<10.3f}"
            f"{_safe_float(e.get('dsp_penalty'), 0.0):<9.2f}"
            f"{_safe_float(e.get('max_net_boost_db'), 0.0):<13.2f}"
            f"{int(e.get('events_total', 0)):<8}"
            f"{_safe_float(e.get('lr_delta_score'), 0.0):<10.3f}\n"
        )
    return summary_content


def _export_winner_rank_score_from_metrics(best_metrics: dict | None) -> float:
    try:
        best_metrics = attach_official_rank_score(best_metrics or {})
    except Exception:
        return float("nan")
    rank_score = official_rank_score(best_metrics)
    if not math.isfinite(rank_score):
        return float("nan")
    return float(rank_score)
