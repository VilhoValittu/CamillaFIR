import numpy as np

from .shared import (
    AUTO_MODE_ADAPTIVE_SHRINK_ENABLED,
    AUTO_MODE_ADAPTIVE_SHRINK_MAX,
    AUTO_MODE_ADAPTIVE_SHRINK_MIN,
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_GOAL_ROOM_SAFE,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ,
    AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    AUTO_MODE_PHASE2_PARETO_BOOST_EPS,
    AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS,
    AUTO_MODE_PHASE2_PARETO_PREPOST_EPS,
    AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS,
    AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB,
    AUTO_MODE_REFINE_TIEBREAK_ENABLE,
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    MAX_SAFE_BOOST,
    _auto_goal_norm,
    _auto_safe_float,
    _m,
)


def _auto_pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _tc_score(tc: dict | None) -> float:
    return float(
        _auto_safe_float(
            (tc or {}).get(
                "preselect_score",
                (tc or {}).get("fit_rms_db", float("inf")),
            ),
            float("inf"),
        )
    )


def _auto_build_winner_explanation(
    best_metrics: dict | None,
    prev_metrics: dict | None = None,
    *,
    phase_label: str | None = None,
    target_name: str | None = None,
) -> dict:
    phase_txt = str(phase_label or "").strip() or None
    target_txt = str(target_name or "").strip() or None
    fallback = {
        "summary": "Winner explanation unavailable.",
        "reasons": [],
        "deltas": {},
        "phase_label": phase_txt,
        "target_name": target_txt,
    }
    if not isinstance(best_metrics, dict) or not best_metrics:
        return dict(fallback)

    best = dict(best_metrics or {})
    prev = dict(prev_metrics or {}) if isinstance(prev_metrics, dict) else {}

    def _metric(src: dict | None, *keys: str) -> float:
        if not isinstance(src, dict):
            return float("nan")
        for key in keys:
            v = _auto_safe_float(src.get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                return float(v)
        return float("nan")

    def _fmt(val: float, *, unit: str = "", decimals: int = 2) -> str:
        if not np.isfinite(val):
            return "n/a"
        return f"{float(val):.{int(decimals)}f}{unit}"

    def _fmt_signed(val: float, *, unit: str = "", decimals: int = 2) -> str:
        if not np.isfinite(val):
            return "n/a"
        return f"{float(val):+.{int(decimals)}f}{unit}"

    deltas: dict[str, float] = {}
    delta_specs = (
        ("avg_score_delta", ("avg_score",)),
        ("rank_score_delta", ("rank_score",)),
        ("mode_ripple_delta", ("mode_ripple_db",)),
        ("boost_delta", ("max_net_boost_db",)),
        ("event_penalty_delta", ("event_penalty",)),
    )
    for out_key, keys in delta_specs:
        best_v = _metric(best, *keys)
        prev_v = _metric(prev, *keys)
        if np.isfinite(best_v) and np.isfinite(prev_v):
            deltas[str(out_key)] = float(best_v - prev_v)

    reasons: list[str] = []
    summary_bits: list[str] = []

    def _push(summary_bit: str | None, reason: str | None) -> None:
        sb = str(summary_bit or "").strip()
        rs = str(reason or "").strip()
        if sb:
            summary_bits.append(sb)
        if rs:
            reasons.append(rs)

    rank_score = _metric(best, "rank_score")
    rank_delta = deltas.get("rank_score_delta", float("nan"))
    if np.isfinite(rank_score):
        if np.isfinite(rank_delta) and float(rank_delta) > 0.0:
            _push(
                "improved rank score",
                f"Improved rank score to {_fmt(rank_score)} ({_fmt_signed(rank_delta)}).",
            )
        else:
            _push("rank score " + _fmt(rank_score), f"Rank score {_fmt(rank_score)}.")

    avg_score = _metric(best, "avg_score")
    avg_delta = deltas.get("avg_score_delta", float("nan"))
    if np.isfinite(avg_score):
        if np.isfinite(avg_delta) and float(avg_delta) > 0.0:
            _push(
                "improved average score",
                f"Improved average score to {_fmt(avg_score)} ({_fmt_signed(avg_delta)}).",
            )
        else:
            _push("average score " + _fmt(avg_score), f"Average score {_fmt(avg_score)}.")

    mode_ripple = _metric(best, "mode_ripple_db")
    mode_delta = deltas.get("mode_ripple_delta", float("nan"))
    if np.isfinite(mode_ripple):
        if np.isfinite(mode_delta) and float(mode_delta) < 0.0:
            _push(
                "reduced mode ripple",
                f"Reduced mode ripple to {_fmt(mode_ripple, unit=' dB')} ({_fmt_signed(mode_delta, unit=' dB')}).",
            )
        else:
            _push(
                "mode ripple " + _fmt(mode_ripple, unit=" dB"),
                f"Mode ripple {_fmt(mode_ripple, unit=' dB')}.",
            )

    net_boost = _metric(best, "max_net_boost_db")
    boost_delta = deltas.get("boost_delta", float("nan"))
    if np.isfinite(net_boost):
        if np.isfinite(boost_delta) and float(boost_delta) < 0.0:
            _push(
                "reduced net boost",
                f"Reduced net boost to {_fmt(net_boost, unit=' dB')} ({_fmt_signed(boost_delta, unit=' dB')}).",
            )
        elif float(net_boost) <= float(MAX_SAFE_BOOST) * 0.5:
            _push(
                "kept net boost controlled",
                f"Kept net boost controlled at {_fmt(net_boost, unit=' dB')}.",
            )
        else:
            _push(
                "net boost " + _fmt(net_boost, unit=" dB"),
                f"Net boost {_fmt(net_boost, unit=' dB')}.",
            )

    event_penalty = _metric(best, "event_penalty")
    event_delta = deltas.get("event_penalty_delta", float("nan"))
    if np.isfinite(event_penalty):
        if np.isfinite(event_delta) and float(event_delta) < 0.0:
            _push(
                "reduced event penalty",
                f"Reduced event penalty to {_fmt(event_penalty)} ({_fmt_signed(event_delta)}).",
            )
        elif abs(float(event_penalty)) <= 1e-9:
            _push("avoided event penalty", "Avoided event penalty.")
        else:
            _push("event penalty " + _fmt(event_penalty), f"Event penalty {_fmt(event_penalty)}.")

    focus_rms = _metric(best, "focus_rms_db")
    if np.isfinite(focus_rms):
        _push("focus RMS " + _fmt(focus_rms, unit=" dB"), f"Focus RMS {_fmt(focus_rms, unit=' dB')}.")

    fit_rms = _metric(best, "fit_rms_db", "mode_fit_rms_db")
    if np.isfinite(fit_rms):
        _push("target-fit RMS " + _fmt(fit_rms, unit=" dB"), f"Target-fit RMS {_fmt(fit_rms, unit=' dB')}.")

    rms_20_200 = _metric(best, "rms_20_200", "realized_rms_20_200_db")
    if np.isfinite(rms_20_200):
        _push(
            "LF RMS 20-200 Hz " + _fmt(rms_20_200, unit=" dB"),
            f"LF RMS (20-200 Hz) {_fmt(rms_20_200, unit=' dB')}.",
        )

    if not reasons:
        return dict(fallback)

    summary_top = list(summary_bits[:3])
    if not summary_top:
        summary = "Winner selected based on available auto-mode metrics."
    elif len(summary_top) == 1:
        summary = f"Won on {summary_top[0]}."
    elif len(summary_top) == 2:
        summary = f"Won on {summary_top[0]} and {summary_top[1]}."
    else:
        summary = f"Won on {summary_top[0]}, {summary_top[1]}, and {summary_top[2]}."

    return {
        "summary": str(summary),
        "reasons": list(reasons),
        "deltas": dict(deltas),
        "phase_label": phase_txt,
        "target_name": target_txt,
    }


def _auto_target_result_rank_key(item: dict | None) -> tuple:
    bm = dict((item or {}).get("best_metrics", {}) or {})
    return (
        -_auto_safe_float(bm.get("rank_score"), 0.0),
        -_auto_safe_float((item or {}).get("avg_rank_score"), 0.0),
        _auto_safe_float((item or {}).get("fit_rms_db"), 1e9),
    )


def _auto_target_result_mode_ripple(item: dict | None) -> float:
    bm = dict((item or {}).get("best_metrics", {}) or {})
    v = _auto_safe_float(bm.get("mode_ripple_db", float("nan")), float("nan"))
    if np.isfinite(v):
        return float(v)
    return float("inf")


def _auto_target_mildness_index(hc_name: str) -> int:
    name = str(hc_name or "").strip()
    if not name:
        return 10_000
    ladders = (
        ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
        ("BK_Light", "BK_Medium", "BK_Strong"),
    )
    for ladder in ladders:
        if name in ladder:
            return int(ladder.index(name))
    return 10_000


def _auto_target_result_tie_key(item: dict | None) -> tuple:
    it = dict(item or {})
    return (
        -_auto_safe_float(it.get("avg_rank_score"), 0.0),
        _auto_target_result_mode_ripple(it),
        _auto_safe_float(it.get("boost_penalty", 0.0), 0.0),
        _auto_safe_float(it.get("fit_rms_db"), 1e9),
        _tc_score(it),
        _auto_target_mildness_index(str(it.get("hc_mode", "") or "").strip()),
        str(it.get("hc_mode", "") or "").strip(),
    )


def _auto_rank_key(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_room_safe(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
    )


def _auto_mode_ripple_for_pareto(metrics: dict | None) -> float:
    for k in ("mode_ripple_db", "focus_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_realized_rms_20_200_for_pareto(metrics: dict | None) -> float:
    v = _m(metrics, "realized_rms_20_200_db", float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    return float("inf")


def _auto_rank_key_low_ripple(metrics: dict) -> tuple:
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    ripple_fallback = _auto_safe_float(metrics.get("focus_ripple_db"), float("inf"))
    realized_lf = _auto_realized_rms_20_200_for_pareto(metrics)
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        mode_ripple if np.isfinite(mode_ripple) else ripple_fallback,
        realized_lf,
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("mixed_freq_penalty"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_flat(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_realized_rms_20_200_for_pareto(metrics),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
    )


def _auto_rank_key_acoustic(metrics: dict) -> tuple:
    return _auto_rank_key_flat(metrics)


def _auto_rank_key_hybrid(metrics: dict) -> tuple:
    return _auto_rank_key_low_ripple(metrics)


def _auto_hybrid_mixed_freq_penalty(
    preset: dict | None,
    *,
    base_data: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> float:
    if _auto_goal_norm(goal) != AUTO_MODE_GOAL_LOW_RIPPLE:
        return 0.0

    p = dict(base_data or {})
    p.update(dict(preset or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    if "mixed" not in ft:
        return 0.0
    if not bool(p.get("bass_first_ai", True)):
        return 0.0

    mixed_freq = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
    if not np.isfinite(mixed_freq):
        return 0.0
    pen = max(
        0.0,
        (float(mixed_freq) - float(AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ))
        / float(max(1e-6, AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ)),
    )
    return float(np.clip(pen, 0.0, 2.5))


def _auto_apply_goal_tiebreak_metrics(
    metrics: dict,
    *,
    preset: dict | None,
    base_data: dict | None,
    goal: str,
) -> dict:
    out = dict(metrics or {})
    out["mixed_freq_penalty"] = float(
        _auto_hybrid_mixed_freq_penalty(preset, base_data=base_data, goal=goal)
    )
    return out


def _auto_build_refine_profile(
    *,
    base_data: dict,
    phase1_top: list,
) -> dict:
    mixed_vals = []
    tdc_vals = []
    for it in (phase1_top or []):
        p = dict(it.get("preset", {}) or {})
        mf = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
        td = _auto_safe_float(p.get("tdc_strength", float("nan")), float("nan"))
        if np.isfinite(mf):
            mixed_vals.append(float(mf))
        if np.isfinite(td):
            tdc_vals.append(float(td))

    if not mixed_vals:
        mixed_center = 120.0
        focus_lo = float(max(20.0, float(mixed_center) - 70.0))
        focus_hi = float(min(220.0, float(mixed_center) + 50.0))
        bf_hi = float("nan")
        if bool(base_data.get("bass_first_ai", True)):
            bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
            if np.isfinite(bf_hi):
                focus_hi = min(float(focus_hi), float(bf_hi))
        focus_lo = float(np.clip(focus_lo, 20.0, 200.0))
        focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
        if focus_hi <= focus_lo:
            focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
        if focus_hi <= focus_lo:
            focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))
        return {
            "mixed_center": float(mixed_center),
            "mixed_span": 60.0,
            "focus_lo": float(focus_lo),
            "focus_hi": float(focus_hi),
            "tdc_lo": 45.0,
            "tdc_hi": 70.0,
        }

    mixed_center = float(np.median(mixed_vals))
    mixed_spread = float(np.std(mixed_vals)) if len(mixed_vals) > 1 else 20.0
    mixed_span = float(np.clip(mixed_spread * 1.5, 25.0, 80.0))
    focus_lo = float(max(20.0, float(mixed_center) - 70.0))
    focus_hi = float(min(220.0, float(mixed_center) + 50.0))
    bf_hi = float("nan")
    if bool(base_data.get("bass_first_ai", True)):
        bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
    focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
    if np.isfinite(bf_hi):
        focus_hi = min(float(focus_hi), float(bf_hi))
    if focus_hi <= focus_lo:
        focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
    if focus_hi <= focus_lo:
        focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))

    tdc_center = float(np.median(tdc_vals)) if tdc_vals else 60.0
    return {
        "mixed_center": mixed_center,
        "mixed_span": mixed_span,
        "focus_lo": float(focus_lo),
        "focus_hi": float(focus_hi),
        "tdc_lo": float(np.clip(tdc_center - 12.0, 35.0, 80.0)),
        "tdc_hi": float(np.clip(tdc_center + 12.0, 40.0, 85.0)),
    }


def _auto_goal_uses_local_refine(goal: str | None) -> bool:
    g = _auto_goal_norm(goal)
    return bool(
        g in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
        )
    )


def _auto_rank_key_goal(metrics: dict, goal: str = AUTO_MODE_GOAL_DEFAULT) -> tuple:
    g = _auto_goal_norm(goal)
    if g == AUTO_MODE_GOAL_FLAT:
        return _auto_rank_key_flat(metrics)
    if g == AUTO_MODE_GOAL_ROOM_SAFE:
        return _auto_rank_key_room_safe(metrics)
    if g == AUTO_MODE_GOAL_LOW_RIPPLE:
        return _auto_rank_key_low_ripple(metrics)
    return _auto_rank_key(metrics)


def _auto_is_better_refine(
    new_metrics: dict,
    best_metrics: dict,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    *,
    return_reason: bool = False,
) -> bool | tuple[bool, str]:
    new_m = dict(new_metrics or {})
    best_m = dict(best_metrics or {})
    rank_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RANK_EPS, 0.20)))
    ripple_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS, 0.02)))
    mode_guard_gain = float(
        max(0.0, _auto_safe_float(AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB, 0.06))
    )
    new_rank_raw = _auto_safe_float(new_m.get("rank_score"), 0.0)
    best_rank_raw = _auto_safe_float(best_m.get("rank_score"), 0.0)
    raw_rank_diff = float(new_rank_raw - best_rank_raw)
    new_rank_ref = _auto_safe_float(new_m.get("rank_score_refine", new_rank_raw), new_rank_raw)
    best_rank_ref = _auto_safe_float(best_m.get("rank_score_refine", best_rank_raw), best_rank_raw)
    ref_rank_diff = float(new_rank_ref - best_rank_ref)
    new_mode_ripple = _auto_safe_float(new_m.get("mode_ripple_db"), float("nan"))
    best_mode_ripple = _auto_safe_float(best_m.get("mode_ripple_db"), float("nan"))
    mode_pair_ok = bool(np.isfinite(new_mode_ripple) and np.isfinite(best_mode_ripple))

    if ref_rank_diff > 1e-9:
        if abs(raw_rank_diff) <= rank_eps and mode_pair_ok:
            mode_improve = float(best_mode_ripple - new_mode_ripple)
            if mode_improve > ripple_eps:
                new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                boost_rise = float(new_boost - best_boost)
                if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                    out = (False, "mode_guard")
                else:
                    out = (True, "mode_ripple")
            else:
                out = (True, "rank_refine")
        else:
            out = (True, "rank_refine")
    elif ref_rank_diff < -1e-9:
        if abs(raw_rank_diff) <= rank_eps and mode_pair_ok and (float(new_mode_ripple - best_mode_ripple) > ripple_eps):
            out = (False, "mode_ripple")
        else:
            out = (False, "rank_refine")
    else:
        out = (False, "rank_tie")
        if bool(AUTO_MODE_REFINE_TIEBREAK_ENABLE):
            if mode_pair_ok:
                mode_improve = float(best_mode_ripple - new_mode_ripple)
                if mode_improve > ripple_eps:
                    new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                    best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                    boost_rise = float(new_boost - best_boost)
                    if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                        out = (False, "mode_guard")
                    else:
                        out = (True, "mode_ripple")
                elif float(new_mode_ripple - best_mode_ripple) > ripple_eps:
                    out = (False, "mode_ripple")

            if out[1] == "rank_tie":
                new_ripple = _auto_safe_float(new_m.get("focus_ripple_db"), float("nan"))
                best_ripple = _auto_safe_float(best_m.get("focus_ripple_db"), float("nan"))
                if np.isfinite(new_ripple) and np.isfinite(best_ripple):
                    if float(best_ripple - new_ripple) > ripple_eps:
                        out = (True, "focus_ripple")
                    elif float(new_ripple - best_ripple) > ripple_eps:
                        out = (False, "focus_ripple")

        if out[1] == "rank_tie":
            out = (
                bool(_auto_rank_key_goal(new_m, goal) < _auto_rank_key_goal(best_m, goal)),
                "goal_key",
            )
    return out if bool(return_reason) else bool(out[0])


def _auto_prepost_lr_for_pareto(metrics: dict | None) -> tuple[float, float, float]:
    m = dict(metrics or {})
    l = _m(m, "ir_pre_post_energy_ratio_l", float("nan"))
    r = _m(m, "ir_pre_post_energy_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(m, "ir_pre_post_ratio_l", float("nan"))
    if not np.isfinite(r):
        r = _m(m, "ir_pre_post_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(dict(m.get("dsp_dbg_l", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(r):
        r = _m(dict(m.get("dsp_dbg_r", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(l):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            l = float(global_v)
    if not np.isfinite(r):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            r = float(global_v)

    vals = [float(v) for v in (l, r) if np.isfinite(v)]
    mx = float(max(vals)) if vals else float("inf")
    return (
        float(l) if np.isfinite(l) else float("nan"),
        float(r) if np.isfinite(r) else float("nan"),
        float(mx),
    )


def _auto_prepost_for_pareto(metrics: dict | None) -> float:
    _, _, mx = _auto_prepost_lr_for_pareto(metrics)
    if np.isfinite(mx):
        return float(max(0.0, mx))
    return float("inf")


def _auto_ripple_metric_for_gate(metrics: dict | None) -> float:
    for k in ("focus_ripple_db", "mode_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_gate_threshold(values: list[float], keep_fraction: float) -> float:
    vals = [float(v) for v in (values or []) if np.isfinite(v)]
    if not vals:
        return float("inf")
    kf = float(np.clip(_auto_safe_float(keep_fraction, 1.0), 0.05, 1.0))
    vals = sorted(vals)
    idx = int(np.floor((len(vals) - 1) * kf))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def _auto_phase2_hard_gate_pool(
    pool: list[dict],
    *,
    min_keep: int = AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    keep_event_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    keep_ripple_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    fallback_to_rank: bool = AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
) -> tuple[list[dict], float, float]:
    if not isinstance(pool, list) or not pool:
        return [], float("inf"), float("inf")
    n_in = int(len(pool))
    min_keep = int(max(1, min_keep))
    if n_in <= (min_keep + 2):
        return [dict(x or {}) for x in pool], float("inf"), float("inf")

    ev = []
    rp = []
    for it in pool:
        m = dict((it or {}).get("metrics", {}) or {})
        ev.append(_m(m, "events_severity", float("nan")))
        rp.append(_auto_ripple_metric_for_gate(m))

    ev_thr = _auto_gate_threshold(ev, float(keep_event_fraction))
    rp_thr = _auto_gate_threshold(rp, float(keep_ripple_fraction))
    gated = []
    gated_or = []
    for it in pool:
        m = dict((it or {}).get("metrics", {}) or {})
        ev_i = _m(m, "events_severity", float("inf"))
        rp_i = _auto_ripple_metric_for_gate(m)
        ok_ev = bool(np.isfinite(ev_i) and float(ev_i) <= float(ev_thr))
        ok_rp = bool(np.isfinite(rp_i) and float(rp_i) <= float(rp_thr))
        if ok_ev and ok_rp:
            gated.append(dict(it or {}))
        if ok_ev or ok_rp:
            gated_or.append(dict(it or {}))

    if len(gated) >= min_keep:
        return gated, float(ev_thr), float(rp_thr)
    if len(gated_or) >= min_keep:
        return gated_or, float(ev_thr), float(rp_thr)
    if bool(fallback_to_rank):
        kept = sorted(
            [dict(x or {}) for x in pool],
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "rank_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )[:min_keep]
        return kept, float(ev_thr), float(rp_thr)
    kept = gated_or or gated or [dict(x or {}) for x in pool]
    return kept, float(ev_thr), float(rp_thr)


def _auto_adaptive_shrink_factor(
    phase1_top: list[dict],
    *,
    base_shrink: float,
    plateau_hit: bool,
) -> float:
    if not bool(AUTO_MODE_ADAPTIVE_SHRINK_ENABLED):
        return float(base_shrink)
    base = float(np.clip(_auto_safe_float(base_shrink, 0.35), 0.05, 1.0))
    if not isinstance(phase1_top, list) or len(phase1_top) < 2:
        if bool(plateau_hit):
            return float(np.clip(base * 0.85, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))
        return float(np.clip(base, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))

    mixed = []
    tdc = []
    fdw = []
    reg = []
    for it in phase1_top[:4]:
        p = dict((it or {}).get("preset", {}) or {})
        mixed.append(_auto_safe_float(p.get("mixed_freq", float("nan")), float("nan")))
        tdc.append(_auto_safe_float(p.get("tdc_strength", float("nan")), float("nan")))
        fdw.append(_auto_safe_float(p.get("fdw_cycles", float("nan")), float("nan")))
        reg.append(_auto_safe_float(p.get("reg_strength", float("nan")), float("nan")))

    def _spread(vals: list[float]) -> float:
        vv = [float(v) for v in vals if np.isfinite(v)]
        if len(vv) < 2:
            return 0.0
        vv = sorted(vv)
        return float(vv[-1] - vv[0])

    spread_score = 0.0
    spread_score += _spread(mixed) / 80.0
    spread_score += _spread(tdc) / 15.0
    spread_score += _spread(fdw) / 3.0
    spread_score += _spread(reg) / 20.0
    if spread_score <= 0.35:
        mul = 0.75
    elif spread_score <= 0.70:
        mul = 0.85
    elif spread_score <= 1.10:
        mul = 0.95
    else:
        mul = 1.05
    if bool(plateau_hit):
        mul *= 0.90
    out = float(base * mul)
    return float(np.clip(out, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))


def _pareto_dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    le_all = True
    lt_any = False
    for ai, bi in zip(a, b):
        if ai > bi:
            le_all = False
            break
        if ai < bi:
            lt_any = True
    return bool(le_all and lt_any)


def _auto_phase2_pareto_vector(metrics: dict | None) -> tuple[float, float, float, float, float]:
    avg = _m(metrics, "avg_score", float("nan"))
    neg_avg = -float(avg) if np.isfinite(avg) else float("inf")
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    rms_20_200 = _auto_realized_rms_20_200_for_pareto(metrics)
    net_boost = _m(metrics, "max_net_boost_db", float("nan"))
    net_boost = float(net_boost) if np.isfinite(net_boost) else float("inf")
    prepost = _auto_prepost_for_pareto(metrics)
    return (float(neg_avg), float(mode_ripple), float(rms_20_200), float(net_boost), float(prepost))


def _auto_phase2_pareto_front(pool: list[dict]) -> list[dict]:
    front = []
    if not isinstance(pool, list) or not pool:
        return front
    vectors = [_auto_phase2_pareto_vector(dict(it.get("metrics", {}) or {})) for it in pool]
    for i, cand in enumerate(pool):
        dominated = False
        for j, other in enumerate(pool):
            if i == j:
                continue
            if _pareto_dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            front.append(dict(cand or {}))
    return front


def _auto_phase2_pick_pareto_winner(
    front: list[dict],
    pool: list[dict],
    *,
    acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
) -> dict | None:
    def _lex_better(a: dict, b: dict) -> bool:
        ma = dict(a.get("metrics", {}) or {})
        mb = dict(b.get("metrics", {}) or {})
        avg_a = _m(ma, "avg_score", float("-inf"))
        avg_b = _m(mb, "avg_score", float("-inf"))
        if float(avg_a) > float(avg_b):
            return True
        if float(avg_a) < float(avg_b):
            return False

        prepost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_PREPOST_EPS, 0.002)))
        prepost_a = _auto_prepost_for_pareto(ma)
        prepost_b = _auto_prepost_for_pareto(mb)
        if float(prepost_a) < float(prepost_b) - float(prepost_eps):
            return True
        if float(prepost_b) < float(prepost_a) - float(prepost_eps):
            return False

        mode_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS, 0.005)))
        mode_a = _auto_mode_ripple_for_pareto(ma)
        mode_b = _auto_mode_ripple_for_pareto(mb)
        if float(mode_a) < float(mode_b) - float(mode_eps):
            return True
        if float(mode_b) < float(mode_a) - float(mode_eps):
            return False

        rms_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS, 0.003)))
        rms_a = _auto_realized_rms_20_200_for_pareto(ma)
        rms_b = _auto_realized_rms_20_200_for_pareto(mb)
        if float(rms_a) < float(rms_b) - float(rms_eps):
            return True
        if float(rms_b) < float(rms_a) - float(rms_eps):
            return False

        boost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_BOOST_EPS, 0.02)))
        boost_a = _m(ma, "max_net_boost_db", float("inf"))
        boost_b = _m(mb, "max_net_boost_db", float("inf"))
        if float(boost_a) < float(boost_b) - float(boost_eps):
            return True
        if float(boost_b) < float(boost_a) - float(boost_eps):
            return False
        return bool(_auto_rank_key(ma) < _auto_rank_key(mb))

    front_list = [dict(x or {}) for x in (front or []) if isinstance(x, dict)]
    pool_list = [dict(x or {}) for x in (pool or []) if isinstance(x, dict)]
    if not front_list:
        return None

    avg_vals = [
        _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
        for it in pool_list
    ]
    avg_vals = [float(v) for v in avg_vals if np.isfinite(v)]
    best_avg = max(avg_vals) if avg_vals else float("nan")
    drop = float(max(0.0, _auto_safe_float(acoustic_drop, AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP)))

    acceptable: list[dict] = []
    if np.isfinite(best_avg):
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg) and float(avg) >= float(best_avg) - float(drop):
                acceptable.append(dict(it))
    choose_from = acceptable

    if not choose_from:
        front_with_avg = []
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg):
                front_with_avg.append((float(avg), dict(it)))
        if front_with_avg:
            front_with_avg = sorted(
                front_with_avg,
                key=lambda t: (
                    -float(t[0]),
                    _auto_rank_key(dict((t[1] or {}).get("metrics", {}) or {})),
                ),
            )
            choose_from = [dict(front_with_avg[0][1])]
        else:
            choose_from = list(front_list)

    if choose_from:
        winner = dict(choose_from[0])
        for cand in choose_from[1:]:
            cand_d = dict(cand or {})
            if _lex_better(cand_d, winner):
                winner = cand_d
        return dict(winner)

    if pool_list:
        pool_sorted = sorted(
            pool_list,
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "avg_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )
        return dict(pool_sorted[0])
    return None


def _auto_select_best_scored(scored: list[dict]) -> dict | None:
    pool = [dict(x or {}) for x in (scored or []) if isinstance(x, dict)]
    if not pool:
        return None

    select_kind = str(pool[0].get("_auto_select_kind", "rank_metrics") or "rank_metrics").strip().lower()
    if select_kind == "target_curve":
        rank_tie_eps = float(
            max(
                0.0,
                _auto_safe_float(
                    pool[0].get("_target_rank_tie_eps", AUTO_MODE_TARGET_BEST_RANK_TIE_EPS),
                    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
                ),
            )
        )
        winner = dict(sorted(pool, key=_auto_target_result_rank_key)[0])
        winner_rank = _auto_safe_float(
            dict(winner.get("best_metrics", {}) or {}).get("rank_score"),
            0.0,
        )
        winner["_auto_selection_method"] = "top3x10_trials"
        near_top = []
        for it in pool:
            it_rank = _auto_safe_float(
                dict(it.get("best_metrics", {}) or {}).get("rank_score"),
                0.0,
            )
            if abs(float(winner_rank) - float(it_rank)) < rank_tie_eps:
                near_top.append(dict(it))
        if len(near_top) >= 2:
            winner = dict(sorted(near_top, key=_auto_target_result_tie_key)[0])
            winner["_auto_selection_method"] = "top3x10_trials_rank_tie_composite"
        if bool(winner.get("from_cache_wildcard", False)):
            winner["_auto_selection_method"] = "trial_with_cache_wildcard"
        return winner

    if select_kind == "phase2_pareto":
        acoustic_drop = float(
            max(
                0.0,
                _auto_safe_float(
                    pool[0].get("_phase2_pareto_acoustic_drop", AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP),
                    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
                ),
            )
        )
        front = _auto_phase2_pareto_front(pool)
        winner = _auto_phase2_pick_pareto_winner(
            front,
            pool,
            acoustic_drop=float(acoustic_drop),
        )
        if isinstance(winner, dict):
            return dict(winner)

    return dict(
        sorted(
            pool,
            key=lambda it: _auto_rank_key(dict(it.get("metrics", {}) or {})),
        )[0]
    )


def _auto_reject(metrics: dict, st_l: dict | None, st_r: dict | None, goal: str) -> bool:
    g = _auto_goal_norm(goal)
    if g != AUTO_MODE_GOAL_FLAT:
        return False
    if _auto_safe_float(metrics.get("max_net_boost_db"), 0.0) > float(MAX_SAFE_BOOST):
        return True

    ratio_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    gd_keys = (
        "gd_grad_limiter_after_max_ms_per_oct",
        "gd_grad_limiter_before_max_ms_per_oct",
        "gd_limiter_max_grad_ms_per_oct",
        "gd_grad_limiter_max_grad_ms_per_oct",
        "gd_limiter_max_grad_after_ms_per_oct",
        "gd_grad_limiter_max_grad_after_ms_per_oct",
        "gd_limiter_max_grad_before_ms_per_oct",
        "gd_grad_limiter_max_grad_before_ms_per_oct",
    )
    for st in (dict(st_l or {}), dict(st_r or {})):
        pre_suspect = bool(st.get("pre_energy_metric_suspect", False))
        if not pre_suspect:
            ratio = _auto_pick_metric(st, ratio_keys, nonneg=True)
            if ratio is not None and float(ratio) > 0.05:
                return True
        gd_grad = _auto_pick_metric(st, gd_keys, abs_value=True, nonneg=True)
        if gd_grad is not None and float(gd_grad) > 45.0:
            return True
    return False
