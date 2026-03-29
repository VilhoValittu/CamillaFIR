from __future__ import annotations

from pywebio.output import put_html

from ..resources.i8n.camillafir_i18n import t


def plot_smoothing_label(psl) -> str:
    if isinstance(psl, str):
        if psl == "Psychoacoustic":
            return t("smooth_safe_reference")
        return psl
    return f"1/{int(psl)} octave"


def safe_float(v, default=float("nan")) -> float:
    try:
        x = float(v)
        if x == x and abs(x) != float("inf"):
            return float(x)
    except Exception:
        pass
    return float(default)


def safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def float3_or_na(v) -> str:
    x = safe_float(v, float("nan"))
    return f"{x:.3f}" if x == x else "n/a"


def boost_diag(st: dict) -> tuple[float, float, float]:
    boost_pre = float(st.get("boost_peak_db", 0.0) or 0.0)
    auto_gain = float(st.get("auto_global_gain_db", 0.0) or 0.0)
    net_boost = boost_pre + auto_gain
    return boost_pre, auto_gain, net_boost


def metric_cell(value, compare=None) -> dict:
    cmp = compare if compare is not None else value
    if cmp is None:
        cmp = "-"
    return {"render": value, "compare": str(cmp)}


def metric_row(label, left, right, *, left_compare=None, right_compare=None) -> dict:
    return {
        "label": str(label),
        "left": metric_cell(left, left_compare),
        "right": metric_cell(right, right_compare),
    }


def fmt_ai_match(ai: dict) -> dict:
    match = ai.get("match", None)
    if match is None:
        return metric_cell("n/a", "n/a")
    return metric_cell(f"{float(match):.1f}%", f"{float(match):.3f}")


def fmt_ai_score(ai: dict) -> dict:
    score = ai.get("score", None)
    if score is None:
        return metric_cell("n/a", "n/a")
    return metric_cell(f"{float(score):.3f}/100", f"{float(score):.3f}")


def fmt_tilt(st: dict, warn_thr: float = 1.5):
    tilt = st.get("tilt_slope_db_per_oct", None)
    if tilt is None:
        return "-"
    try:
        tilt = float(tilt)
        if abs(tilt) > warn_thr:
            return put_html(
                "<span title=\"Large broadband tilt detected during leveling, "
                "house curve not suitable for speaker in room.\">"
                f"{tilt:+.2f} dB/oct &#9888;"
                "</span>"
            )
        return f"{tilt:+.2f} dB/oct"
    except Exception:
        return "-"


def auto_choice_text(method) -> str:
    try:
        key = str(method or "").strip().lower()
    except Exception:
        key = ""
    mapping = {
        "adaptive": "adaptive (synthesized from measurements)",
        "cache_signature_hit": "cache hit",
        "cache_measurement": "measurement cache seed",
        "cache_signature": "signature cache seed",
        "trial_with_cache_wildcard": "cache wildcard winner",
        "top3x10_trials": "trial comparison",
        "top3x10_trials_rank_tie_composite": "trial comparison with tie-break",
        "fit_rms": "quick fit preselect",
    }
    return str(mapping.get(key, key or "unknown"))


def fmt_freq_window(win) -> str:
    try:
        if isinstance(win, (list, tuple)) and len(win) >= 2:
            lo = float(win[0])
            hi = float(win[1])
            if lo == lo and hi == hi and hi > lo:
                return f"{lo:.0f}-{hi:.0f} Hz"
    except Exception:
        pass
    return "n/a"


def stereo_link_mode_label(st: dict, *, stereo_link_enabled: bool) -> str:
    try:
        resolved = str(st.get("stereo_link_mode", "") or "").strip().lower()
    except Exception:
        resolved = ""
    try:
        requested = str(st.get("stereo_link_requested_mode", "") or "").strip().lower()
    except Exception:
        requested = ""
    if not resolved:
        return "Off" if not bool(stereo_link_enabled) else "On"
    resolved_label = {
        "shared": "Shared",
        "hybrid": "Hybrid",
    }.get(resolved, resolved.title())
    if requested == "auto":
        return f"Auto -> {resolved_label}"
    return resolved_label


def shared_window_label(st: dict) -> str:
    try:
        resolved = str(st.get("stereo_link_mode", "") or "").strip().lower()
    except Exception:
        resolved = ""
    win = st.get("stereo_link_shared_window", None)
    if resolved == "hybrid":
        return "Not used (Hybrid)"
    return fmt_freq_window(win)


def anchor_label(st: dict, *, stereo_link_enabled: bool) -> str:
    try:
        anchor = str(st.get("stereo_link_level_anchor_channel", "") or "").strip().lower()
    except Exception:
        anchor = ""
    try:
        resolved = str(st.get("stereo_link_mode", "") or "").strip().lower()
    except Exception:
        resolved = ""
    if anchor == "left":
        if resolved in {"shared", "hybrid"}:
            return "Left (right attenuated toward left)"
        return "Left"
    if anchor == "right":
        if resolved in {"shared", "hybrid"}:
            return "Right (left attenuated toward right)"
        return "Right"
    if not bool(stereo_link_enabled):
        return "-"
    return "-"


def phase_clamp_label(st: dict) -> str:
    try:
        lim = float((st or {}).get("phase_corr_clamp_deg", 0.0) or 0.0)
        bef = float((st or {}).get("phase_corr_max_before_deg", 0.0) or 0.0)
        clipped = bool((st or {}).get("phase_corr_clipped", False))
        if lim <= 0.0:
            return "-"
        if clipped:
            return f"max={bef:.1f} deg -> {lim:.1f} deg"
        return f"max={bef:.1f} deg (limit {lim:.1f} deg)"
    except Exception:
        return "-"


def xo_fc_wrapped_label(st: dict) -> str:
    try:
        if not isinstance(st, dict) or not st:
            return "-"
        xo_summary = str(st.get("xo_summary", "") or "")
        freqs = []
        for part in xo_summary.split(","):
            part = part.strip()
            if "Hz" in part:
                try:
                    freqs.append(float(part.split("Hz")[0].strip()))
                except Exception:
                    freqs.append(None)
        items = []
        for i in range(1, 6):
            key = f"xo{i}_dphi_wrapped_deg@fc"
            if key not in st:
                continue
            try:
                val = float(st.get(key))
            except Exception:
                continue
            if i <= len(freqs) and freqs[i - 1] is not None:
                label = f"{int(round(freqs[i - 1]))}Hz"
            else:
                label = f"XO{i}"
            items.append(f"{label}:{val:+.1f} deg")
        return " | ".join(items) if items else "-"
    except Exception:
        return "-"


def xo_fc_gd_label(st: dict) -> str:
    try:
        if not isinstance(st, dict) or not st:
            return "-"
        xo_summary = str(st.get("xo_summary", "") or "")
        freqs = []
        for part in xo_summary.split(","):
            part = part.strip()
            if "Hz" in part:
                try:
                    freqs.append(float(part.split("Hz")[0].strip()))
                except Exception:
                    freqs.append(None)
        items = []
        for i in range(1, 6):
            key = f"xo{i}_dgd_ms@fc"
            if key not in st:
                continue
            try:
                val = float(st.get(key))
            except Exception:
                continue
            if i <= len(freqs) and freqs[i - 1] is not None:
                label = f"{int(round(freqs[i - 1]))}Hz"
            else:
                label = f"XO{i}"
            items.append(f"{label}:{val:+.2f} ms")
        return " | ".join(items) if items else "-"
    except Exception:
        return "-"


def xo_phase_model_label(st: dict) -> str:
    try:
        summary = (st or {}).get("xo_summary", None)
        if summary is None or str(summary).strip() == "":
            return "-"
        return str(summary)
    except Exception:
        return "-"


def xo_diff_raw_label(st: dict) -> str:
    try:
        p = (st or {}).get("xo_diff_raw_max_phase_deg", None)
        pf = (st or {}).get("xo_diff_raw_max_phase_hz", None)
        pfc = (st or {}).get("xo_diff_raw_max_phase_xo_fc_hz", None)
        g = (st or {}).get("xo_diff_raw_max_gd_ms", None)
        gf = (st or {}).get("xo_diff_raw_max_gd_hz", None)
        gfc = (st or {}).get("xo_diff_raw_max_gd_xo_fc_hz", None)
        if p is None and g is None:
            return "-"
        parts = []
        if p is not None and pf is not None:
            parts.append(
                f"max delta phi {float(p):.1f} deg @ {float(pf):.0f} Hz"
                + (f" (XO {float(pfc):.0f} Hz)" if pfc is not None else "")
            )
        if g is not None and gf is not None:
            parts.append(
                f"max delta GD {float(g):.2f} ms @ {float(gf):.0f} Hz"
                + (f" (XO {float(gfc):.0f} Hz)" if gfc is not None else "")
            )
        return " | ".join(parts) if parts else "-"
    except Exception:
        return "-"


def xo_fc_gd_badge(st: dict) -> str:
    try:
        if not isinstance(st, dict) or not st:
            return ""
        vals = []
        for i in range(1, 6):
            key = f"xo{i}_dgd_ms@fc"
            if key not in st:
                continue
            try:
                vals.append(abs(float(st.get(key))))
            except Exception:
                pass
        if not vals:
            return ""
        worst = max(vals)
        if worst < 0.7:
            label, bg, fg, title = (
                "LOW",
                "rgba(46, 125, 50, 0.15)",
                "rgba(46, 125, 50, 1.0)",
                "Small XO delta GD at fc (typically subtle).",
            )
        elif worst < 1.5:
            label, bg, fg, title = (
                "MED",
                "rgba(255, 143, 0, 0.15)",
                "rgba(255, 143, 0, 1.0)",
                "Moderate XO delta GD at fc (often audible improvement with XO phase correction).",
            )
        else:
            label, bg, fg, title = (
                "HIGH",
                "rgba(211, 47, 47, 0.15)",
                "rgba(211, 47, 47, 1.0)",
                "Large XO delta GD at fc (aggressive crossover / lots of time smear).",
            )
        return (
            f"<span title='{title}' style='display:inline-block; margin-left:6px; padding:1px 6px; "
            f"border-radius:10px; font-size:11px; font-weight:600; background:{bg}; color:{fg}; "
            f"vertical-align:middle;'>{label}</span>"
        )
    except Exception:
        return ""


def hpf_diff_raw_label(st: dict) -> str:
    try:
        p = (st or {}).get("hpf_diff_raw_max_phase_deg", None)
        pf = (st or {}).get("hpf_diff_raw_max_phase_hz", None)
        g = (st or {}).get("hpf_diff_raw_max_gd_ms", None)
        gf = (st or {}).get("hpf_diff_raw_max_gd_hz", None)
        if p is None and g is None:
            return "-"
        parts = []
        if p is not None and pf is not None:
            parts.append(f"max delta phi {float(p):.1f} deg @ {float(pf):.0f} Hz")
        if g is not None and gf is not None:
            parts.append(f"max delta GD {float(g):.2f} ms @ {float(gf):.0f} Hz")
        return " | ".join(parts) if parts else "-"
    except Exception:
        return "-"


def hpf_model_label(st: dict) -> str:
    try:
        summary = (st or {}).get("hpf_summary", None)
        if summary is None or str(summary).strip() == "":
            return "-"
        return str(summary)
    except Exception:
        return "-"


def format_ir_window(data: dict) -> str:
    mode = str(data.get("ir_export_window_mode", "") or "").lower()
    if mode == "rew_asym":
        left = data.get("ir_window_left", None)
        right = data.get("ir_window_right", data.get("ir_window", None))
        try:
            if left is not None and right is not None:
                return f"Asymmetric (Left {float(left):.1f} ms, Right {float(right):.1f} ms)"
        except Exception:
            pass
        return "Asymmetric"
    return "Auto (adaptive)"


def mixed_blend_label(st: dict, key: str) -> str:
    try:
        val = (st or {}).get(key, None)
        if val is None:
            return "-"
        return f"{float(val):.1f} Hz"
    except Exception:
        return "-"


def gd_limiter_label(st: dict) -> str:
    try:
        state = st or {}
        enabled = bool(state.get("gd_limiter_enabled", state.get("gd_grad_limiter_enabled", False)))
        reason = str(state.get("gd_limiter_reason", state.get("gd_grad_limiter_reason", "unknown")) or "unknown")
        limit = state.get("gd_limiter_limit_ms_per_oct", state.get("gd_grad_limit_ms_per_oct", None))
        limit_txt = "n/a" if limit is None else f"{float(limit):.2f}"
        return f"{'ON' if enabled else 'OFF'} (reason={reason}, limit={limit_txt} ms/oct)"
    except Exception:
        return "n/a"


def gd_grad_max_label(st: dict) -> str:
    try:
        state = st or {}
        for key in (
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ):
            try:
                val = float(state.get(key, None))
                if val == val and abs(val) < float("inf"):
                    return f"{val:.2f} ms/oct"
            except Exception:
                continue
        return "n/a"
    except Exception:
        return "n/a"


__all__ = [
    "anchor_label",
    "auto_choice_text",
    "boost_diag",
    "float3_or_na",
    "fmt_ai_match",
    "fmt_ai_score",
    "fmt_freq_window",
    "fmt_tilt",
    "format_ir_window",
    "gd_grad_max_label",
    "gd_limiter_label",
    "hpf_diff_raw_label",
    "hpf_model_label",
    "metric_cell",
    "metric_row",
    "mixed_blend_label",
    "phase_clamp_label",
    "plot_smoothing_label",
    "safe_float",
    "safe_int",
    "shared_window_label",
    "stereo_link_mode_label",
    "xo_fc_gd_badge",
    "xo_fc_gd_label",
    "xo_fc_wrapped_label",
    "xo_diff_raw_label",
    "xo_phase_model_label",
]
