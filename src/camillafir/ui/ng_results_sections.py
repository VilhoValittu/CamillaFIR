"""NiceGUI results rendering after a DSP run.

Replaces camillafir_ui._render_results() + results_sections.py.

render_results() has the same signature as _render_results() so ng_bridge.py
can call it without changes to workflow code.
"""
from __future__ import annotations

import html
import logging
import math
import time
from typing import Any

from ..resources.i8n.camillafir_i18n import t
from . import camillafir_plot as plots
from . import ui_state
from .results_formatters import (
    anchor_label,
    boost_diag,
    fmt_ai_match,
    fmt_ai_score,
    fmt_freq_window,
    fmt_tilt,
    format_ir_window,
    gd_grad_max_label,
    gd_limiter_label,
    hpf_diff_raw_label,
    hpf_model_label,
    metric_row,
    mixed_blend_label,
    phase_clamp_label,
    plot_smoothing_label,
    safe_float,
    shared_window_label,
    stereo_link_mode_label,
    xo_fc_gd_label,
)

logger = logging.getLogger("CamillaFIR")


def _format_recommended_xo_hz(value: float) -> str:
    hz = float(value)
    if math.isclose(hz, round(hz), abs_tol=1e-6):
        return f"{hz:.0f} Hz"
    return f"{hz:.1f} Hz"


def render_results(
    data,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    l_imp_f,
    r_imp_f,
    l_st_f,
    r_st_f,
    fname,
    zip_buffer,
    *,
    dash_html_l=None,
    dash_html_r=None,
    run_started_at=None,
    perf_stats=None,
    per_fs_stats=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
    sub_imp_f=None,
    sub_meas_f=None,
    sub_st_f=None,
) -> None:
    from nicegui import ui
    from .ng_run_section import get_progress_element, get_results_container  # noqa: PLC0415

    if run_started_at is not None:
        try:
            elapsed = max(0.0, float(time.perf_counter() - float(run_started_at)))
            ui_state.update_status(f"{t('stat_plot')} | {elapsed:.1f} s")
        except Exception:
            ui_state.update_status(t("stat_plot"))
    else:
        ui_state.update_status(t("stat_plot"))

    prog = get_progress_element()
    if prog is not None:
        try:
            prog.set_value(0.8)
        except Exception:
            pass

    container = get_results_container()
    if container is None:
        logger.warning("render_results: results container not available")
        return

    container.clear()

    with container:
        if l_st_f is None or r_st_f is None:
            ui.label(t("results_error_no_results")).classes("text-red-500 font-bold")
            return

        psl_str = plot_smoothing_label(data.get("plot_smoothing_level", "Psychoacoustic"))

        _render_run_overview(data=data, l_st_f=l_st_f, r_st_f=r_st_f)
        _render_auto_diagnostics(data=data)
        _render_bass_integration(data=data)
        _update_crossover_recommendation_label(data)
        _render_ir_alignment(l_st_f=l_st_f)
        _render_dsp_quality(data=data, l_st_f=l_st_f, r_st_f=r_st_f, psl_str=psl_str)
        _render_plots_and_export(
            data=data,
            f_l=f_l,
            m_l=m_l,
            p_l=p_l,
            f_r=f_r,
            m_r=m_r,
            p_r=p_r,
            l_imp_f=l_imp_f,
            r_imp_f=r_imp_f,
            l_st_f=l_st_f,
            r_st_f=r_st_f,
            fname=fname,
            zip_buffer=zip_buffer,
            dash_html_l=dash_html_l,
            dash_html_r=dash_html_r,
            saved_filters_dir=saved_filters_dir,
            auto_cache_path=auto_cache_path,
            optuna_storage_path=optuna_storage_path,
            sub_imp_f=sub_imp_f,
            sub_meas_f=sub_meas_f,
            sub_st_f=sub_st_f,
        )

    done_msg = t("done_msg")
    done_status = t("stat_done")
    if saved_filters_dir:
        try:
            done_status = done_status.format(path=str(saved_filters_dir))
        except Exception:
            done_status = f"{done_status} {saved_filters_dir}"

    if run_started_at is not None:
        try:
            total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
            ui_state.update_status_notices(summary_text=done_msg, info_text="")
            ui_state.update_status(f"{done_status} | {total_s:.1f} s")
        except Exception:
            ui_state.update_status_notices(summary_text=done_msg, info_text="")
            ui_state.update_status(done_status)
    else:
        ui_state.update_status_notices(summary_text=done_msg, info_text="")
        ui_state.update_status(done_status)

    if prog is not None:
        try:
            prog.set_value(1.0)
            prog.set_text_color("light-green-4")
        except Exception:
            pass

    if l_st_f is not None and r_st_f is not None:
        try:
            l_ai = plots.calc_ai_summary_from_stats(l_st_f)
            r_ai = plots.calc_ai_summary_from_stats(r_st_f)
            l_score = l_ai.get("score")
            r_score = r_ai.get("score")
            l_match = l_ai.get("match")
            r_match = r_ai.get("match")
            l_conf = float(l_st_f.get("avg_confidence") or 0.0)
            r_conf = float(r_st_f.get("avg_confidence") or 0.0)
            ui_state.set_last_run_info(
                {
                    "score": (float(l_score) + float(r_score)) / 2.0
                    if (l_score is not None and r_score is not None)
                    else None,
                    "match": (float(l_match) + float(r_match)) / 2.0
                    if (l_match is not None and r_match is not None)
                    else None,
                    "conf": (l_conf + r_conf) / 2.0,
                }
            )
        except Exception:
            logger.debug("set_last_run_info failed", exc_info=True)


def _esc(v: Any) -> str:
    return html.escape(str(v) if v is not None else "-")


def _metric_table_html(rows: list[dict]) -> str:
    """Build an HTML table from metric_row() / fmt_* dicts."""
    shared: list[tuple[str, str]] = []
    stereo: list[tuple[str, str, str]] = []

    for row in rows:
        left = dict(row.get("left", {}) or {})
        right = dict(row.get("right", {}) or {})
        label = str(row.get("label", "") or "")
        if str(left.get("compare", "")) == str(right.get("compare", "")):
            shared.append((label, left.get("render", "-")))
        else:
            stereo.append((label, left.get("render", "-"), right.get("render", "-")))

    parts = []
    if shared:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_value'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(lbl)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{val}</td></tr>"
                for lbl, val in shared
            )
            + "</tbody></table>"
        )
    if stereo:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_left_short'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_right_short'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(lbl)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{lv}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{rv}</td></tr>"
                for lbl, lv, rv in stereo
            )
            + "</tbody></table>"
        )
    return "\n".join(parts)


def _section(title: str, rows: list[dict], summary_lines: list[str] | None = None) -> None:
    """Render a collapsible metric section."""
    from nicegui import ui  # noqa: PLC0415

    with ui.expansion(title).classes("w-full"):
        for line in list(summary_lines or []):
            if line:
                ui.markdown(str(line))
        table_html = _metric_table_html(rows)
        if table_html:
            ui.html(table_html)


def _render_run_overview(*, data: dict, l_st_f: dict, r_st_f: dict) -> None:
    l_ai = plots.calc_ai_summary_from_stats(l_st_f)
    r_ai = plots.calc_ai_summary_from_stats(r_st_f)
    l_score = float(l_ai.get("score") or 0.0)
    r_score = float(r_ai.get("score") or 0.0)
    avg_pred = (l_score + r_score) / 2.0
    l_match = l_ai.get("match")
    r_match = r_ai.get("match")
    avg_match = 0.0 if (l_match is None or r_match is None) else (float(l_match) + float(r_match)) / 2.0

    stereo_link_enabled = bool(data.get("stereo_link", False))
    l_window_used = l_st_f.get("stereo_link_window_used", l_st_f.get("smart_scan_range", [0, 0]))
    r_window_used = r_st_f.get("stereo_link_window_used", r_st_f.get("smart_scan_range", [0, 0]))

    acoustic_summary = []
    if l_ai.get("score") is not None and r_ai.get("score") is not None:
        acoustic_summary.append(t("results_summary_avg_acoustic_score").format(score=avg_pred))
    if l_match is not None and r_match is not None:
        acoustic_summary.append(t("results_summary_avg_target_match").format(match=avg_match))

    l_boost_pre, _, l_net_boost = boost_diag(l_st_f)
    r_boost_pre, _, r_net_boost = boost_diag(r_st_f)

    _section(
        t("results_section_acoustic_summary"),
        [
            metric_row(
                t("results_metric_stereo_leveling"),
                stereo_link_mode_label(l_st_f, stereo_link_enabled=stereo_link_enabled),
                stereo_link_mode_label(r_st_f, stereo_link_enabled=stereo_link_enabled),
            ),
            metric_row(
                t("results_metric_target_level"),
                f"{l_st_f.get('eff_target_db', 0):.1f} dB",
                f"{r_st_f.get('eff_target_db', 0):.1f} dB",
            ),
            metric_row(
                t("results_metric_window_used_for_leveling"),
                fmt_freq_window(l_window_used),
                fmt_freq_window(r_window_used),
            ),
            metric_row(
                t("results_metric_shared_leveling_window"),
                shared_window_label(l_st_f),
                shared_window_label(r_st_f),
            ),
            metric_row(
                t("results_metric_leveling_tilt"),
                fmt_tilt(l_st_f),
                fmt_tilt(r_st_f),
                left_compare=safe_float(l_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
                right_compare=safe_float(r_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
            ),
            metric_row(
                t("results_metric_offset_to_meas"),
                f"{l_st_f.get('offset_db', 0):.1f} dB",
                f"{r_st_f.get('offset_db', 0):.1f} dB",
            ),
            metric_row(
                t("results_metric_stereo_anchor"),
                anchor_label(l_st_f, stereo_link_enabled=stereo_link_enabled),
                anchor_label(r_st_f, stereo_link_enabled=stereo_link_enabled),
            ),
            {"label": t("results_metric_target_match"), "left": fmt_ai_match(l_ai), "right": fmt_ai_match(r_ai)},
            metric_row(
                t("results_metric_acoustic_confidence"),
                f"{l_st_f.get('avg_confidence', 0):.1f}%",
                f"{r_st_f.get('avg_confidence', 0):.1f}%",
            ),
            {"label": t("results_metric_acoustic_score"), "left": fmt_ai_score(l_ai), "right": fmt_ai_score(r_ai)},
            metric_row(
                t("results_metric_estimated_rt60"),
                f"{l_st_f.get('rt60_val', 0):.2f} s",
                f"{r_st_f.get('rt60_val', 0):.2f} s",
            ),
        ],
        summary_lines=acoustic_summary,
    )

    tdc_text = (
        t("results_value_tdc_on").format(
            strength=float(data.get("tdc_strength", 0)),
            max_reduction=float(data.get("tdc_max_reduction_db", 0)),
        )
        if bool(data.get("enable_tdc", False))
        else t("results_value_off")
    )
    _section(
        t("results_section_gain_headroom"),
        [
            metric_row(t("results_metric_tdc"), tdc_text, tdc_text),
            metric_row(
                t("results_metric_auto_gain_margin"),
                f"{float(l_st_f.get('gain_margin_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('gain_margin_db', 0) or 0):.2f} dB",
            ),
            metric_row(
                t("results_metric_applied_auto_gain"),
                f"{float(l_st_f.get('auto_global_gain_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('auto_global_gain_db', 0) or 0):.2f} dB",
            ),
            metric_row(
                t("results_metric_net_boost_pre_to_net"),
                f"{l_boost_pre:.2f} dB -> {l_net_boost:.2f} dB",
                f"{r_boost_pre:.2f} dB -> {r_net_boost:.2f} dB",
            ),
            metric_row(
                t("results_metric_final_max_post_gain"),
                f"{float(l_st_f.get('final_max_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('final_max_db', 0) or 0):.2f} dB",
            ),
        ],
    )


def _render_auto_diagnostics(*, data: dict) -> None:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    auto_meta = data.get("_auto_mode_meta", None)
    if not (auto_enabled and isinstance(auto_meta, dict)):
        return

    from nicegui import ui  # noqa: PLC0415
    from ..auto_mode.rank_score import attach_official_rank_score, official_rank_score  # noqa: PLC0415

    bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
    rank_sc = safe_float(official_rank_score(bm), 0.0)
    avg_sc = safe_float(bm.get("avg_score", 0.0), 0.0)
    boost_db = safe_float(bm.get("max_net_boost_db", 0.0), 0.0)
    tc_meta = dict(data.get("_auto_target_curve_meta", {}) or {})
    tc_selected = str(tc_meta.get("selected_hc_mode", data.get("hc_mode", "n/a")) or "n/a")
    trials_ok = int(auto_meta.get("trials_ok", 0) or 0)
    trials_total = int(auto_meta.get("trials_total", 0) or 0)

    with ui.expansion(t("results_auto_diag_title")).classes("w-full"):
        ui.markdown(
            t("results_auto_diag_summary").format(
                target=tc_selected,
                rank=rank_sc,
                avg=avg_sc,
                boost=boost_db,
                ok=trials_ok,
                total=trials_total,
            )
        )


def _update_crossover_recommendation_label(data: dict) -> None:
    """Update the crossover recommendation label in the input tab after a run."""
    try:
        from . import ng_controls as ctrl  # noqa: PLC0415
        label_el = ctrl.get("avr_crossover_hz_recommendation")
        if label_el is None:
            return
        bi_meta = dict((data or {}).get("_bass_integration_meta", {}) or {})
        rec = bi_meta.get("recommended_crossover_hz", None)
        if rec is not None:
            bi_mode = str(
                bi_meta.get("mode", (data or {}).get("bass_integration_mode", "avr_lfe_main_decomposed"))
                or "avr_lfe_main_decomposed"
            ).strip().lower()
            rec_label = (
                t("bass_integration_main_sub_crossover_recommended")
                if bi_mode == "direct_dac"
                else t("bass_integration_crossover_recommended")
            )
            label_el.set_text(f"{rec_label}: {_format_recommended_xo_hz(float(rec))}")
        else:
            label_el.set_text("")
    except Exception:
        pass


def _render_bass_integration(*, data: dict) -> None:
    if not bool((data or {}).get("bass_integration_enable", False)):
        return

    bi_meta = dict((data or {}).get("_bass_integration_meta", {}) or {})
    auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
    best_metrics = dict(auto_meta.get("best_metrics", {}) or {})
    diag = dict(bi_meta.get("diagnostics", {}) or {})
    sub_st = dict(bi_meta.get("sub_filter_stats", {}) or {})
    allpass_meta = dict(bi_meta.get("recommended_allpass", {}) or {})
    allpass_baseline = dict(bi_meta.get("allpass_baseline_metrics", {}) or {})
    allpass_optimized = dict(bi_meta.get("allpass_optimized_metrics", {}) or {})
    allpass_auto_enabled = bool(data.get("bass_integration_allpass_auto_enable", False))
    allpass_on = bool(allpass_meta.get("enabled", False))

    cancellation_risk = safe_float(
        best_metrics.get("bass_cancellation_risk", diag.get("cancellation_risk", float("nan"))),
        float("nan"),
    )
    overlap_ripple = safe_float(
        best_metrics.get("bass_overlap_ripple", diag.get("overlap_ripple_db", float("nan"))),
        float("nan"),
    )
    sub_dominance = safe_float(
        best_metrics.get("bass_sub_dominance", diag.get("sub_dominance_db", float("nan"))),
        float("nan"),
    )
    xo_gd_mismatch = safe_float(
        best_metrics.get("bass_xo_gd_mismatch_ms", float("nan")),
        float("nan"),
    )
    xo_main_gd = safe_float(best_metrics.get("bass_xo_main_gd_ms", float("nan")), float("nan"))
    xo_sub_gd = safe_float(best_metrics.get("bass_xo_sub_gd_ms", float("nan")), float("nan"))

    def _fmt(v: float, unit: str = "") -> str:
        if v != v or abs(v) == float("inf"):
            return "n/a"
        return f"{float(v):.3f}{unit}"

    def _fmt_transition(before: float, after: float, unit: str = "") -> str:
        before_ok = before == before and abs(before) != float("inf")
        after_ok = after == after and abs(after) != float("inf")
        if before_ok and after_ok:
            return f"{float(before):.3f}{unit} -> {float(after):.3f}{unit}"
        if after_ok:
            return _fmt(after, unit)
        if before_ok:
            return _fmt(before, unit)
        return "n/a"

    bi_mode = str(
        bi_meta.get("mode", data.get("bass_integration_mode", "avr_lfe_main_decomposed"))
        or "avr_lfe_main_decomposed"
    ).strip().lower()
    bi_mode_label = (
        t("bi_mode_direct_dac")
        if bi_mode == "direct_dac"
        else t("bass_integration_mode_avr_lfe_main_decomposed")
    )
    xo_metric_label = (
        t("results_metric_main_sub_crossover")
        if bi_mode == "direct_dac"
        else t("results_metric_avr_crossover")
    )
    xo_rec_metric_label = (
        t("results_metric_main_sub_crossover_recommended")
        if bi_mode == "direct_dac"
        else t("results_metric_avr_crossover_recommended")
    )
    playback_note = (
        t("bass_integration_direct_playback_match")
        if bi_mode == "direct_dac"
        else t("bass_integration_playback_match")
    )
    baseline_cancel = safe_float(allpass_baseline.get("cancellation_risk", float("nan")), float("nan"))
    baseline_ripple = safe_float(allpass_baseline.get("overlap_ripple_db", float("nan")), float("nan"))
    baseline_gd = safe_float(allpass_baseline.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
    optimized_cancel = safe_float(
        best_metrics.get("bass_cancellation_risk", allpass_optimized.get("cancellation_risk", cancellation_risk)),
        float("nan"),
    )
    optimized_ripple = safe_float(
        best_metrics.get("bass_overlap_ripple", allpass_optimized.get("overlap_ripple_db", overlap_ripple)),
        float("nan"),
    )
    optimized_gd = safe_float(
        best_metrics.get("bass_xo_gd_mismatch_ms", allpass_optimized.get("xo_gd_mismatch_ms", xo_gd_mismatch)),
        float("nan"),
    )

    _section(
        t("results_section_bass_integration"),
        [
            metric_row(
                t("results_metric_bass_integration_mode"),
                bi_mode_label,
                bi_mode_label,
            ),
            metric_row(
                xo_metric_label,
                f"{float(bi_meta.get('avr_crossover_hz', data.get('avr_crossover_hz', 80.0)) or 80.0):.1f} Hz",
                f"{float(bi_meta.get('avr_crossover_hz', data.get('avr_crossover_hz', 80.0)) or 80.0):.1f} Hz",
            ),
            *(
                [
                    metric_row(
                        xo_rec_metric_label,
                        _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz"])),
                        _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz"])),
                    )
                ]
                if bi_meta.get("recommended_crossover_hz") is not None
                else []
            ),
            metric_row(
                t("results_metric_bass_integration_profile"),
                str(bi_meta.get("profile", data.get("bass_integration_profile", "safe")) or "safe"),
                str(bi_meta.get("profile", data.get("bass_integration_profile", "safe")) or "safe"),
            ),
            metric_row(
                t("results_metric_bass_allpass"),
                t("state_on") if allpass_on else t("state_off"),
                t("state_on") if allpass_on else t("state_off"),
            ),
            metric_row(
                t("results_metric_bass_allpass_freq"),
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
            ),
            metric_row(
                t("results_metric_bass_allpass_q"),
                f"{float(allpass_meta.get('q', 0.707) or 0.707):.3f}" if allpass_on else "n/a",
                f"{float(allpass_meta.get('q', 0.707) or 0.707):.3f}" if allpass_on else "n/a",
            ),
            metric_row(
                t("results_metric_bass_allpass_improvement"),
                _fmt(safe_float(allpass_meta.get("improvement_score", float("nan")), float("nan"))),
                _fmt(safe_float(allpass_meta.get("improvement_score", float("nan")), float("nan"))),
            ),
            metric_row(
                t("results_metric_bass_cancellation_risk"),
                _fmt_transition(baseline_cancel, optimized_cancel),
                _fmt_transition(baseline_cancel, optimized_cancel),
            ),
            metric_row(
                t("results_metric_bass_overlap_smoothness"),
                _fmt_transition(baseline_ripple, optimized_ripple, " dB p2p"),
                _fmt_transition(baseline_ripple, optimized_ripple, " dB p2p"),
            ),
            metric_row(
                t("results_metric_bass_sub_dominance"),
                _fmt(sub_dominance, " dB"),
                _fmt(sub_dominance, " dB"),
            ),
            metric_row(
                t("results_metric_bass_xo_gd_mismatch"),
                _fmt_transition(baseline_gd, optimized_gd, " ms"),
                _fmt_transition(baseline_gd, optimized_gd, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_main_gd"),
                _fmt(xo_main_gd, " ms"),
                _fmt(xo_main_gd, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_sub_gd"),
                _fmt(xo_sub_gd, " ms"),
                _fmt(xo_sub_gd, " ms"),
            ),
            *(
                [
                    metric_row(
                        t("results_metric_sub_filter_boost"),
                        _fmt(safe_float(sub_st.get("max_boost_db_effective", sub_st.get("max_boost_db", float("nan"))), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("max_boost_db_effective", sub_st.get("max_boost_db", float("nan"))), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_cut"),
                        _fmt(safe_float(sub_st.get("max_cut_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("max_cut_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_gain_margin"),
                        _fmt(safe_float(sub_st.get("gain_margin_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("gain_margin_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_applied_gain"),
                        _fmt(safe_float(sub_st.get("auto_global_gain_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("auto_global_gain_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_confidence"),
                        _fmt(safe_float(sub_st.get("avg_confidence", float("nan")), float("nan")), "%"),
                        _fmt(safe_float(sub_st.get("avg_confidence", float("nan")), float("nan")), "%"),
                    ),
                ]
                if sub_st
                else []
            ),
        ],
        summary_lines=[
            playback_note,
            *(
                [t("bass_allpass_no_improvement")]
                if bi_mode == "direct_dac" and allpass_auto_enabled and not allpass_on
                else []
            ),
        ],
    )


def _render_ir_alignment(*, l_st_f: dict) -> None:
    """Renderöi Mittausten IR-infot -osion jos ir_align- tai ir_align_sub-data löytyy."""
    ir_align = dict(l_st_f.get("ir_align") or {})
    ir_align_sub = dict(l_st_f.get("ir_align_sub") or {})
    if not ir_align and not ir_align_sub:
        return

    def _fmt_ms(v) -> str:
        try:
            x = float(v)
            if math.isfinite(x):
                return f"{x:+.2f} ms"
        except Exception:
            pass
        return "-"

    def _fmt_db(v) -> str:
        try:
            x = float(v)
            if math.isfinite(x):
                return f"{x:+.1f} dB"
        except Exception:
            pass
        return "-"

    def _fmt_deg(v) -> str:
        try:
            x = float(v)
            if math.isfinite(x):
                return f"{x:+.1f}°"
        except Exception:
            pass
        return "-"

    def _fmt_pct(v) -> str:
        try:
            x = float(v)
            if math.isfinite(x):
                return f"{x * 100.0:.0f}%"
        except Exception:
            pass
        return "-"

    def _fmt_dbfs(v) -> str:
        try:
            x = float(v)
            if math.isfinite(x):
                return f"{x:.1f} dBFS"
        except Exception:
            pass
        return "-"

    def _polarity_str(d: dict) -> str:
        inv = bool(d.get("ir_align_polarity_inverted", False)) or bool(d.get("ir_align_xcorr_polarity_flip", False))
        return t("ir_align_value_inverted") if inv else t("ir_align_value_ok")

    def _xo_label(d: dict) -> str:
        xo = safe_float(d.get("ir_align_xo_hz"), None)
        if xo is not None and math.isfinite(xo):
            return f"{xo:.0f}"
        return "80"

    rows: list[dict] = []

    if ir_align:
        rows.append(metric_row(f"── {t('ir_align_group_lr')} ──", "", ""))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_offset"),
            _fmt_ms(ir_align.get("ir_align_xcorr_offset_ms")),
            _fmt_ms(ir_align.get("ir_align_xcorr_offset_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_confidence"),
            _fmt_pct(ir_align.get("ir_align_xcorr_confidence")),
            _fmt_pct(ir_align.get("ir_align_xcorr_confidence")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_polarity"),
            _polarity_str(ir_align),
            _polarity_str(ir_align),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_peak"),
            _fmt_dbfs(ir_align.get("ir_align_level_peak_a_dbfs")),
            _fmt_dbfs(ir_align.get("ir_align_level_peak_b_dbfs")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_diff"),
            _fmt_db(ir_align.get("ir_align_level_rms_diff_db")),
            _fmt_db(ir_align.get("ir_align_level_rms_diff_db")),
        ))

    if ir_align_sub:
        xo_lbl = _xo_label(ir_align_sub)
        rows.append(metric_row(f"── {t('ir_align_group_sub')} ──", "", ""))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_offset"),
            _fmt_ms(ir_align_sub.get("ir_align_xcorr_offset_ms")),
            _fmt_ms(ir_align_sub.get("ir_align_xcorr_offset_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_confidence"),
            _fmt_pct(ir_align_sub.get("ir_align_xcorr_confidence")),
            _fmt_pct(ir_align_sub.get("ir_align_xcorr_confidence")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_polarity"),
            _polarity_str(ir_align_sub),
            _polarity_str(ir_align_sub),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_peak"),
            _fmt_dbfs(ir_align_sub.get("ir_align_level_peak_a_dbfs")),
            _fmt_dbfs(ir_align_sub.get("ir_align_level_peak_b_dbfs")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_diff"),
            _fmt_db(ir_align_sub.get("ir_align_level_rms_diff_db")),
            _fmt_db(ir_align_sub.get("ir_align_level_rms_diff_db")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_phase_diff").format(xo=xo_lbl),
            _fmt_deg(ir_align_sub.get("ir_align_phase_diff_deg")),
            _fmt_deg(ir_align_sub.get("ir_align_phase_diff_deg")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_gd").format(xo=xo_lbl),
            _fmt_ms(ir_align_sub.get("ir_align_gd_a_ms")),
            _fmt_ms(ir_align_sub.get("ir_align_gd_b_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_gd_diff"),
            _fmt_ms(ir_align_sub.get("ir_align_gd_diff_ms")),
            _fmt_ms(ir_align_sub.get("ir_align_gd_diff_ms")),
        ))

    # Yhteenvetorivi
    issues: list[str] = []
    for d in (ir_align, ir_align_sub):
        if not d:
            continue
        if bool(d.get("ir_align_polarity_inverted")) or bool(d.get("ir_align_xcorr_polarity_flip")):
            issues.append(t("ir_align_issue_polarity"))
        off = safe_float(d.get("ir_align_xcorr_offset_ms"), None)
        if off is not None and math.isfinite(off) and abs(off) >= 5.0:
            issues.append(t("ir_align_issue_timing"))
        if not bool(d.get("ir_align_phase_in_phase", True)) and d is ir_align_sub:
            issues.append(t("ir_align_issue_phase"))
        rms = safe_float(d.get("ir_align_level_rms_diff_db"), None)
        if rms is not None and math.isfinite(rms) and abs(rms) >= 10.0:
            issues.append(t("ir_align_issue_level"))

    seen: set[str] = set()
    unique_issues = [x for x in issues if not (x in seen or seen.add(x))]
    if unique_issues:
        summary_line = t("ir_align_summary_issues").format(issues=", ".join(unique_issues))
    else:
        summary_line = t("ir_align_summary_ok")

    _section(t("results_section_ir_alignment"), rows, summary_lines=[summary_line])


def _render_dsp_quality(*, data: dict, l_st_f: dict, r_st_f: dict, psl_str: str) -> None:
    _section(
        t("results_section_filter_ir"),
        [
            metric_row(
                t("results_metric_filter_type"),
                str(data.get("filter_type", "") or ""),
                str(data.get("filter_type", "") or ""),
            ),
            metric_row(t("results_metric_ir_window"), format_ir_window(l_st_f), format_ir_window(r_st_f)),
            metric_row(
                t("results_metric_correction_range"),
                f"{data.get('mag_c_min', 0):.0f}-{data.get('mag_c_max', 0):.0f} Hz",
                f"{data.get('mag_c_min', 0):.0f}-{data.get('mag_c_max', 0):.0f} Hz",
            ),
            metric_row(t("results_metric_plot_smoothing"), psl_str, psl_str),
        ],
    )
    _section(
        t("results_section_phase_gd"),
        [
            metric_row(t("results_metric_xo_phase_model"), xo_fc_gd_label(l_st_f), xo_fc_gd_label(r_st_f)),
            metric_row(t("results_metric_phase_clamp"), phase_clamp_label(l_st_f), phase_clamp_label(r_st_f)),
            metric_row(t("results_metric_gd_limiter"), gd_limiter_label(l_st_f), gd_limiter_label(r_st_f)),
            metric_row(t("results_metric_gd_gradient_max"), gd_grad_max_label(l_st_f), gd_grad_max_label(r_st_f)),
            metric_row(t("results_metric_hpf_model"), hpf_model_label(l_st_f), hpf_model_label(r_st_f)),
            metric_row(t("results_metric_hpf_diff_raw"), hpf_diff_raw_label(l_st_f), hpf_diff_raw_label(r_st_f)),
            metric_row(
                t("results_metric_mixed_blend_split"),
                mixed_blend_label(l_st_f, "mixed_blend_split_hz"),
                mixed_blend_label(r_st_f, "mixed_blend_split_hz"),
            ),
            metric_row(
                t("results_metric_mixed_blend_transition"),
                mixed_blend_label(l_st_f, "mixed_blend_transition_hz"),
                mixed_blend_label(r_st_f, "mixed_blend_transition_hz"),
            ),
        ],
    )


def _render_plots_and_export(
    *,
    data,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    l_imp_f,
    r_imp_f,
    l_st_f,
    r_st_f,
    fname,
    zip_buffer,
    dash_html_l=None,
    dash_html_r=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
    sub_imp_f=None,
    sub_meas_f=None,
    sub_st_f=None,
) -> None:
    from nicegui import ui  # noqa: PLC0415

    psl = data.get("plot_smoothing_level", "Psychoacoustic")
    mixed_freq = data.get("mixed_freq", 200.0)
    fs_v = int(data.get("fs", 48000) or 48000)

    def _render_channel_plot(*, title, f_ch, m_ch, p_ch, imp_f, st_f) -> None:
        try:
            result = plots.generate_prediction_plot(
                f_ch,
                m_ch,
                p_ch,
                imp_f,
                fs_v,
                title,
                None,
                st_f,
                mixed_freq,
                "low",
                False,
                True,
                psl,
            )
            fig = result[1] if isinstance(result, tuple) else None
            if fig is not None:
                ui.plotly(fig).classes("w-full")
            else:
                ui.label(t("results_plot_unavailable")).classes("text-gray-400")
        except Exception:
            logger.debug("Plot generation failed", exc_info=True)
            ui.label(t("results_plot_unavailable")).classes("text-gray-400")

    sub_meas = dict(sub_meas_f or {})
    has_sub = (
        sub_imp_f is not None
        and sub_meas.get("f_sub") is not None
        and len(sub_meas.get("f_sub", [])) > 0
    )

    with ui.card().classes("w-full"):
        with ui.tabs().classes("w-full") as plot_tabs:
            tab_left = ui.tab(t("results_left_channel"))
            tab_right = ui.tab(t("results_right_channel"))
            tab_sub = ui.tab(t("results_sub_channel")) if has_sub else None

        with ui.tab_panels(plot_tabs, value=tab_left).classes("w-full"):
            with ui.tab_panel(tab_left).classes("w-full"):
                _render_channel_plot(
                    title=t("results_left_channel"),
                    f_ch=f_l,
                    m_ch=m_l,
                    p_ch=p_l,
                    imp_f=l_imp_f,
                    st_f=l_st_f,
                )

            with ui.tab_panel(tab_right).classes("w-full"):
                _render_channel_plot(
                    title=t("results_right_channel"),
                    f_ch=f_r,
                    m_ch=m_r,
                    p_ch=p_r,
                    imp_f=r_imp_f,
                    st_f=r_st_f,
                )

            if has_sub and tab_sub is not None:
                with ui.tab_panel(tab_sub).classes("w-full"):
                    _render_channel_plot(
                        title=t("results_sub_channel"),
                        f_ch=sub_meas["f_sub"],
                        m_ch=sub_meas["m_sub"],
                        p_ch=sub_meas["p_sub"],
                        imp_f=sub_imp_f,
                        st_f=sub_st_f,
                    )

    with ui.row().classes("w-full items-center gap-4 mt-2"):
        if zip_buffer is not None and fname:
            try:
                zip_bytes = zip_buffer.getvalue()
                ui.button(
                    t("results_download_zip").format(fname=fname),
                    on_click=lambda: ui.download(zip_bytes, filename=fname),
                ).props('color="primary" unelevated').classes("font-bold")
            except Exception:
                logger.debug("ZIP download button failed", exc_info=True)

        if saved_filters_dir:
            ui.label(t("results_saved_to").format(path=saved_filters_dir)).classes("text-sm text-gray-400")
