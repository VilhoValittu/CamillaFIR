"""NiceGUI results rendering after a DSP run.

Replaces camillafir_ui._render_results() + results_sections.py.

render_results() has the same signature as _render_results() so ng_bridge.py
can call it without changes to workflow code.
"""
from __future__ import annotations

import html
import logging
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

    with ui.card().classes("w-full"):
        with ui.tabs().classes("w-full") as plot_tabs:
            tab_left = ui.tab(t("results_left_channel"))
            tab_right = ui.tab(t("results_right_channel"))

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
