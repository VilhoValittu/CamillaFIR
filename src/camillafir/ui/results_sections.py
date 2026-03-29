from __future__ import annotations

import time

from pywebio.output import (
    put_collapse,
    put_file,
    put_html,
    put_info,
    put_markdown,
    put_table,
    put_tabs,
)

from ..io.auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ..resources.i8n.camillafir_i18n import t
from . import camillafir_plot as plots
from .results_formatters import (
    anchor_label,
    auto_choice_text,
    boost_diag,
    float3_or_na,
    fmt_ai_match,
    fmt_ai_score,
    fmt_freq_window,
    fmt_tilt,
    format_ir_window,
    gd_grad_max_label,
    gd_limiter_label,
    hpf_diff_raw_label,
    hpf_model_label,
    metric_cell,
    metric_row,
    mixed_blend_label,
    phase_clamp_label,
    safe_float,
    safe_int,
    shared_window_label,
    stereo_link_mode_label,
    xo_fc_gd_badge,
    xo_fc_gd_label,
    xo_fc_wrapped_label,
    xo_diff_raw_label,
    xo_phase_model_label,
)


def _put_metric_collapse(title, rows, *, summary_lines=None, open=False):
    content = []
    for line in list(summary_lines or []):
        if line:
            content.append(put_markdown(str(line)))

    shared_rows = [["Metric", "Value"]]
    stereo_rows = [["Metric", "L", "R"]]
    for row in list(rows or []):
        left = dict(row.get("left", {}) or {})
        right = dict(row.get("right", {}) or {})
        if str(left.get("compare", "")) == str(right.get("compare", "")):
            shared_rows.append([row.get("label", ""), left.get("render", "-")])
        else:
            stereo_rows.append([
                row.get("label", ""),
                left.get("render", "-"),
                right.get("render", "-"),
            ])

    if len(shared_rows) > 1:
        content.append(put_table(shared_rows))
    if len(stereo_rows) > 1:
        content.append(put_table(stereo_rows))
    if content:
        put_collapse(title, content, open=open)


def render_run_overview_section(*, data: dict, l_st_f: dict, r_st_f: dict) -> None:
    l_ai = plots.calc_ai_summary_from_stats(l_st_f)
    r_ai = plots.calc_ai_summary_from_stats(r_st_f)

    l_score = float(l_ai.get("score") or 0.0)
    r_score = float(r_ai.get("score") or 0.0)
    avg_pred = (l_score + r_score) / 2.0
    l_match = l_ai.get("match")
    r_match = r_ai.get("match")
    if (l_match is None) or (r_match is None):
        avg_match = 0.0
    else:
        avg_match = (float(l_match) + float(r_match)) / 2.0

    stereo_link_enabled = bool(data.get("stereo_link", False))
    l_window_used = l_st_f.get("stereo_link_window_used", l_st_f.get("smart_scan_range", [0, 0]))
    r_window_used = r_st_f.get("stereo_link_window_used", r_st_f.get("smart_scan_range", [0, 0]))
    l_link_mode = stereo_link_mode_label(l_st_f, stereo_link_enabled=stereo_link_enabled)
    r_link_mode = stereo_link_mode_label(r_st_f, stereo_link_enabled=stereo_link_enabled)
    l_shared_window = shared_window_label(l_st_f)
    r_shared_window = shared_window_label(r_st_f)
    l_anchor = anchor_label(l_st_f, stereo_link_enabled=stereo_link_enabled)
    r_anchor = anchor_label(r_st_f, stereo_link_enabled=stereo_link_enabled)
    l_boost_pre, _l_auto_gain, l_net_boost = boost_diag(l_st_f)
    r_boost_pre, _r_auto_gain, r_net_boost = boost_diag(r_st_f)

    acoustic_summary = []
    if l_ai.get("score") is not None and r_ai.get("score") is not None:
        acoustic_summary.append(f"**Average acoustic score:** {avg_pred:.3f}/100")
    if l_match is not None and r_match is not None:
        acoustic_summary.append(f"**Average target match:** {avg_match:.1f}%")

    _put_metric_collapse(
        " Acoustic summary",
        [
            metric_row("Stereo Leveling", l_link_mode, r_link_mode),
            metric_row("Target Level", f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"),
            metric_row("Window Used for Leveling", fmt_freq_window(l_window_used), fmt_freq_window(r_window_used)),
            metric_row("Shared Leveling Window", l_shared_window, r_shared_window),
            metric_row(
                "Leveling Tilt",
                fmt_tilt(l_st_f),
                fmt_tilt(r_st_f),
                left_compare=safe_float(l_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
                right_compare=safe_float(r_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
            ),
            metric_row("Offset to Meas.", f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"),
            metric_row("Stereo Anchor", l_anchor, r_anchor),
            {"label": "Target Match", "left": fmt_ai_match(l_ai), "right": fmt_ai_match(r_ai)},
            metric_row("Acoustic Confidence", f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"),
            {"label": "Acoustic Score", "left": fmt_ai_score(l_ai), "right": fmt_ai_score(r_ai)},
            metric_row("Estimated RT60", f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"),
        ],
        summary_lines=acoustic_summary,
        open=False,
    )

    tdc_text = (
        f"ON ({float(data.get('tdc_strength', 0)):.0f}%, -{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
        if bool(data.get("enable_tdc", False))
        else "OFF"
    )
    _put_metric_collapse(
        " Gain and headroom",
        [
            metric_row("TDC (Temporal Decay Control)", tdc_text, tdc_text),
            metric_row(
                "Auto Gain Margin",
                f"{float(l_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB",
                f"{float(r_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB",
            ),
            metric_row(
                "Applied Auto Gain",
                f"{float(l_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB",
                f"{float(r_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB",
            ),
            metric_row(
                "Extra Headroom",
                f"{float(l_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB",
                f"{float(r_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB",
            ),
            metric_row(
                "Net Boost (pre -> net)",
                f"{l_boost_pre:.2f} dB -> {l_net_boost:.2f} dB",
                f"{r_boost_pre:.2f} dB -> {r_net_boost:.2f} dB",
            ),
            metric_row(
                "Final Max (post gain)",
                f"{float(l_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB",
                f"{float(r_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB",
            ),
        ],
        summary_lines=["Net boost is reduced by baked auto gain (preamp) to prevent clipping."],
        open=False,
    )


def render_auto_mode_diagnostics_section(*, data: dict) -> None:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    auto_meta = data.get("_auto_mode_meta", None)
    if not (auto_enabled and isinstance(auto_meta, dict)):
        return

    bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
    top = list(auto_meta.get("top", []) or [])
    tc_meta = dict(data.get("_auto_target_curve_meta", {}) or {})
    trials_ok = int(auto_meta.get("trials_ok", 0) or 0)
    trials_total = int(auto_meta.get("trials_total", 0) or 0)

    rank_sc = safe_float(official_rank_score(bm), 0.0)
    avg_sc = safe_float(bm.get("avg_score", 0.0), 0.0)
    dsp_pen = safe_float(bm.get("dsp_penalty", 0.0), 0.0)
    exc_pen = safe_float(bm.get("exc_penalty", 0.0), 0.0)
    boost_db = safe_float(bm.get("max_net_boost_db", 0.0), 0.0)
    events_n = safe_int(bm.get("events_total", 0), 0)
    events_sev = safe_float(bm.get("events_severity", 0.0), 0.0)
    lr_delta = safe_float(bm.get("lr_delta_score", 0.0), 0.0)
    prepost = safe_float(bm.get("ir_pre_post_energy_ratio_max", float("nan")), float("nan"))
    mode_ripple = safe_float(bm.get("mode_ripple_db", float("nan")), float("nan"))
    lf_rms_20_200 = safe_float(bm.get("realized_rms_20_200_db", float("nan")), float("nan"))

    tc_selected = str(tc_meta.get("selected_hc_mode", data.get("hc_mode", "n/a")) or "n/a")
    tc_method = auto_choice_text(tc_meta.get("selection_method", ""))
    atm = str(data.get("auto_target_mode", "auto") or "auto").strip().lower()
    if atm == "adaptive":
        tc_mode_txt = "Adaptive (synthesized from measurements)"
    elif atm == "selected":
        tc_mode_txt = "User selection"
    else:
        tc_mode_txt = "Auto (best built-in)"
    search_fs = safe_int(auto_meta.get("search_fs", 0), 0)
    search_taps = safe_int(auto_meta.get("search_taps", 0), 0)
    exc_seed = safe_float(
        auto_meta.get(
            "auto_exc_seed_freq_hz",
            data.get("_auto_exc_seed_freq_hz", data.get("_auto_exc_freq_hz", float("nan"))),
        ),
        float("nan"),
    )
    exc_final = safe_float(
        auto_meta.get(
            "best_auto_exc_freq_hz",
            data.get("_auto_exc_freq_hz", data.get("exc_freq", float("nan"))),
        ),
        float("nan"),
    )
    exc_text = "n/a"
    if exc_seed == exc_seed and exc_final == exc_final:
        exc_text = f"{exc_seed:.1f} Hz -> {exc_final:.1f} Hz"
    elif exc_final == exc_final:
        exc_text = f"{exc_final:.1f} Hz"
    elif exc_seed == exc_seed:
        exc_text = f"{exc_seed:.1f} Hz"

    auto_rows = [
        ["Goal / basis", f"{str(auto_meta.get('auto_goal', 'balanced') or 'balanced')} / preset_objective_score"],
        ["Trials", f"{trials_ok}/{trials_total}"],
        ["Search grid", f"{search_fs} Hz / {search_taps} taps"],
        ["Target mode", tc_mode_txt],
        ["Target curve", tc_selected],
        ["Target selection", tc_method],
        ["Excursion protection", exc_text],
        ["Best rank score", f"{rank_sc:.3f}/100"],
        ["Average acoustic score", f"{avg_sc:.3f}"],
        ["Pre/post energy ratio (max)", f"{prepost:.4f}" if prepost == prepost else "n/a"],
        ["Mode ripple", f"{mode_ripple:.3f} dB" if mode_ripple == mode_ripple else "n/a"],
        ["LF RMS 20-200 Hz", f"{lf_rms_20_200:.3f} dB" if lf_rms_20_200 == lf_rms_20_200 else "n/a"],
        ["Max net boost", f"{boost_db:.2f} dB"],
        ["L/R delta", f"{lr_delta:.3f}"],
        ["DSP penalty", f"{dsp_pen:.2f}"],
        ["Excursion penalty", f"{exc_pen:.2f}"],
        ["Events / severity", f"{events_n} / {events_sev:.2f}"],
    ]

    auto_content = [
        put_markdown(
            "Selection order: **Rank score** -> **avg score** -> **pre/post ratio** -> "
            "**mode ripple** -> **LF RMS (20-200 Hz)** -> **net boost**."
        ),
        put_table([["Metric", "Value"], *auto_rows]),
    ]
    best_preset = dict(auto_meta.get("best_preset", {}) or {})
    preset_rows = []
    if best_preset:
        filter_type = str(best_preset.get("filter_type", data.get("filter_type", "")) or "").strip().lower()
        if filter_type == "mixed" and "mixed_freq" in best_preset:
            preset_rows.append(["Mixed split", f"{safe_float(best_preset.get('mixed_freq'), 0.0):.1f} Hz"])
        if filter_type in ("linear", "asym", "asymmetric") and "phase_limit" in best_preset:
            preset_rows.append(["Phase limit", f"{safe_float(best_preset.get('phase_limit'), 0.0):.1f} Hz"])
        if "fdw_cycles" in best_preset:
            preset_rows.append(["FDW", f"{safe_float(best_preset.get('fdw_cycles'), 0.0):.1f}"])
        if "tdc_strength" in best_preset and "tdc_max_reduction_db" in best_preset:
            preset_rows.append([
                "TDC preset",
                f"{safe_float(best_preset.get('tdc_strength'), 0.0):.0f}% / -{safe_float(best_preset.get('tdc_max_reduction_db'), 0.0):.1f} dB",
            ])
        if "reg_strength" in best_preset:
            preset_rows.append(["Regularization", f"{safe_float(best_preset.get('reg_strength'), 0.0):.2f}"])
        if "max_boost" in best_preset:
            preset_rows.append(["Max boost", f"{safe_float(best_preset.get('max_boost'), 0.0):.1f} dB"])
        if "mag_c_min" in best_preset and "mag_c_max" in best_preset:
            preset_rows.append([
                "Magnitude correction range",
                f"{safe_float(best_preset.get('mag_c_min'), 0.0):.0f}-{safe_float(best_preset.get('mag_c_max'), 0.0):.0f} Hz",
            ])
    if preset_rows:
        auto_content.append(put_table([["Best preset", "Value"], *preset_rows]))

    put_collapse(" Automatic mode winner", auto_content, open=False)

    tc_eval = list(tc_meta.get("evaluated", []) or [])
    tc_rows = [[
        "#",
        "Target curve",
        "Best rank",
        "Avg rank",
        "Fit RMS",
        "Preselect",
        "Boost pen",
        "Asym",
        "Trials",
    ]]
    if tc_eval:
        tc_eval_sorted = sorted(
            list(tc_eval),
            key=lambda it: (
                -safe_float(official_rank_score(dict((it or {}).get("best_metrics", {}) or {})), 0.0),
                -safe_float((it or {}).get("avg_rank_score", 0.0), 0.0),
                safe_float((it or {}).get("fit_rms_db", 1e9), 1e9),
            ),
        )
        for idx, item in enumerate(tc_eval_sorted[:3], start=1):
            bm_tc = attach_official_rank_score((item or {}).get("best_metrics", {}))
            tc_rows.append([
                f"{idx}",
                str((item or {}).get("hc_mode", "n/a") or "n/a"),
                f"{safe_float(official_rank_score(bm_tc), 0.0):.3f}",
                f"{safe_float((item or {}).get('avg_rank_score', 0.0), 0.0):.3f}",
                float3_or_na((item or {}).get("fit_rms_db", float("nan"))),
                float3_or_na((item or {}).get("preselect_score", float("nan"))),
                float3_or_na((item or {}).get("boost_penalty", float("nan"))),
                float3_or_na((item or {}).get("asym_penalty_db", float("nan"))),
                f"{safe_int((item or {}).get('trials_ok', 0), 0)}/{safe_int((item or {}).get('trials_total', 0), 0)}",
            ])
    else:
        tc_cands = list(tc_meta.get("candidates", []) or [])
        tc_cands_sorted = sorted(
            list(tc_cands),
            key=lambda it: (
                safe_float((it or {}).get("preselect_score", (it or {}).get("fit_rms_db", 1e9)), 1e9),
                safe_float((it or {}).get("fit_rms_db", 1e9), 1e9),
            ),
        )
        for idx, item in enumerate(tc_cands_sorted[:3], start=1):
            tc_rows.append([
                f"{idx}",
                str((item or {}).get("hc_mode", "n/a") or "n/a"),
                "n/a",
                "n/a",
                float3_or_na((item or {}).get("fit_rms_db", float("nan"))),
                float3_or_na((item or {}).get("preselect_score", float("nan"))),
                float3_or_na((item or {}).get("boost_penalty", float("nan"))),
                float3_or_na((item or {}).get("asym_penalty_db", float("nan"))),
                "n/a",
            ])
    if len(tc_rows) > 1:
        put_collapse(" Target curve top-3", [put_table(tc_rows)], open=False)

    if not top:
        return

    rank_best = max((safe_float(official_rank_score(dict(it.get("metrics", {}) or {})), 0.0) for it in top), default=rank_sc)
    if rank_best > (rank_sc + 1e-6):
        put_info(
            "Note: Top-5 below is rank-ordered. Final winner may differ when Pareto tie-break "
            "prefers cleaner pre/post, lower ripple or lower LF RMS."
        )
    top_rows = [[
        "#",
        "Trial",
        "Rank",
        "Avg",
        "Pre/post",
        "Mode ripple",
        "LF RMS 20-200",
        "Boost dB",
        "L/R delta",
        "DSP pen",
    ]]
    for idx, item in enumerate(top[:5], start=1):
        metrics = attach_official_rank_score(item.get("metrics", {}))
        prepost_t = safe_float(metrics.get("ir_pre_post_energy_ratio_max", float("nan")), float("nan"))
        mode_t = safe_float(metrics.get("mode_ripple_db", float("nan")), float("nan"))
        rms_t = safe_float(metrics.get("realized_rms_20_200_db", float("nan")), float("nan"))
        top_rows.append([
            f"{idx}",
            f"{safe_int(metrics.get('trial', 0), 0)}",
            f"{safe_float(official_rank_score(metrics), 0.0):.3f}",
            f"{safe_float(metrics.get('avg_score', 0.0), 0.0):.3f}",
            f"{prepost_t:.4f}" if prepost_t == prepost_t else "n/a",
            f"{mode_t:.3f}" if mode_t == mode_t else "n/a",
            f"{rms_t:.3f}" if rms_t == rms_t else "n/a",
            f"{safe_float(metrics.get('max_net_boost_db', 0.0), 0.0):.2f}",
            f"{safe_float(metrics.get('lr_delta_score', 0.0), 0.0):.3f}",
            f"{safe_float(metrics.get('dsp_penalty', 0.0), 0.0):.2f}",
        ])
    put_collapse(" Automatic mode top-5 (by rank)", [put_table(top_rows)], open=False)


def render_dsp_quality_metrics_section(*, data: dict, l_st_f: dict, r_st_f: dict, psl_str: str) -> None:
    _put_metric_collapse(
        " Filter setup",
        [
            metric_row("Length", f"{data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)", f"{data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)"),
            metric_row("Resolution", f"{data['fs']/data['taps']:.2f} Hz", f"{data['fs']/data['taps']:.2f} Hz"),
            metric_row("IR window", format_ir_window(data), format_ir_window(data)),
            metric_row("FDW", f"{data['fdw_cycles']}", f"{data['fdw_cycles']}"),
            metric_row("House curve", f"{data['hc_mode']} - {data.get('hc_source', 'Unknown')}", f"{data['hc_mode']} - {data.get('hc_source', 'Unknown')}"),
            metric_row("Correction range", f"{data['mag_c_min']}-{data['mag_c_max']} Hz", f"{data['mag_c_min']}-{data['mag_c_max']} Hz"),
            metric_row("Filter type", f"{data['filter_type']}", f"{data['filter_type']}"),
            metric_row("HPF model", hpf_model_label(l_st_f), hpf_model_label(r_st_f)),
            metric_row("Mixed blend split", mixed_blend_label(l_st_f, "mixed_blend_split_hz"), mixed_blend_label(r_st_f, "mixed_blend_split_hz")),
            metric_row("Mixed blend transition", mixed_blend_label(l_st_f, "mixed_blend_transition_hz"), mixed_blend_label(r_st_f, "mixed_blend_transition_hz")),
            metric_row("Smoothing view", psl_str, psl_str),
            metric_row("Leveling algo", str(data.get("lvl_algo", "") or "-"), str(data.get("lvl_algo", "") or "-")),
        ],
        open=False,
    )

    xo_gd_left = xo_fc_gd_label(l_st_f)
    xo_gd_right = xo_fc_gd_label(r_st_f)
    xo_gd_badge_html = xo_fc_gd_badge(l_st_f) or xo_fc_gd_badge(r_st_f)
    _put_metric_collapse(
        " Phase and crossover",
        [
            metric_row("XO phase model", xo_phase_model_label(l_st_f), xo_phase_model_label(r_st_f)),
            metric_row("XO delta phi@fc (wrapped)", xo_fc_wrapped_label(l_st_f), xo_fc_wrapped_label(r_st_f)),
            {
                "label": "XO delta GD@fc",
                "left": metric_cell(put_html(f"{xo_gd_left} {xo_gd_badge_html}" if xo_gd_badge_html else xo_gd_left), xo_gd_left),
                "right": metric_cell(xo_gd_right, xo_gd_right),
            },
            metric_row("XO effect (theoretical raw)", xo_diff_raw_label(l_st_f), xo_diff_raw_label(r_st_f)),
            metric_row("HPF effect (theoretical raw)", hpf_diff_raw_label(l_st_f), hpf_diff_raw_label(r_st_f)),
            metric_row("Phase correction clamp", phase_clamp_label(l_st_f), phase_clamp_label(r_st_f)),
            metric_row("GD limiter", gd_limiter_label(l_st_f), gd_limiter_label(r_st_f)),
            metric_row("A/B GD-gradient max", gd_grad_max_label(l_st_f), gd_grad_max_label(r_st_f)),
        ],
        open=False,
    )


def render_plot_export_section(
    *,
    data: dict,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    l_imp_f,
    r_imp_f,
    l_st_f: dict,
    r_st_f: dict,
    fname: str,
    zip_buffer,
    dash_html_l=None,
    dash_html_r=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
) -> None:
    if dash_html_l is None:
        dash_html_l = plots.generate_prediction_plot(
            f_l,
            m_l,
            p_l,
            l_imp_f,
            data["fs"],
            "Left",
            None,
            l_st_f,
            data["mixed_freq"],
            "low",
            create_full_html=False,
            plot_smoothing_level=data.get("plot_smoothing_level", "Psychoacoustic"),
        )
    if dash_html_r is None:
        dash_html_r = plots.generate_prediction_plot(
            f_r,
            m_r,
            p_r,
            r_imp_f,
            data["fs"],
            "Right",
            None,
            r_st_f,
            data["mixed_freq"],
            "low",
            create_full_html=False,
            plot_smoothing_level=data.get("plot_smoothing_level", "Psychoacoustic"),
        )

    put_tabs([
        {"title": "Left Channel", "content": put_html(dash_html_l)},
        {"title": "Right Channel", "content": put_html(dash_html_r)},
    ])
    put_file(fname, zip_buffer.getvalue(), label=" DOWNLOAD FILTER ZIP")

    try:
        path_rows = [["Item", "Path"]]
        if saved_filters_dir:
            path_rows.append([t("paths_export_folder"), str(saved_filters_dir)])
        if auto_cache_path:
            path_rows.append([t("paths_auto_mode_cache"), str(auto_cache_path)])
        if optuna_storage_path:
            path_rows.append([t("paths_optuna_storage"), str(optuna_storage_path)])
        if len(path_rows) > 1:
            path_rows[0] = [t("paths_item"), t("paths_path")]
            put_markdown(f"### {t('paths_title')}")
            put_table(path_rows)
    except Exception:
        pass


def render_raw_debug_section(
    *,
    render_started_at: float,
    run_started_at=None,
    perf_stats=None,
    per_fs_stats=None,
) -> None:
    try:
        rows = [["Stage", "Time (s)"]]
        perf = dict(perf_stats or {})
        read_s = float(perf.get("read_s", 0.0) or 0.0)
        dsp_s = float(perf.get("dsp_s", 0.0) or 0.0)
        zip_s = float(perf.get("zip_png_s", 0.0) or 0.0)
        render_s = max(0.0, float(time.perf_counter() - render_started_at))
        rows.append(["Read", f"{read_s:.2f}"])
        rows.append(["DSP L/R", f"{dsp_s:.2f}"])
        rows.append(["ZIP + PNG", f"{zip_s:.2f}"])
        rows.append(["Results render", f"{render_s:.2f}"])

        if isinstance(per_fs_stats, dict) and per_fs_stats:
            for fs_k in sorted(per_fs_stats.keys(), key=lambda x: int(x)):
                stats = per_fs_stats.get(fs_k, {}) or {}
                fs_dsp = float(stats.get("dsp_s", 0.0) or 0.0)
                fs_zip = float(stats.get("zip_png_s", 0.0) or 0.0)
                rows.append([f"DSP @ {int(fs_k)} Hz", f"{fs_dsp:.2f}"])
                rows.append([f"ZIP+PNG @ {int(fs_k)} Hz", f"{fs_zip:.2f}"])

        if run_started_at is not None:
            total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
            known_s = max(0.0, read_s + dsp_s + zip_s + render_s)
            other_s = max(0.0, total_s - known_s)
            rows.append(["Other", f"{other_s:.2f}"])
            rows.append(["Total", f"{total_s:.2f}"])

        put_markdown("### Timing")
        put_table(rows)
    except Exception:
        pass


__all__ = [
    "render_auto_mode_diagnostics_section",
    "render_dsp_quality_metrics_section",
    "render_plot_export_section",
    "render_raw_debug_section",
    "render_run_overview_section",
]
