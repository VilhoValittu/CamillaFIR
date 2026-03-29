import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.ndimage
from plotly.subplots import make_subplots

from ..resources.i8n.camillafir_i18n import t
from .plot_common import (
    GD_SMOOTH_OCT,
    PHASE_SMOOTH_OCT,
    _align_meas_to_target_window,
    _confidence_bad_segments,
    _filter_focus_band,
    _maybe_shift_to_abs,
    _plotly_js_path,
    _robust_axis_range,
    _view_mags_for_plot,
    calculate_clean_gd,
    logger,
    remove_ir_peak_delay,
    smooth_complex,
)


def generate_prediction_plot(
    orig_freqs,
    orig_mags,
    orig_phases,
    filt_ir,
    fs,
    title,
    save_filename=None,
    target_stats=None,
    mixed_split=None,
    zoom_hint="",
    create_full_html=True,
    return_fig: bool = False,
    plot_smoothing_level="Psychoacoustic",
):
    try:
        MIN_FFT_SIZE = 131072
        FFT_MUL = 4
        MAX_FFT_SIZE = None
        VIS_POINTS = 4000
        fig_height, fig_width = 1520, 1750

        n_fft = max(len(filt_ir) * FFT_MUL, MIN_FFT_SIZE)
        if MAX_FFT_SIZE is not None:
            n_fft = min(n_fft, int(MAX_FFT_SIZE))
        f_lin = scipy.fft.rfftfreq(n_fft, d=1 / fs)
        h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
        h_filt_display, filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)

        avg_t = target_stats.get("eff_target_db", 75) if target_stats else 75
        if target_stats and "smart_scan_range" in target_stats:
            match_range = target_stats.get("smart_scan_range", [500, 2000])
        else:
            match_range = target_stats.get("match_range", [500, 2000]) if target_stats else [500, 2000]
        try:
            f_win_min = float(match_range[0])
            f_win_max = float(match_range[1])
        except Exception:
            f_win_min, f_win_max = 500.0, 2000.0

        if target_stats and "measured_mags" in target_stats:
            f_stats = np.asarray(target_stats.get("freq_axis", []), dtype=float)
            m_stats = _maybe_shift_to_abs(target_stats.get("measured_mags", []), avg_t)
            t_stats = _maybe_shift_to_abs(target_stats.get("target_mags", []), avg_t) if "target_mags" in target_stats else None

            m_interp = np.interp(f_lin, f_stats, m_stats)
            if t_stats is not None and np.asarray(t_stats).size == f_stats.size:
                t_interp = np.interp(f_lin, f_stats, np.asarray(t_stats, dtype=float))
                m_interp = _align_meas_to_target_window(f_lin, m_interp, t_interp, f_win_min, f_win_max)

            m_lin_clean = _view_mags_for_plot(
                f_lin,
                m_interp,
                plot_smoothing_level=plot_smoothing_level,
            )
        else:
            m_raw = np.interp(f_lin, orig_freqs, orig_mags)
            m_lin_clean = _view_mags_for_plot(
                f_lin,
                m_raw,
                plot_smoothing_level=plot_smoothing_level,
            )

        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10 ** (m_lin_clean / 20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt

        plot_level_comp_db = 0.0
        ag_db = 0.0
        ah_db = 0.0
        try:
            if target_stats is not None:
                ag_db = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_db = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_db) and np.isfinite(ah_db):
                    plot_level_comp_db = -(ag_db + ah_db)
                elif np.isfinite(ag_db):
                    plot_level_comp_db = -ag_db
        except Exception:
            plot_level_comp_db = 0.0
            ag_db = 0.0
            ah_db = 0.0

        p_sm_export = _view_mags_for_plot(
            f_lin,
            20.0 * np.log10(np.abs(total_spec) + 1e-12),
            plot_smoothing_level=plot_smoothing_level,
        )
        p_sm_comp = p_sm_export.copy()
        if plot_level_comp_db != 0.0:
            p_sm_comp = p_sm_comp + float(plot_level_comp_db)
        filt_sm_phase = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
        ph_sm = (np.rad2deg(np.angle(filt_sm_phase)) + 180) % 360 - 180

        filt_sm_gd = smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT)
        gd_sm = calculate_clean_gd(f_lin, filt_sm_gd)

        filt_db_export = 20.0 * np.log10(np.abs(h_filt) + 1e-12)
        filt_db_comp = filt_db_export.copy()
        if plot_level_comp_db != 0.0:
            filt_db_comp = filt_db_comp + float(plot_level_comp_db)

        f_vis = np.geomspace(2, fs / 2, VIS_POINTS)

        m_vis = np.interp(f_vis, f_lin, m_lin_clean)
        p_vis_export = np.interp(f_vis, f_lin, p_sm_export)
        p_vis_comp = np.interp(f_vis, f_lin, p_sm_comp)
        ph_vis = np.interp(f_vis, f_lin, ph_sm)
        gd_vis = np.interp(f_vis, f_lin, gd_sm)
        filt_vis_export = np.interp(f_vis, f_lin, filt_db_export)
        filt_vis_comp = np.interp(f_vis, f_lin, filt_db_comp)
        focus_band = _filter_focus_band(f_vis, filt_vis_comp)
        gd_range = _robust_axis_range(
            f_vis,
            gd_vis,
            focus_band=focus_band,
            q_lo=0.02,
            q_hi=0.98,
            pad_ratio=0.16,
            min_span=6.0,
            max_span=120.0,
            include_zero=True,
        )

        fig = make_subplots(
            rows=5,
            cols=1,
            vertical_spacing=0.045,
            subplot_titles=(
                "<b>Magnitude & Alignment</b>",
                "<b>Filter Phase (delay compensated)</b>",
                "<b>Filter Group Delay (delay compensated)</b>",
                "<b>Filter (dB)</b>",
                "<b>A-FDW Effective BW (oct)</b>",
            ),
        )

        if target_stats and "smart_scan_range" in target_stats:
            s_min, s_max = target_stats["smart_scan_range"]
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=s_min,
                x1=s_max,
                y0=avg_t - 40,
                y1=avg_t + 60,
                fillcolor="rgba(200, 200, 200, 0.15)",
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

        try:
            ref_level = float(avg_t)

            if target_stats and "smart_scan_range" in target_stats:
                _r = target_stats.get("smart_scan_range", None)
            else:
                _r = target_stats.get("match_range", None)

            if isinstance(_r, (list, tuple)) and len(_r) == 2:
                win_label = f"{int(round(_r[0]))}-{int(round(_r[1]))} Hz"
            else:
                win_label = "level window"

            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=2.0,
                x1=fs / 2.0,
                y0=ref_level,
                y1=ref_level,
                line=dict(color="rgba(255,255,255,0.40)", width=1, dash="dot"),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"Level reference ({win_label})",
                    line=dict(color="rgba(255,255,255,0.40)", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        except Exception:
            pass
        if target_stats:
            try:
                cmin = float(target_stats.get("mag_c_min", 0.0) or 0.0)
                cmax = float(target_stats.get("mag_c_max", 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=cmin,
                        x1=cmax,
                        y0=avg_t - 40,
                        y1=avg_t + 60,
                        fillcolor="rgba(80, 140, 255, 0.08)",
                        layer="below",
                        line_width=0,
                        row=1,
                        col=1,
                    )
            except Exception:
                pass

        if target_stats and "confidence_mask" in target_stats:
            c_freqs = np.array(target_stats["freq_axis"])
            c_mask = np.array(target_stats["confidence_mask"])
            conf_line = (avg_t - 15) + (c_mask * 10)
            fig.add_trace(
                go.Scatter(
                    x=c_freqs,
                    y=conf_line,
                    name="Confidence",
                    line=dict(color="magenta", width=1),
                    opacity=0.3,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            try:
                for seg_start, seg_end in _confidence_bad_segments(c_freqs, c_mask, thr=0.35):
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=float(seg_start),
                        x1=float(seg_end),
                        y0=avg_t - 40,
                        y1=avg_t + 60,
                        fillcolor="rgba(255, 0, 0, 0.06)",
                        layer="below",
                        line_width=0,
                        row=1,
                        col=1,
                    )
            except Exception:
                pass

        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=m_vis,
                name="Measured",
                line=dict(color="rgba(0,0,255,0.4)", width=1.5),
            ),
            row=1,
            col=1,
        )

        if target_stats and "target_mags" in target_stats:
            t_mags = _maybe_shift_to_abs(target_stats.get("target_mags", []), avg_t)
            fig.add_trace(
                go.Scatter(
                    x=target_stats["freq_axis"],
                    y=t_mags,
                    name="Target",
                    line=dict(color="green", dash="dash", width=2.0),
                ),
                row=1,
                col=1,
            )

        idx_pred_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_export,
                name="Predicted (exported)",
                line=dict(color="orange", width=1.5),
            ),
            row=1,
            col=1,
        )

        idx_pred_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_comp,
                name="Predicted (compensated)",
                line=dict(color="orange", width=1.5, dash="dot"),
                visible=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=ph_vis,
                name="Filter Phase",
                line=dict(color="orange", width=0.9),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=gd_vis,
                name="Filter Group Delay",
                line=dict(color="orange", width=0.9),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        idx_filter_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=filt_vis_export,
                name="Filter dB (exported)",
                line=dict(color="red", width=0.9),
                showlegend=True,
                visible=True,
            ),
            row=4,
            col=1,
        )
        idx_filter_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=filt_vis_comp,
                name="Filter dB (compensated)",
                line=dict(color="red", width=0.9, dash="dot"),
                showlegend=True,
                visible=False,
            ),
            row=4,
            col=1,
        )

        try:
            if target_stats is not None:
                ag_txt = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_txt = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_txt) or np.isfinite(ah_txt):
                    fig.add_annotation(
                        x=0.01,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"Auto gain: {ag_txt:+.2f} dB | Headroom: {ah_txt:+.2f} dB | Filter delay removed: {filt_delay_ms:.2f} ms",
                        showarrow=False,
                        align="left",
                        font=dict(size=12, color="#f3f4f6"),
                        bgcolor="rgba(0,0,0,0.88)",
                        bordercolor="rgba(255,255,255,0.16)",
                        borderwidth=1,
                    )
        except Exception:
            pass

        try:
            n_tr = len(fig.data)
            vis_export = [True] * n_tr
            vis_comp = [True] * n_tr
            vis_both = [True] * n_tr

            vis_export[idx_pred_comp] = False
            vis_export[idx_pred_export] = True
            vis_export[idx_filter_comp] = False
            vis_export[idx_filter_export] = True

            vis_comp[idx_pred_export] = False
            vis_comp[idx_pred_comp] = True
            vis_comp[idx_filter_export] = False
            vis_comp[idx_filter_comp] = True

            vis_both[idx_pred_export] = True
            vis_both[idx_pred_comp] = True
            vis_both[idx_filter_export] = True
            vis_both[idx_filter_comp] = True

            fig.update_layout(
                margin=dict(t=120),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.01,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        bgcolor="rgba(0,0,0,0.92)",
                        bordercolor="rgba(255,255,255,0.15)",
                        borderwidth=1,
                        font=dict(size=12, color="#5f6061"),
                        pad=dict(t=4, r=6, b=4, l=6),
                        buttons=[
                            dict(
                                label=t("plot_level_exported"),
                                method="update",
                                args=[{"visible": vis_export}],
                            ),
                            dict(
                                label=t("plot_level_compensated"),
                                method="update",
                                args=[{"visible": vis_comp}],
                            ),
                            dict(
                                label=t("plot_level_both"),
                                method="update",
                                args=[{"visible": vis_both}],
                            ),
                        ],
                    )
                ],
            )
        except Exception:
            pass

        try:
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0.78)",
                    bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1,
                )
            )
        except Exception:
            pass

        if target_stats:
            try:
                cmin = float(target_stats.get("mag_c_min", 0.0) or 0.0)
                cmax = float(target_stats.get("mag_c_max", 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=cmin,
                        x1=cmax,
                        y0=-15,
                        y1=10,
                        fillcolor="rgba(80, 140, 255, 0.06)",
                        layer="below",
                        line_width=0,
                        row=4,
                        col=1,
                    )
            except Exception:
                pass

        bw_vis = None
        bw_vis_smooth = None
        bw_dbg = ""

        mode = "native"
        if target_stats:
            mode = str(target_stats.get("analysis_mode", "native")).lower()

        try:
            if target_stats:
                if mode == "comparison":
                    fx_raw = target_stats.get("cmp_freq_axis")
                    bw_raw = target_stats.get("cmp_afdw_bw_plot_oct", target_stats.get("cmp_afdw_bw_oct"))
                else:
                    fx_raw = target_stats.get("freq_axis")
                    bw_raw = target_stats.get("afdw_bw_plot_oct", target_stats.get("afdw_bw_oct"))

                if fx_raw is not None and bw_raw is not None:
                    fx = np.asarray(fx_raw, dtype=float)
                    bw = np.asarray(bw_raw, dtype=float)

                    if fx.size == bw.size and fx.size > 16:
                        bw_vis = np.interp(f_vis, fx, bw)
                        bw_vis = np.clip(bw_vis, 1.0 / 96.0, 1.0 / 3.0)
                        bw_vis_smooth = scipy.ndimage.gaussian_filter1d(bw_vis, sigma=5.0)
                        bw_vis_smooth = np.clip(bw_vis_smooth, 1.0 / 96.0, 1.0 / 3.0)
                        fig.add_trace(
                            go.Scatter(
                                x=f_vis,
                                y=bw_vis_smooth,
                                mode="lines",
                                fill="tozeroy",
                                fillcolor="rgba(56, 189, 248, 0.22)",
                                opacity=0.6,
                                line=dict(color="#38bdf8", width=2.2),
                                showlegend=False,
                                name="A-FDW BW",
                            ),
                            row=5,
                            col=1,
                        )
                    else:
                        bw_dbg = f"shape mismatch: fx={fx.size} bw={bw.size}"
                else:
                    bw_dbg = "missing afdw bw data"
            else:
                bw_dbg = "target_stats is None"
        except Exception as e:
            bw_dbg = f"{type(e).__name__}: {e}"

        if bw_vis is None:
            fig.add_annotation(
                text=f"No A-FDW BW data ({bw_dbg})",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="#f3f4f6"),
                bgcolor="rgba(0,0,0,0.78)",
                bordercolor="rgba(255,255,255,0.12)",
                borderwidth=1,
                row=5,
                col=1,
            )

        t_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in (1, 2, 3, 4, 5):
            fig.update_xaxes(matches="x", row=r, col=1)
            fig.update_xaxes(type="log", range=[np.log10(2), np.log10(20000)], tickvals=t_vals, row=r, col=1)

        fig.update_yaxes(range=[avg_t - 20, avg_t + 30], row=1, col=1)
        fig.update_yaxes(range=[-90, 90], row=2, col=1)
        if gd_range is not None:
            fig.update_yaxes(range=gd_range, row=3, col=1)
        fig.update_yaxes(range=[-30, 12], row=4, col=1)
        if bw_vis_smooth is not None and len(bw_vis_smooth) > 0:
            bw_data_min = float(np.min(bw_vis_smooth))
            bw_data_max = float(np.max(bw_vis_smooth))
            bw_span = bw_data_max - bw_data_min
            margin = max(bw_span * 0.3, 0.01)
            bw_lo = max(0.0, bw_data_min - margin)
            bw_hi = min(1.0 / 3.0, bw_data_max + margin)
            if bw_hi - bw_lo < 0.02:
                bw_lo = max(0.0, (bw_data_min + bw_data_max) / 2.0 - 0.01)
                bw_hi = bw_lo + 0.02
            fig.update_yaxes(range=[bw_lo, bw_hi], row=5, col=1)
        else:
            fig.update_yaxes(range=[0.0, 1.0 / 3.0], row=5, col=1)

        fig.update_yaxes(title_text="oct", row=5, col=1)

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#f3f4f6"),
            title_text=f"{title} Analysis",
            uirevision="keep",
        )

        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.18)",
            zerolinecolor="rgba(255,255,255,0.10)",
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.18)",
            zerolinecolor="rgba(255,255,255,0.10)",
        )

        if create_full_html:
            if _plotly_js_path():
                js_mode = "assets/plotly.min.js"
            else:
                js_mode = "cdn"
        else:
            if _plotly_js_path():
                js_mode = "assets/plotly.min.js"
            else:
                js_mode = "cdn"

        config = {
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": False,
        }

        html = fig.to_html(
            include_plotlyjs=js_mode,
            full_html=create_full_html,
            config=config,
        )

        _active_btn_js = """
<script>
(function() {
  function _fixActiveBtns(root) {
    var rects = (root || document).querySelectorAll('.updatemenu-item-rect');
    rects.forEach(function(r) {
      var fill = r.getAttribute('fill') || r.style.fill || '';
      if (fill && fill !== 'none' && fill !== 'rgba(0,0,0,0)' &&
          fill !== 'transparent' && fill.toLowerCase() !== '#000' &&
          fill.toLowerCase() !== 'black') {
        var isLight = (
          fill.startsWith('rgb(2') || fill.startsWith('rgb(1') ||
          fill.startsWith('#f') || fill.startsWith('#e') ||
          fill.startsWith('#d') || fill.startsWith('#c') ||
          fill === 'white' || fill === '#fff' || fill === '#ffffff'
        );
        if (isLight) {
          r.setAttribute('fill', '#1e3a5f');
          r.style.fill = '#1e3a5f';
          r.setAttribute('stroke', '#38bdf8');
        }
      }
    });
  }
  var _obs = new MutationObserver(function(muts) {
    muts.forEach(function(m) {
      if (m.type === 'attributes' && (m.attributeName === 'fill' || m.attributeName === 'style')) {
        _fixActiveBtns(m.target.closest('.updatemenu') || document);
      } else if (m.type === 'childList') {
        _fixActiveBtns(document);
      }
    });
  });
  function _attach() {
    _fixActiveBtns(document);
    document.querySelectorAll('.updatemenu').forEach(function(el) {
      _obs.observe(el, { subtree: true, attributes: true, attributeFilter: ['fill', 'style'], childList: true });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(_attach, 300); });
  } else {
    setTimeout(_attach, 300);
  }
})();
</script>"""
        if "</body>" in html:
            html = html.replace("</body>", _active_btn_js + "\n</body>", 1)
        else:
            html = html + _active_btn_js

        if bool(return_fig):
            return html, fig
        return html

    except Exception as e:
        msg = f"Visual Engine Error: {str(e)}"
        if bool(return_fig):
            return msg, None
        return msg
