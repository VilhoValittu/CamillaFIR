import io
import logging
import os
import sys

import matplotlib
import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.ndimage
from plotly.subplots import make_subplots

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..dsp.smoothing import apply_smoothing_std, psychoacoustic_smoothing
from ..resources.i8n.camillafir_i18n import t

PHASE_SMOOTH_OCT = 0.1
GD_SMOOTH_OCT = 0.1
GD_SMOOTH_SIGMA = 0.1
logger = logging.getLogger('CamillaFIR')

def _maybe_shift_to_abs(mags_db, avg_t_db):
    """Sisainen apufunktio: maybe shift to abs."""
    try:
        a = np.asarray(mags_db, dtype=float)
        if a.size == 0:
            return a
        med = float(np.nanmedian(a))
        if np.isfinite(med) and med < 40.0:
            return a + float(avg_t_db)
        return a
    except Exception:
        return np.asarray(mags_db, dtype=float)


def _align_meas_to_target_window(freqs_hz, meas_db, targ_db, f_min_hz, f_max_hz):
    """Sisainen apufunktio: align meas to target window."""
    try:
        f = np.asarray(freqs_hz, dtype=float)
        m = np.asarray(meas_db, dtype=float)
        t = np.asarray(targ_db, dtype=float)
        if f.size < 16 or m.size != f.size or t.size != f.size:
            return m
        f_min = float(f_min_hz); f_max = float(f_max_hz)
        if not (np.isfinite(f_min) and np.isfinite(f_max) and f_min > 0 and f_max > f_min):
            return m
        mask = (f >= f_min) & (f <= f_max) & np.isfinite(m) & np.isfinite(t)
        if np.count_nonzero(mask) < 20:
            return m
        off = float(np.median(m[mask] - t[mask]))
        if not np.isfinite(off):
            return m
        return m - off
    except Exception:
        return np.asarray(meas_db, dtype=float)


def _resource_path(rel_path: str) -> str:
    """Sisainen apufunktio: resource path."""
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)


def _plotly_js_path() -> str | None:
    """Sisainen apufunktio: plotly js path."""
    p = _resource_path(os.path.join("assets", "plotly.min.js"))
    return p if os.path.isfile(p) else None


def _confidence_bad_segments(
    freqs,
    conf_mask,
    *,
    thr: float = 0.35,
    min_width_hz: float = 30.0,
    min_gap_hz: float = 35.0,
    max_segments: int = 24,
):
    try:
        f = np.asarray(freqs, dtype=float)
        c = np.asarray(conf_mask, dtype=float)
    except Exception:
        return []
    if f.size != c.size or f.size < 8:
        return []

    valid = np.isfinite(f) & np.isfinite(c) & (f > 0.0)
    if np.count_nonzero(valid) < 8:
        return []
    f = f[valid]
    c = c[valid]
    bad = np.asarray(c < float(thr), dtype=bool)
    if not np.any(bad):
        return []

    raw = []
    in_seg = False
    seg_start = None
    for fx, is_bad in zip(f, bad):
        if is_bad and not in_seg:
            in_seg = True
            seg_start = float(fx)
        elif (not is_bad) and in_seg:
            in_seg = False
            seg_end = float(fx)
            if seg_start is not None and seg_end > seg_start:
                raw.append((float(seg_start), float(seg_end)))
    if in_seg and seg_start is not None and float(f[-1]) > float(seg_start):
        raw.append((float(seg_start), float(f[-1])))

    if not raw:
        return []

    merged = []
    for start, end in raw:
        if not merged:
            merged.append([float(start), float(end)])
            continue
        prev = merged[-1]
        if float(start - prev[1]) <= float(min_gap_hz):
            prev[1] = max(float(prev[1]), float(end))
        else:
            merged.append([float(start), float(end)])

    kept = [
        (float(start), float(end))
        for start, end in merged
        if float(end - start) >= float(min_width_hz)
    ]
    if not kept:
        kept = [(float(start), float(end)) for start, end in merged]

    if len(kept) > int(max_segments):
        kept = sorted(
            kept,
            key=lambda seg: (-(float(seg[1]) - float(seg[0])), float(seg[0])),
        )[: int(max_segments)]
        kept = sorted(kept, key=lambda seg: float(seg[0]))

    return kept


def smooth_complex(freqs, spec, oct_frac=1.0):
    """Kasittelee signaalia tai dataa: smooth complex."""
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    return real_s + 1j * imag_s


def calculate_clean_gd(freqs, complex_resp, *, sigma: float = GD_SMOOTH_SIGMA):
    """Laskee: calculate clean gd."""
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sigma_v = float(sigma)
    except Exception:
        sigma_v = float(GD_SMOOTH_SIGMA)
    if sigma_v > 0.0:
        gd_ms = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_v)
    return gd_ms


def remove_ir_peak_delay(freqs, complex_resp, ir, fs):
    """Poistaa nayttokuvista FIR:n bulk-viiveen IR-piikin perusteella."""
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        h = np.asarray(complex_resp, dtype=complex).reshape(-1)
        x = np.asarray(ir, dtype=float).reshape(-1)
        fs_v = float(fs)
        if f.size == 0 or h.size != f.size or x.size == 0 or fs_v <= 0.0:
            return h, 0.0
        peak_idx = int(np.argmax(np.abs(x)))
        if peak_idx <= 0:
            return h, 0.0
        delay_s = float(peak_idx) / fs_v
        rot = np.exp(1j * 2.0 * np.pi * f * delay_s)
        return h * rot, delay_s * 1000.0
    except Exception:
        return np.asarray(complex_resp, dtype=complex).reshape(-1), 0.0


def _filter_focus_band(freqs, filt_db, *, delta_db: float = 0.75) -> tuple[float, float] | None:
    """Arvioi suotimen aktiivisen taajuusalueen display-skaalausta varten."""
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        g = np.asarray(filt_db, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(g) & (f > 0.0)
        if np.count_nonzero(valid) < 16:
            return None
        fv = f[valid]
        gv = g[valid]
        order = np.argsort(fv, kind="mergesort")
        fv = fv[order]
        gv = gv[order]
        hi_start = int(max(0, round(0.8 * (fv.size - 1))))
        baseline = float(np.median(gv[hi_start:])) if hi_start < gv.size else float(np.median(gv))
        active = np.abs(gv - baseline) >= float(max(0.1, delta_db))
        if np.count_nonzero(active) < 8:
            return None
        lo = float(fv[np.where(active)[0][0]])
        hi = float(fv[np.where(active)[0][-1]])
        lo = max(float(fv[0]), lo / 1.35)
        hi = min(float(fv[-1]), hi * 1.6)
        if hi <= lo:
            return None
        return lo, hi
    except Exception:
        return None


def _robust_axis_range(
    freqs,
    values,
    *,
    focus_band: tuple[float, float] | None = None,
    q_lo: float = 0.03,
    q_hi: float = 0.97,
    pad_ratio: float = 0.18,
    min_span: float = 10.0,
    max_span: float | None = None,
    include_zero: bool = False,
):
    """Laskee robustin y-akselin alueen kuvaajille."""
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(y)
        if focus_band is not None:
            lo_f, hi_f = focus_band
            valid &= (f >= float(lo_f)) & (f <= float(hi_f))
        if np.count_nonzero(valid) < 12:
            valid = np.isfinite(y)
        if np.count_nonzero(valid) < 4:
            return None
        yy = y[valid]
        lo = float(np.quantile(yy, float(q_lo)))
        hi = float(np.quantile(yy, float(q_hi)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo = float(np.min(yy))
            hi = float(np.max(yy))
        if hi <= lo:
            mid = float(lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)
        span = float(hi - lo)
        lo -= span * float(pad_ratio)
        hi += span * float(pad_ratio)
        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        span = float(hi - lo)
        if span < float(min_span):
            mid = 0.5 * (hi + lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)
        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        if max_span is not None and (hi - lo) > float(max_span):
            center = 0.0 if include_zero else 0.5 * (hi + lo)
            lo = center - 0.5 * float(max_span)
            hi = center + 0.5 * float(max_span)
        return [float(lo), float(hi)]
    except Exception:
        return None


def _view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
    """Sisainen apufunktio: view mags for plot."""
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)

    if f.size == 0 or m.size == 0:
        return m

    psl = plot_smoothing_level

    if isinstance(psl, str) and ("psy" in psl.strip().lower()):
        return psychoacoustic_smoothing(f, m)

    try:
        lvl = int(psl)
    except Exception:
        return m

    lvl = max(1, lvl)
    out, _ = apply_smoothing_std(f, m, np.zeros_like(m), 1.0 / float(lvl))
    return out


def generate_prediction_plot(
    orig_freqs, orig_mags, orig_phases, filt_ir, fs, title,
    save_filename=None, target_stats=None, mixed_split=None,
    zoom_hint="", create_full_html=True, return_fig: bool = False,
    plot_smoothing_level="Psychoacoustic",
):
    """Rakentaa tai generoi: generate prediction plot."""
    try:
        MIN_FFT_SIZE = 131072
        FFT_MUL = 4
        MAX_FFT_SIZE = None
        VIS_POINTS = 4000
        fig_height, fig_width = 1520, 1750

        n_fft = max(len(filt_ir) * FFT_MUL, MIN_FFT_SIZE)
        if MAX_FFT_SIZE is not None:
            n_fft = min(n_fft, int(MAX_FFT_SIZE))
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
        h_filt_display, filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)
        
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        if target_stats and 'smart_scan_range' in target_stats:
            match_range = target_stats.get('smart_scan_range', [500, 2000])
        else:
            match_range = target_stats.get('match_range', [500, 2000]) if target_stats else [500, 2000]
        try:
            f_win_min = float(match_range[0])
            f_win_max = float(match_range[1])
        except Exception:
            f_win_min, f_win_max = 500.0, 2000.0

        if target_stats and 'measured_mags' in target_stats:
            f_stats = np.asarray(target_stats.get('freq_axis', []), dtype=float)
            m_stats = _maybe_shift_to_abs(target_stats.get('measured_mags', []), avg_t)
            t_stats = _maybe_shift_to_abs(target_stats.get('target_mags', []), avg_t) if 'target_mags' in target_stats else None

            m_interp = np.interp(f_lin, f_stats, m_stats)
            if t_stats is not None and np.asarray(t_stats).size == f_stats.size:
                t_interp = np.interp(f_lin, f_stats, np.asarray(t_stats, dtype=float))
                m_interp = _align_meas_to_target_window(f_lin, m_interp, t_interp, f_win_min, f_win_max)

            m_lin_clean = _view_mags_for_plot(
                f_lin, m_interp,
                plot_smoothing_level=plot_smoothing_level,
            )
        else:
            m_raw = np.interp(f_lin, orig_freqs, orig_mags)
            m_lin_clean = _view_mags_for_plot(
                f_lin, m_raw,
                plot_smoothing_level=plot_smoothing_level,
            )

        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**(m_lin_clean/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt

        # Plot-level compensation is ONLY for visualization alignment.
        # The exported filter IR already includes any applied auto-gain/headroom.
        # We therefore:
        #  - keep "Predicted" / phase / GD plots optionally compensated (for easier comparison),
        #  - but show BOTH filter magnitudes: Exported (baked) and Compensated (pre-gain-staging).
        plot_level_comp_db = 0.0
        ag_db = 0.0
        ah_db = 0.0
        try:
            if target_stats is not None:
                ag_db = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_db = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_db) and np.isfinite(ah_db):
                    # remove baked staging for *visual* comparison
                    plot_level_comp_db = -(ag_db + ah_db)
                elif np.isfinite(ag_db):
                    plot_level_comp_db = -ag_db
        except Exception:
            plot_level_comp_db = 0.0
            ag_db = 0.0
            ah_db = 0.0
        # Predicted magnitude (Exported)
        p_sm_export = _view_mags_for_plot(
            f_lin,
            20.0 * np.log10(np.abs(total_spec) + 1e-12),
            plot_smoothing_level=plot_smoothing_level,
        )
        # Predicted magnitude (Compensated): removes baked-in staging for easier comparison vs target/measured.
        p_sm_comp = p_sm_export.copy()
        if plot_level_comp_db != 0.0:
            p_sm_comp = p_sm_comp + float(plot_level_comp_db)
        # Phase and GD plots should describe the filter itself, not the
        # measured speaker+room response after filtering.
        filt_sm_phase = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
        ph_sm = (np.rad2deg(np.angle(filt_sm_phase)) + 180) % 360 - 180

        filt_sm_gd = smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT)
        gd_sm = calculate_clean_gd(f_lin, filt_sm_gd)
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)
        if plot_level_comp_db != 0.0:
            filt_db = filt_db + float(plot_level_comp_db)

        # Filter magnitude from IR FFT.
        # NOTE: this is the *exported/baked* filter magnitude (includes staging already).
        filt_db_export = 20.0 * np.log10(np.abs(h_filt) + 1e-12)
        # "Compensated" view removes baked auto-gain/headroom for easier comparison.
        filt_db_comp = filt_db_export.copy()
        if plot_level_comp_db != 0.0:
            filt_db_comp = filt_db_comp + float(plot_level_comp_db)

        f_vis = np.geomspace(2, fs/2, VIS_POINTS)
        
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
            rows=5, cols=1, vertical_spacing=0.045,
            subplot_titles=(
                "<b>Magnitude & Alignment</b>",
                "<b>Filter Phase (delay compensated)</b>",
                "<b>Filter Group Delay (delay compensated)</b>",
                "<b>Filter (dB)</b>",
                "<b>A-FDW Effective BW (oct)</b>",
                
            )
        )

        if target_stats and 'smart_scan_range' in target_stats:
            s_min, s_max = target_stats['smart_scan_range']
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=s_min, x1=s_max,
                          y0=avg_t-40, y1=avg_t+60,
                          fillcolor="rgba(200, 200, 200, 0.15)", layer="below", line_width=0, row=1, col=1)

        try:
            ref_level = float(avg_t)

            if target_stats and 'smart_scan_range' in target_stats:
                _r = target_stats.get('smart_scan_range', None)
            else:
                _r = target_stats.get('match_range', None)

            if isinstance(_r, (list, tuple)) and len(_r) == 2:
                win_label = f"{int(round(_r[0]))}-{int(round(_r[1]))} Hz"
            else:
                win_label = "level window"

            fig.add_shape(
                type="line",
                xref="x", yref="y",
                x0=2.0, x1=fs / 2.0,
                y0=ref_level, y1=ref_level,
                line=dict(color="rgba(255,255,255,0.40)", width=1, dash="dot"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"Level reference ({win_label})",
                    line=dict(color="rgba(255,255,255,0.40)", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=True
                ),
                row=1,
                col=1
            )
        except Exception:
            pass
        if target_stats:
            try:
                cmin = float(target_stats.get('mag_c_min', 0.0) or 0.0)
                cmax = float(target_stats.get('mag_c_max', 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=cmin, x1=cmax,
                        y0=avg_t-40, y1=avg_t+60,
                        fillcolor="rgba(80, 140, 255, 0.08)", layer="below", line_width=0,
                        row=1, col=1
                    )
            except Exception:
                pass


        if target_stats and 'confidence_mask' in target_stats:
            c_freqs = np.array(target_stats['freq_axis'])
            c_mask = np.array(target_stats['confidence_mask'])
            conf_line = (avg_t - 15) + (c_mask * 10)
            fig.add_trace(go.Scatter(x=c_freqs, y=conf_line, name='Confidence', 
                                     line=dict(color='magenta', width=1), opacity=0.3, hoverinfo='skip'), row=1, col=1)
            try:
                for seg_start, seg_end in _confidence_bad_segments(c_freqs, c_mask, thr=0.35):
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=float(seg_start), x1=float(seg_end),
                        y0=avg_t-40, y1=avg_t+60,
                        fillcolor="rgba(255, 0, 0, 0.06)", layer="below", line_width=0,
                        row=1, col=1
                    )
            except Exception:
                pass


        # --- Measured ---
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=m_vis,
                name='Measured',
                line=dict(color='rgba(0,0,255,0.4)', width=1.5)
            ),
            row=1, col=1
        )

        # --- Target ---
        if target_stats and 'target_mags' in target_stats:
            t_mags = _maybe_shift_to_abs(target_stats.get('target_mags', []), avg_t)
            fig.add_trace(
                go.Scatter(
                    x=target_stats['freq_axis'],
                    y=t_mags,
                    name='Target',
                    line=dict(color='green', dash='dash', width=2.0)
                ),
                row=1, col=1
            )

        # --- Predicted (Exported = baked IR level) ---
        idx_pred_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_export,   # <-- baked (IR FFT, auto gain mukana)
                name='Predicted (exported)',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )

        # --- Predicted (Compensated = without auto gain/headroom) ---
        idx_pred_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_comp,     # <-- plot_level_comp_db poistettu
                name='Predicted (compensated)',
                line=dict(color='orange', width=1.5, dash='dot'),
                visible=False     # oletuksena piilossa
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=ph_vis,
                name="Filter Phase",
                line=dict(color='orange', width=0.9),
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
                line=dict(color='orange', width=0.9),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # ---- Filter panel: show BOTH baked/exported and compensated views (pro-level clarity) ----
        # Exported (baked): what you actually load into DSP (IR FFT).
        idx_filter_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis, y=filt_vis_export,
                name="Filter dB (exported)",
                line=dict(color='red', width=0.9),
                showlegend=True,
                visible=True,
            ),
            row=4, col=1
        )
        # Compensated: removes applied auto gain/headroom for visual comparison vs target.
        idx_filter_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis, y=filt_vis_comp,
                name="Filter dB (compensated)",
                line=dict(color='red', width=0.9, dash="dot"),
                showlegend=True,
                visible=False,
            ),
            row=4, col=1
        )

        # Small annotation with staging values (when available)
        try:
            if target_stats is not None:
                ag_txt = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_txt = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_txt) or np.isfinite(ah_txt):
                    fig.add_annotation(
                        x=0.01, y=0.98, xref="paper", yref="paper",
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

        # toggle: switch BOTH Predicted + Filter between Exported/Compensated/Both
        try:
            n_tr = len(fig.data)
            vis_export = [True] * n_tr
            vis_comp = [True] * n_tr
            vis_both = [True] * n_tr

            # Exported-only: show exported traces, hide compensated traces
            vis_export[idx_pred_comp] = False
            vis_export[idx_pred_export] = True
            vis_export[idx_filter_comp] = False
            vis_export[idx_filter_export] = True

            # Compensated-only
            vis_comp[idx_pred_export] = False
            vis_comp[idx_pred_comp] = True
            vis_comp[idx_filter_export] = False
            vis_comp[idx_filter_comp] = True

            # Both
            vis_both[idx_pred_export] = True
            vis_both[idx_pred_comp] = True
            vis_both[idx_filter_export] = True
            vis_both[idx_filter_comp] = True

            fig.update_layout(
            margin=dict(t=120),  # enemmän ylätilaa

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
                    ]
                )
            ],
        )
        except Exception:
            pass

        # Make legend a bit more helpful when using "Both"
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
                cmin = float(target_stats.get('mag_c_min', 0.0) or 0.0)
                cmax = float(target_stats.get('mag_c_max', 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=cmin, x1=cmax,
                        y0=-15, y1=10,
                        fillcolor="rgba(80, 140, 255, 0.06)", layer="below", line_width=0,
                        row=4, col=1
                    )
            except Exception:
                pass

        
        bw_vis = None
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
                        bw_vis = np.clip(bw_vis, 1.0/96.0, 1.0/3.0)
                        bw_vis_smooth = scipy.ndimage.gaussian_filter1d(bw_vis, sigma=5.0)
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
                            row=5, col=1
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
                col=1
            )


        t_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in (1, 2, 3, 4, 5):
            fig.update_xaxes(matches="x", row=r, col=1)
            fig.update_xaxes(type="log", range=[np.log10(2), np.log10(20000)], tickvals=t_vals, row=r, col=1)


        fig.update_yaxes(range=[avg_t-20, avg_t+30], row=1, col=1)
        fig.update_yaxes(range=[-90, 90], row=2, col=1)
        if gd_range is not None:
            fig.update_yaxes(range=gd_range, row=3, col=1)
        fig.update_yaxes(range=[-30, 12], row=4, col=1)
        if bw_vis is not None and len(bw_vis) > 0:
            bw_lo = 0.0
            bw_hi = min(1.0/3.0,  float(np.max(bw_vis)) * 1.1)
            if bw_hi - bw_lo < 1e-6:
                bw_lo, bw_hi = (0.0, 1.0/3.0)
            fig.update_yaxes(range=[bw_lo, bw_hi], row=5, col=1) 
        else:
            fig.update_yaxes(range=[0.0, 1.0/3.0], row=5, col=1)

        fig.update_yaxes(title_text="oct", row=5, col=1)

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#f3f4f6"),
            title_text=f"{title} Analysis",
            uirevision="keep"
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
            "doubleClick": False
        }

        html = fig.to_html(
            include_plotlyjs=js_mode,
            full_html=create_full_html,
            config=config
        )

        # Plotly's activecolor property is not supported in the installed version.
        # Inject a MutationObserver to override the white active-button fill that
        # Plotly sets via inline SVG attributes.
        _active_btn_js = """
<script>
(function() {
  function _fixActiveBtns(root) {
    var rects = (root || document).querySelectorAll('.updatemenu-item-rect');
    rects.forEach(function(r) {
      var fill = r.getAttribute('fill') || r.style.fill || '';
      // Plotly sets active buttons to a light/white fill; darken them
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


def plotly_fig_to_png(fig, *, scale=2, width=None, height=None):
    """Funktio: plotly fig to png."""
    try:
        import plotly.io as pio
        kwargs = {"format": "png", "scale": float(scale)}
        if width is not None:
            kwargs["width"] = int(width)
        if height is not None:
            kwargs["height"] = int(height)
        return pio.to_image(fig, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Plotly PNG export failed: {e}"
        )


def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    """Kuvan rakennus generate combined plot mpl."""
    try:
        n_fft = len(filt_ir); f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs); h_filt = scipy.fft.rfft(filt_ir)
        h_filt_display, _filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        m_lin = np.interp(f_lin, orig_freqs, orig_mags); p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        filt_phase = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
        filt_phase_deg = (np.rad2deg(np.angle(filt_phase)) + 180) % 360 - 180
        filt_gd = calculate_clean_gd(f_lin, smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT))
        filt_db = 20*np.log10(np.abs(h_filt)+1e-12)
        focus_band = _filter_focus_band(f_lin, filt_db)
        gd_range = _robust_axis_range(
            f_lin,
            filt_gd,
            focus_band=focus_band,
            q_lo=0.02,
            q_hi=0.98,
            pad_ratio=0.16,
            min_span=6.0,
            max_span=120.0,
            include_zero=True,
        )
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18))
        ax1.semilogx(orig_freqs, orig_mags + offset, 'b:', alpha=0.3)
        ax1.semilogx(f_lin, psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12)), 'orange', linewidth=2)
        if target_stats: ax1.semilogx(target_stats['freq_axis'], target_stats['target_mags'], 'g--')
        
        if target_stats and 'smart_scan_range' in target_stats:
            f_min, f_max = target_stats['smart_scan_range']
            ax1.axvline(f_min, color='red', linestyle='--', alpha=0.6, label=f'Final Min: {f_min:.0f}Hz')
            ax1.axvline(f_max, color='green', linestyle='--', alpha=0.6, label=f'Final Max: {f_max:.0f}Hz')
            ax1.legend(loc='upper right', fontsize='small')
        
        
        ax1.set_ylim(avg_t-15, avg_t+15)
        ax2.semilogx(f_lin, filt_phase_deg, 'orange', linewidth=0.9)
        ax2.set_ylim(-90, 90)
        ax3.semilogx(f_lin, filt_gd, 'orange', linewidth=0.9)
        if gd_range is not None:
            ax3.set_ylim(gd_range)
        ax4.semilogx(f_lin, filt_db, 'r', linewidth=0.9)
        
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xscale('log'); ax.set_xlim(20, 20000); ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Visualization error ({title}): {e}")
        return b""


__all__ = [
    'PHASE_SMOOTH_OCT',
    'GD_SMOOTH_OCT',
    'GD_SMOOTH_SIGMA',
    '_maybe_shift_to_abs',
    '_align_meas_to_target_window',
    '_resource_path',
    '_plotly_js_path',
    '_confidence_bad_segments',
    'smooth_complex',
    'calculate_clean_gd',
    'remove_ir_peak_delay',
    '_filter_focus_band',
    '_robust_axis_range',
    '_view_mags_for_plot',
    'generate_prediction_plot',
    'plotly_fig_to_png',
    'generate_combined_plot_mpl',
]
