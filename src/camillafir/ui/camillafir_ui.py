import logging
from textwrap import dedent

from pywebio.output import (
    put_collapse,
    put_error,
    put_file,
    put_html,
    put_info,
    put_markdown,
    put_success,
    put_table,
    put_tabs,
    set_processbar,
    use_scope,
)
from pywebio.pin import pin

from ..resources.i8n.camillafir_i18n import t
from . import app as _app
from . import camillafir_plot as plots

logger = logging.getLogger("CamillaFIR")
PROGRAM_NAME = "CamillaFIR"
VERSION = ""
MAX_SAFE_BOOST = 8.0


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    g = globals()
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    return _app.build_app(
        process_run=process_run,
        PROGRAM_NAME=PROGRAM_NAME,
        VERSION=VERSION,
        MAX_SAFE_BOOST=MAX_SAFE_BOOST,
    )


main = _app.main
update_status = _app.update_status


def _log_df_smoothing_for_fs(cfg, fs_v, df_on):
    if df_on:
        try:
            fsmooth = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 12)) or 12)
            if fsmooth <= 0: fsmooth = 12
            base_sigma = 60 // (fsmooth / 12 if fsmooth > 0 else 1)
            df_ref = 44100.0 / 65536.0
            sigma_hz = base_sigma * df_ref
            df_cur = (fs_v / cfg.num_taps)
            sigma_bins = sigma_hz / df_cur if df_cur > 0 else base_sigma

            logger.info(
                f"{fs_v//1000} kHz -> DF smoothing ON "
                f"(sigma = {sigma_bins:.1f} bins -> {sigma_hz:.1f} Hz)"
            )
        except Exception:
            logger.info(f"{fs_v//1000} kHz -> DF smoothing ON")
    else:
        logger.info(f"{fs_v//1000} kHz -> DF smoothing OFF")

def _pin_get(key, default=None):
    """Sisainen apufunktio: pin get."""
    try:
        v = pin.get(key, None)
        if v is None:
            return default
        return v
    except Exception:
        try:
            return pin[key]
        except Exception:
            return default

def _json_safe(obj, *, _depth=0, _max_depth=12):
    """Sisainen apufunktio: json safe."""
    try:
        if _depth > _max_depth:
            return str(obj)
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj

        try:
            import numpy as _np
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    ks = str(k)
                except Exception:
                    ks = "key"
                out[ks] = _json_safe(v, _depth=_depth + 1, _max_depth=_max_depth)
            return out

        if isinstance(obj, (list, tuple)):
            return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]

        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return str(obj)

        return str(obj)
    except Exception:
        return str(obj)


def _build_diagnostics_dict(data, fs_v, l_st, r_st):
    """Sisainen apufunktio: build diagnostics dict."""
    def _leveling_block(st):
        if not isinstance(st, dict):
            return {}
        win = st.get("smart_scan_range", None)
        try:
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                win = [float(win[0]), float(win[1])]
            else:
                win = None
        except Exception:
            win = None
        return {
            "method": st.get("offset_method", None),
            "window_hz": win,
            "offset_db": st.get("offset_db", None),
            "eff_target_db": st.get("eff_target_db", None),
            "tilt_slope_db_per_oct": st.get("tilt_slope_db_per_oct", None),
            "avg_confidence_pct": st.get("avg_confidence", None),
        }

    diag = {
        "schema_version": 1,
        "meta": {
            "program": PROGRAM_NAME,
            "version": VERSION,
            "fs_hz": int(fs_v),
            "taps": int(float(data.get("taps", 0) or 0)),
            "filter_type": str(data.get("filter_type", "") or ""),
            "multi_rate": bool(data.get("multi_rate_opt", False)),
            "ir_export_window_mode": str(data.get("ir_export_window_mode", "") or ""),
            "ir_export_window_tag": str(_irwin_tag(data.get("ir_export_window_mode"))),
        },
        "settings": _json_safe(data),
        "leveling": {
            "stereo_link": bool(data.get("stereo_link", False)),
            "left": _leveling_block(l_st),
            "right": _leveling_block(r_st),
        },
        "left": _json_safe(l_st),
        "right": _json_safe(r_st),
    }
    return diag

def _render_results(
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
):
    import time
    _render_started_at = time.perf_counter()
    if run_started_at is not None:
        try:
            _elapsed_now = max(0.0, float(time.perf_counter() - float(run_started_at)))
            update_status(f"{t('stat_plot')} | {_elapsed_now:.1f} s")
        except Exception:
            update_status(t('stat_plot'))
    else:
        update_status(t('stat_plot'))
    time.sleep(0.05)
    set_processbar('bar', 0.8)
    print("plot_smoothing_level =", data.get("plot_smoothing_level"))
    print("filter_smooth =", data.get("filter_smooth"))
    

    psl = data.get('plot_smoothing_level', 'Psychoacoustic')

    if isinstance(psl, str):
        if psl == "Psychoacoustic":
            psl_str = t("smooth_safe_reference")
        else:
            psl_str = psl
    else:
        psl_str = f"1/{int(psl)} octave"

    print("psl_str =", psl_str)

    with use_scope('results', clear=True):
        if l_st_f is None or r_st_f is None:
            put_error("Error: No results captured.")
            return
        

      
            
        l_ai = plots.calc_ai_summary_from_stats(l_st_f)
        r_ai = plots.calc_ai_summary_from_stats(r_st_f)

        l_score = float(l_ai.get("score") or 0.0)
        r_score = float(r_ai.get("score") or 0.0)
        avg_pred = (l_score + r_score) / 2.0
        avg_orig = avg_pred
        improvement = 0.0

        l_match = l_ai.get("match")
        r_match = r_ai.get("match")
        if (l_match is None) or (r_match is None):
            avg_match = 0.0
        else:
            avg_match = (float(l_match) + float(r_match)) / 2.0

        def _fmt_tilt(st, warn_thr=1.5):
            tilt = st.get('tilt_slope_db_per_oct', None)
            if tilt is None:
                return "—"
            try:
                tilt = float(tilt)
                if abs(tilt) > warn_thr:
                    return put_html(
                        f'<span title="Large broadband tilt detected during leveling, house curve not suitable for speaker in room.">'
                        f'{tilt:+.2f} dB/oct ⚠️'
                        f'</span>'
                    )
                else:
                    return f"{tilt:+.2f} dB/oct"
            except Exception:
                return "—"

        def _boost_diag(st):
            boost_pre = float(st.get("boost_peak_db", 0.0) or 0.0)
            auto_gain = float(st.get("auto_global_gain_db", 0.0) or 0.0)
            net_boost = boost_pre + auto_gain
            return boost_pre, auto_gain, net_boost


        l_boost_pre, l_auto_gain, l_net_boost = _boost_diag(l_st_f)
        r_boost_pre, r_auto_gain, r_net_boost = _boost_diag(r_st_f)

        put_table([
            ['Speaker', 'L', 'R'],
            ['Target Level', f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"],
            ['Smart Scan Range',
             f"{l_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz",
             f"{r_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz"],
            ['Leveling Tilt',_fmt_tilt(l_st_f),_fmt_tilt(r_st_f)],
            ['Offset to Meas.', f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"],
            ['Acoustic Confidence', f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"],
            ['Estimated RT60', f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"],
            ['TDC (Temporal Decay Control)',
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             ),
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             )
            ],
            ['Auto Gain Margin', f"{float(l_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB", f"{float(r_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB"],
            ['Applied Auto Gain', f"{float(l_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB", f"{float(r_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB"],
            ['Extra Headroom', f"{float(l_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB", f"{float(r_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB"],
            ['Net Boost (pre → net)',
            f"{l_boost_pre:.2f} dB → {l_net_boost:.2f} dB",
            f"{r_boost_pre:.2f} dB → {r_net_boost:.2f} dB"
            ],

            ['Note',
            "Net boost is reduced by baked auto gain (preamp) to prevent clipping.",
            ""
            ],
            ['Final Max (post gain)', f"{float(l_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB", f"{float(r_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB"],
        ])

        try:
            mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            mode_u = "BASIC"
        auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
        auto_meta = data.get("_auto_mode_meta", None)
        if auto_enabled and isinstance(auto_meta, dict):
            bm = dict(auto_meta.get("best_metrics", {}) or {})
            top = list(auto_meta.get("top", []) or [])
            trials_ok = int(auto_meta.get("trials_ok", 0) or 0)
            trials_total = int(auto_meta.get("trials_total", 0) or 0)

            def _af(v, default=0.0):
                try:
                    x = float(v)
                    if x == x and abs(x) != float("inf"):
                        return float(x)
                except Exception:
                    pass
                return float(default)

            def _ai(v, default=0):
                try:
                    return int(v)
                except Exception:
                    return int(default)

            rank_sc = _af(bm.get("rank_score", 0.0), 0.0)
            avg_sc = _af(bm.get("avg_score", 0.0), 0.0)
            dsp_pen = _af(bm.get("dsp_penalty", 0.0), 0.0)
            exc_pen = _af(bm.get("exc_penalty", 0.0), 0.0)
            boost_db = _af(bm.get("max_net_boost_db", 0.0), 0.0)
            events_n = _ai(bm.get("events_total", 0), 0)
            events_sev = _af(bm.get("events_severity", 0.0), 0.0)
            lr_delta = _af(bm.get("lr_delta_score", 0.0), 0.0)
            prepost = _af(bm.get("ir_pre_post_energy_ratio_max", float("nan")), float("nan"))
            mode_ripple = _af(bm.get("mode_ripple_db", float("nan")), float("nan"))
            lf_rms_20_200 = _af(bm.get("realized_rms_20_200_db", float("nan")), float("nan"))

            put_markdown("### CamillaFIR automatic mode - why this preset won")
            put_info(
                f"Best winner: rank {rank_sc:.3f}/100, avg {avg_sc:.3f}, "
                f"boost {boost_db:.2f} dB, dsp_pen {dsp_pen:.2f}, exc_pen {exc_pen:.2f}, "
                f"events {events_n}, event_sev {events_sev:.2f}, "
                f"L/R delta {lr_delta:.3f} (trials {trials_ok}/{trials_total})."
            )
            put_markdown(
                "Selection order: **Rank score** -> **avg score** -> **pre/post ratio** -> "
                "**mode ripple** -> **LF RMS (20-200 Hz)** -> **net boost**."
            )

            best_rows = [
                ["Metric", "Best preset"],
                ["Rank score (primary)", f"{rank_sc:.3f}/100"],
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
            put_table(best_rows)

            if top:
                rank_best = max((_af(dict(it.get("metrics", {}) or {}).get("rank_score", 0.0), 0.0) for it in top), default=rank_sc)
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
                    "Exc pen",
                ]]
                for idx, item in enumerate(top[:5], start=1):
                    m = dict(item.get("metrics", {}) or {})
                    prepost_t = _af(m.get("ir_pre_post_energy_ratio_max", float("nan")), float("nan"))
                    mode_t = _af(m.get("mode_ripple_db", float("nan")), float("nan"))
                    rms_t = _af(m.get("realized_rms_20_200_db", float("nan")), float("nan"))
                    top_rows.append([
                        f"{idx}",
                        f"{_ai(m.get('trial', 0), 0)}",
                        f"{_af(m.get('rank_score', 0.0), 0.0):.3f}",
                        f"{_af(m.get('avg_score', 0.0), 0.0):.3f}",
                        f"{prepost_t:.4f}" if prepost_t == prepost_t else "n/a",
                        f"{mode_t:.3f}" if mode_t == mode_t else "n/a",
                        f"{rms_t:.3f}" if rms_t == rms_t else "n/a",
                        f"{_af(m.get('max_net_boost_db', 0.0), 0.0):.2f}",
                        f"{_af(m.get('lr_delta_score', 0.0), 0.0):.3f}",
                        f"{_af(m.get('dsp_penalty', 0.0), 0.0):.2f}",
                        f"{_af(m.get('exc_penalty', 0.0), 0.0):.2f}",
                    ])
                put_markdown("#### Automatic mode top-5 (by rank)")
                put_table(top_rows)

        put_markdown(f"###  {t('rep_header')}")
        with put_collapse(" DSP info"):
            def _phase_clamp_str(st: dict) -> str:
                try:
                    lim = float((st or {}).get('phase_corr_clamp_deg', 0.0) or 0.0)
                    bef = float((st or {}).get('phase_corr_max_before_deg', 0.0) or 0.0)
                    clipped = bool((st or {}).get('phase_corr_clipped', False))
                    if lim <= 0.0:
                        return "—"
                    if clipped:
                        return f"max={bef:.1f}° -> {lim:.1f}°"
                    return f"max={bef:.1f}° (limit {lim:.1f}°)"
                except Exception:
                    return "—"

            def _xo_fc_wrapped_str(st: dict) -> str:
                """Sisainen apufunktio: xo fc wrapped str."""
                try:
                    if not isinstance(st, dict) or not st:
                        return "—"

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
                        k = f"xo{i}_dphi_wrapped_deg@fc"
                        if k not in st:
                            continue
                        try:
                            v = float(st.get(k))
                        except Exception:
                            continue
                        f_lbl = None
                        if i <= len(freqs) and freqs[i-1] is not None:
                            f_lbl = f"{int(round(freqs[i-1]))}Hz"
                        else:
                            f_lbl = f"XO{i}"
                        items.append(f"{f_lbl}:{v:+.1f}°")

                    return " | ".join(items) if items else "—"
                except Exception:
                    return "—"
            def _xo_fc_gd_str(st: dict) -> str:
                """Sisainen apufunktio: xo fc gd str."""
                try:
                    if not isinstance(st, dict) or not st:
                        return "—"

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
                        k = f"xo{i}_dgd_ms@fc"
                        if k not in st:
                            continue
                        try:
                            v = float(st.get(k))
                        except Exception:
                            continue
                        
                        if i <= len(freqs) and freqs[i-1] is not None:
                            lbl = f"{int(round(freqs[i-1]))}Hz"
                        else:
                            lbl = f"XO{i}"
                        items.append(f"{lbl}:{v:+.2f} ms")

                    return " | ".join(items) if items else "—"
                except Exception:
                    return "—"

            def _xo_phase_model_str(st: dict) -> str:
                try:
                    s = (st or {}).get("xo_summary", None)
                    if s is None or str(s).strip() == "":
                        return "—"
                    return str(s)
                except Exception:
                    return "—"

            def _xo_diff_raw_str(st: dict) -> str:
                try:
                    p = (st or {}).get("xo_diff_raw_max_phase_deg", None)
                    pf = (st or {}).get("xo_diff_raw_max_phase_hz", None)
                    pfc = (st or {}).get("xo_diff_raw_max_phase_xo_fc_hz", None)
                    g = (st or {}).get("xo_diff_raw_max_gd_ms", None)
                    gf = (st or {}).get("xo_diff_raw_max_gd_hz", None)
                    gfc = (st or {}).get("xo_diff_raw_max_gd_xo_fc_hz", None)
                    if p is None and g is None:
                        return "—"
                    parts = []
                    if p is not None and pf is not None:
                        if pfc is not None:
                            parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz (XO {float(pfc):.0f} Hz)")
                        else:
                            parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz")
                    if g is not None and gf is not None:
                        if gfc is not None:
                            parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz (XO {float(gfc):.0f} Hz)")
                        else:
                            parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz")
                    return " | ".join(parts) if parts else "—"
                except Exception:
                    return "—"

            def _xo_fc_gd_badge(st: dict) -> str:
                """Sisainen apufunktio: xo fc gd badge."""
                try:
                    if not isinstance(st, dict) or not st:
                        return ""
                    vals = []
                    for i in range(1, 6):
                        k = f"xo{i}_dgd_ms@fc"
                        if k not in st:
                            continue
                        try:
                            vals.append(abs(float(st.get(k))))
                        except Exception:
                            pass
                    if not vals:
                        return ""
                    worst = max(vals)

                    if worst < 0.7:
                        label = "LOW"
                        bg = "rgba(46, 125, 50, 0.15)"
                        fg = "rgba(46, 125, 50, 1.0)"
                        title = "Small XO ΔGD@fc (typically subtle)."
                    elif worst < 1.5:
                        label = "MED"
                        bg = "rgba(255, 143, 0, 0.15)"
                        fg = "rgba(255, 143, 0, 1.0)"
                        title = "Moderate XO ΔGD@fc (often audible improvement with XO phase correction)."
                    else:
                        label = "HIGH"
                        bg = "rgba(211, 47, 47, 0.15)"
                        fg = "rgba(211, 47, 47, 1.0)"
                        title = "Large XO ΔGD@fc (aggressive crossover / lots of time smear)."

                    return (
                        f"<span title='{title}' "
                        f"style='display:inline-block; margin-left:6px; padding:1px 6px; "
                        f"border-radius:10px; font-size:11px; font-weight:600; "
                        f"background:{bg}; color:{fg}; vertical-align:middle;'>"
                        f"{label}</span>"
                    )
                except Exception:
                    return "" 

            def _hpf_diff_raw_str(st: dict) -> str:
                try:
                    p = (st or {}).get("hpf_diff_raw_max_phase_deg", None)
                    pf = (st or {}).get("hpf_diff_raw_max_phase_hz", None)
                    g = (st or {}).get("hpf_diff_raw_max_gd_ms", None)
                    gf = (st or {}).get("hpf_diff_raw_max_gd_hz", None)
                    if p is None and g is None:
                        return "—"
                    parts = []
                    if p is not None and pf is not None:
                        parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz")
                    if g is not None and gf is not None:
                        parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz")
                    return " | ".join(parts) if parts else "—"
                except Exception:
                    return "—"


            def _hpf_model_str(st: dict) -> str:
                try:
                    s = (st or {}).get("hpf_summary", None)
                    if s is None or str(s).strip() == "":
                        return "—"
                    return str(s)
                except Exception:
                    return "—"
            def _format_ir_window(data: dict) -> str:
                """Sisainen apufunktio: format ir window."""
                mode = str(data.get('ir_export_window_mode', '') or '').lower()

                if mode == 'rew_asym':
                    l = data.get('ir_window_left', None)
                    r = data.get('ir_window_right', data.get('ir_window', None))
                    try:
                        if l is not None and r is not None:
                            return f"Asymmetric (Left {float(l):.1f} ms, Right {float(r):.1f} ms)"
                    except Exception:
                        pass
                    return "Asymmetric"

                return "Auto (adaptive)"

            _xo_gd_line = (
                f"XO ΔGD@fc: L {_xo_fc_gd_str(l_st_f)} | R {_xo_fc_gd_str(r_st_f)}"
                f"{_xo_fc_gd_badge(l_st_f) or _xo_fc_gd_badge(r_st_f)}"
            )

            def _mixed_blend_str(st: dict, key: str) -> str:
                try:
                    v = (st or {}).get(key, None)
                    if v is None:
                        return "—"
                    return f"{float(v):.1f}"
                except Exception:
                    return "—"

            def _gd_limiter_str(st: dict) -> str:
                try:
                    s = st or {}
                    enabled = bool(s.get("gd_limiter_enabled", s.get("gd_grad_limiter_enabled", False)))
                    reason = str(s.get("gd_limiter_reason", s.get("gd_grad_limiter_reason", "unknown")) or "unknown")
                    lim = s.get("gd_limiter_limit_ms_per_oct", s.get("gd_grad_limit_ms_per_oct", None))
                    if lim is None:
                        lim_txt = "n/a"
                    else:
                        lim_txt = f"{float(lim):.2f}"
                    return f"{'ON' if enabled else 'OFF'} (reason={reason}, limit={lim_txt} ms/oct)"
                except Exception:
                    return "n/a"

            def _gd_grad_max_str(st: dict) -> str:
                try:
                    s = st or {}
                    for k in (
                        "gd_limiter_max_grad_ms_per_oct",
                        "gd_grad_limiter_max_grad_ms_per_oct",
                        "gd_limiter_max_grad_after_ms_per_oct",
                        "gd_grad_limiter_max_grad_after_ms_per_oct",
                        "gd_limiter_max_grad_before_ms_per_oct",
                        "gd_grad_limiter_max_grad_before_ms_per_oct",
                    ):
                        try:
                            v = float(s.get(k, None))
                            if v == v and abs(v) < float("inf"):
                                return f"{v:.2f} ms/oct"
                        except Exception:
                            continue
                    return "n/a"
                except Exception:
                    return "n/a"

            put_markdown(dedent(f"""
            - **Lenght:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
            - **Resolution:** {data['fs']/data['taps']:.2f} Hz
            - **IR window:** {_format_ir_window(data)}
            - **FDW:** {data['fdw_cycles']}
            - **House curve:** {data['hc_mode']} — {data.get('hc_source', 'Unknown')} ({data['mag_c_min']}-{data['mag_c_max']} Hz)
            - **Filter type:** {data['filter_type']}
            - **Mixed blend split:** L {_mixed_blend_str(l_st_f, "mixed_blend_split_hz")} Hz | R {_mixed_blend_str(r_st_f, "mixed_blend_split_hz")} Hz
            - **Mixed blend transition:** L {_mixed_blend_str(l_st_f, "mixed_blend_transition_hz")} Hz | R {_mixed_blend_str(r_st_f, "mixed_blend_transition_hz")} Hz
            - **XO phase model:** L {_xo_phase_model_str(l_st_f)} | R {_xo_phase_model_str(r_st_f)}
            - **XO Δφ@fc (wrapped):** L {_xo_fc_wrapped_str(l_st_f)} | R {_xo_fc_wrapped_str(r_st_f)}
            - {_xo_gd_line}
            - **XO effect (theoretical raw):**
              - **L:** {_xo_diff_raw_str(l_st_f)}
              - **R:** {_xo_diff_raw_str(r_st_f)}
            - **HPF effect (theoretical raw):**
              - **L:** {_hpf_diff_raw_str(l_st_f)}
              - **R:** {_hpf_diff_raw_str(r_st_f)}
            - **Phase correction clamp:** L {_phase_clamp_str(l_st_f)} | R {_phase_clamp_str(r_st_f)}
            - **GD limiter:** L {_gd_limiter_str(l_st_f)} | R {_gd_limiter_str(r_st_f)}
            - **A/B GD-gradient max:** L {_gd_grad_max_str(l_st_f)} | R {_gd_grad_max_str(r_st_f)}
            - **Smoothing view:** {psl_str}
            - **Leveling algo:** {data.get('lvl_algo', '')}
            """), sanitize=False)

        if dash_html_l is None:
            dash_html_l = plots.generate_prediction_plot(
                f_l, m_l, p_l, l_imp_f, data['fs'], "Left",
                None, l_st_f, data['mixed_freq'], "low",
                create_full_html=False,
                plot_smoothing_level=data.get('plot_smoothing_level', 'Psychoacoustic')
            )
        if dash_html_r is None:
            dash_html_r = plots.generate_prediction_plot(
                f_r, m_r, p_r, r_imp_f, data['fs'], "Right",
                None, r_st_f, data['mixed_freq'], "low",
                create_full_html=False,
                plot_smoothing_level=data.get('plot_smoothing_level', 'Psychoacoustic')
            )

        put_tabs([
            {'title': 'Left Channel', 'content': put_html(dash_html_l)},
            {'title': 'Right Channel', 'content': put_html(dash_html_r)}
        ])
        put_file(fname, zip_buffer.getvalue(), label=" DOWNLOAD FILTER ZIP")

        try:
            rows = [['Stage', 'Time (s)']]
            p = dict(perf_stats or {})
            read_s = float(p.get("read_s", 0.0) or 0.0)
            dsp_s = float(p.get("dsp_s", 0.0) or 0.0)
            zip_s = float(p.get("zip_png_s", 0.0) or 0.0)
            render_s = max(0.0, float(time.perf_counter() - _render_started_at))
            rows.append(['Read', f"{read_s:.2f}"])
            rows.append(['DSP L/R', f"{dsp_s:.2f}"])
            rows.append(['ZIP + PNG', f"{zip_s:.2f}"])
            rows.append(['Results render', f"{render_s:.2f}"])

            if isinstance(per_fs_stats, dict) and per_fs_stats:
                for fs_k in sorted(per_fs_stats.keys(), key=lambda x: int(x)):
                    st = per_fs_stats.get(fs_k, {}) or {}
                    fs_dsp = float(st.get("dsp_s", 0.0) or 0.0)
                    fs_zip = float(st.get("zip_png_s", 0.0) or 0.0)
                    rows.append([f"DSP @ {int(fs_k)} Hz", f"{fs_dsp:.2f}"])
                    rows.append([f"ZIP+PNG @ {int(fs_k)} Hz", f"{fs_zip:.2f}"])

            if run_started_at is not None:
                total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
                known_s = max(0.0, read_s + dsp_s + zip_s + render_s)
                other_s = max(0.0, total_s - known_s)
                rows.append(['Other', f"{other_s:.2f}"])
                rows.append(['Total', f"{total_s:.2f}"])

            put_markdown("### Timing")
            put_table(rows)
        except Exception:
            pass
        
        done_status = t('stat_done')
        if saved_filters_dir:
            try:
                done_status = done_status.format(path=str(saved_filters_dir))
            except Exception:
                done_status = f"{done_status} {saved_filters_dir}"

        put_success(t('done_msg'))
        if run_started_at is not None:
            try:
                total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
                put_info(f"Total time: {total_s:.1f} s")
                update_status(f"{done_status} | {total_s:.1f} s")
            except Exception:
                update_status(done_status)
        else:
            update_status(done_status)
        set_processbar('bar', 1.0)
    return main
