import logging
from textwrap import dedent

from pywebio.output import (
    put_collapse,
    put_error,
    put_file,
    put_html,
    put_info,
    put_markdown,
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
update_status_notices = _app.update_status_notices
update_auto_selected_bar = _app.update_auto_selected_bar


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
    auto_cache_path=None,
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
    psl = data.get('plot_smoothing_level', 'Psychoacoustic')

    if isinstance(psl, str):
        if psl == "Psychoacoustic":
            psl_str = t("smooth_safe_reference")
        else:
            psl_str = psl
    else:
        psl_str = f"1/{int(psl)} octave"

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

        def _metric_cell(value, compare=None):
            cmp = compare if compare is not None else value
            if cmp is None:
                cmp = "-"
            return {"render": value, "compare": str(cmp)}

        def _metric_row(label, left, right, *, left_compare=None, right_compare=None):
            return {
                "label": str(label),
                "left": _metric_cell(left, left_compare),
                "right": _metric_cell(right, right_compare),
            }

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

        def _fmt_ai_match(ai):
            match = ai.get("match", None)
            if match is None:
                return _metric_cell("n/a", "n/a")
            return _metric_cell(f"{float(match):.1f}%", f"{float(match):.3f}")

        def _fmt_ai_score(ai):
            score = ai.get("score", None)
            if score is None:
                return _metric_cell("n/a", "n/a")
            return _metric_cell(f"{float(score):.3f}/100", f"{float(score):.3f}")

        def _auto_float(v, default=float("nan")):
            try:
                x = float(v)
                if x == x and abs(x) != float("inf"):
                    return float(x)
            except Exception:
                pass
            return float(default)

        def _auto_int(v, default=0):
            try:
                return int(v)
            except Exception:
                return int(default)

        def _auto_choice_text(method):
            try:
                key = str(method or "").strip().lower()
            except Exception:
                key = ""
            mapping = {
                "cache_signature_hit": "cache hit",
                "cache_measurement": "measurement cache seed",
                "cache_signature": "signature cache seed",
                "trial_with_cache_wildcard": "cache wildcard winner",
                "top3x10_trials": "trial comparison",
                "top3x10_trials_rank_tie_composite": "trial comparison with tie-break",
                "fit_rms": "quick fit preselect",
            }
            return str(mapping.get(key, key or "unknown"))


        l_boost_pre, l_auto_gain, l_net_boost = _boost_diag(l_st_f)
        r_boost_pre, r_auto_gain, r_net_boost = _boost_diag(r_st_f)

        """
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
        """

        acoustic_summary = []
        if l_ai.get("score") is not None and r_ai.get("score") is not None:
            acoustic_summary.append(f"**Average acoustic score:** {avg_pred:.3f}/100")
        if l_match is not None and r_match is not None:
            acoustic_summary.append(f"**Average target match:** {avg_match:.1f}%")

        _put_metric_collapse(
            " Acoustic summary",
            [
                _metric_row("Target Level", f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"),
                _metric_row(
                    "Smart Scan Range",
                    f"{l_st_f.get('smart_scan_range', [0, 0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0, 0])[1]:.0f} Hz",
                    f"{r_st_f.get('smart_scan_range', [0, 0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0, 0])[1]:.0f} Hz",
                ),
                _metric_row(
                    "Leveling Tilt",
                    _fmt_tilt(l_st_f),
                    _fmt_tilt(r_st_f),
                    left_compare=_auto_float(l_st_f.get('tilt_slope_db_per_oct', float("nan")), float("nan")),
                    right_compare=_auto_float(r_st_f.get('tilt_slope_db_per_oct', float("nan")), float("nan")),
                ),
                _metric_row("Offset to Meas.", f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"),
                {"label": "Target Match", "left": _fmt_ai_match(l_ai), "right": _fmt_ai_match(r_ai)},
                _metric_row("Acoustic Confidence", f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"),
                {"label": "Acoustic Score", "left": _fmt_ai_score(l_ai), "right": _fmt_ai_score(r_ai)},
                _metric_row("Estimated RT60", f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"),
            ],
            summary_lines=acoustic_summary,
            open=False,
        )

        _put_metric_collapse(
            " Gain and headroom",
            [
                _metric_row(
                    "TDC (Temporal Decay Control)",
                    (
                        f"ON ({float(data.get('tdc_strength', 0)):.0f}%, -{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                        if bool(data.get('enable_tdc', False)) else "OFF"
                    ),
                    (
                        f"ON ({float(data.get('tdc_strength', 0)):.0f}%, -{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                        if bool(data.get('enable_tdc', False)) else "OFF"
                    ),
                ),
                _metric_row(
                    "Auto Gain Margin",
                    f"{float(l_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB",
                    f"{float(r_st_f.get('gain_margin_db', data.get('gain', 0.0)) or 0.0):.2f} dB",
                ),
                _metric_row(
                    "Applied Auto Gain",
                    f"{float(l_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB",
                    f"{float(r_st_f.get('auto_global_gain_db', 0.0) or 0.0):.2f} dB",
                ),
                _metric_row(
                    "Extra Headroom",
                    f"{float(l_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB",
                    f"{float(r_st_f.get('auto_headroom_db', 0.0) or 0.0):.2f} dB",
                ),
                _metric_row(
                    "Net Boost (pre -> net)",
                    f"{l_boost_pre:.2f} dB -> {l_net_boost:.2f} dB",
                    f"{r_boost_pre:.2f} dB -> {r_net_boost:.2f} dB",
                ),
                _metric_row(
                    "Final Max (post gain)",
                    f"{float(l_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB",
                    f"{float(r_st_f.get('final_max_db', 0.0) or 0.0):.2f} dB",
                ),
            ],
            summary_lines=["Net boost is reduced by baked auto gain (preamp) to prevent clipping."],
            open=False,
        )

        try:
            mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            mode_u = "BASIC"
        auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
        auto_meta = data.get("_auto_mode_meta", None)
        if auto_enabled and isinstance(auto_meta, dict):
            bm = dict(auto_meta.get("best_metrics", {}) or {})
            top = list(auto_meta.get("top", []) or [])
            tc_meta = dict(data.get("_auto_target_curve_meta", {}) or {})
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

            def _f3(v) -> str:
                x = _af(v, float("nan"))
                return f"{x:.3f}" if x == x else "n/a"

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

            tc_selected = str(tc_meta.get("selected_hc_mode", data.get("hc_mode", "n/a")) or "n/a")
            tc_method = _auto_choice_text(tc_meta.get("selection_method", ""))
            search_fs = _ai(auto_meta.get("search_fs", 0), 0)
            search_taps = _ai(auto_meta.get("search_taps", 0), 0)
            exc_seed = _af(
                auto_meta.get(
                    "auto_exc_seed_freq_hz",
                    data.get("_auto_exc_seed_freq_hz", data.get("_auto_exc_freq_hz", float("nan"))),
                ),
                float("nan"),
            )
            exc_final = _af(
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
                ["Goal / basis", f"{str(auto_meta.get('auto_goal', 'balanced') or 'balanced')} / {str(auto_meta.get('selection_basis', 'rank_score') or 'rank_score')}"],
                ["Trials", f"{trials_ok}/{trials_total}"],
                ["Search grid", f"{search_fs} Hz / {search_taps} taps"],
                ["Target curve", tc_selected],
                ["Target selection", tc_method],
                ["Excursion protection", exc_text],
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
                if "mixed_freq" in best_preset:
                    preset_rows.append(["Mixed split", f"{_af(best_preset.get('mixed_freq'), 0.0):.1f} Hz"])
                if "phase_limit" in best_preset:
                    preset_rows.append(["Phase limit", f"{_af(best_preset.get('phase_limit'), 0.0):.1f} Hz"])
                if "fdw_cycles" in best_preset:
                    preset_rows.append(["FDW", f"{_af(best_preset.get('fdw_cycles'), 0.0):.1f}"])
                if "tdc_strength" in best_preset and "tdc_max_reduction_db" in best_preset:
                    preset_rows.append([
                        "TDC preset",
                        f"{_af(best_preset.get('tdc_strength'), 0.0):.0f}% / -{_af(best_preset.get('tdc_max_reduction_db'), 0.0):.1f} dB",
                    ])
                if "reg_strength" in best_preset:
                    preset_rows.append(["Regularization", f"{_af(best_preset.get('reg_strength'), 0.0):.2f}"])
                if "max_boost" in best_preset:
                    preset_rows.append(["Max boost", f"{_af(best_preset.get('max_boost'), 0.0):.1f} dB"])
                if "mag_c_min" in best_preset and "mag_c_max" in best_preset:
                    preset_rows.append([
                        "Magnitude correction range",
                        f"{_af(best_preset.get('mag_c_min'), 0.0):.0f}-{_af(best_preset.get('mag_c_max'), 0.0):.0f} Hz",
                    ])
            if preset_rows:
                auto_content.append(put_table([["Best preset", "Value"], *preset_rows]))

            put_collapse(" Automatic mode winner", auto_content, open=False)

            # Top-3 target curves (auto target selection phase).
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
                        -_af(dict((it or {}).get("best_metrics", {}) or {}).get("rank_score", 0.0), 0.0),
                        -_af((it or {}).get("avg_rank_score", 0.0), 0.0),
                        _af((it or {}).get("fit_rms_db", 1e9), 1e9),
                    ),
                )
                for idx, it in enumerate(tc_eval_sorted[:3], start=1):
                    bm_tc = dict((it or {}).get("best_metrics", {}) or {})
                    tc_rows.append([
                        f"{idx}",
                        str((it or {}).get("hc_mode", "n/a") or "n/a"),
                        f"{_af(bm_tc.get('rank_score', 0.0), 0.0):.3f}",
                        f"{_af((it or {}).get('avg_rank_score', 0.0), 0.0):.3f}",
                        _f3((it or {}).get('fit_rms_db', float('nan'))),
                        _f3((it or {}).get('preselect_score', float('nan'))),
                        _f3((it or {}).get('boost_penalty', float('nan'))),
                        _f3((it or {}).get('asym_penalty_db', float('nan'))),
                        f"{_ai((it or {}).get('trials_ok', 0), 0)}/{_ai((it or {}).get('trials_total', 0), 0)}",
                    ])
            else:
                tc_cands = list(tc_meta.get("candidates", []) or [])
                tc_cands_sorted = sorted(
                    list(tc_cands),
                    key=lambda it: (
                        _af((it or {}).get("preselect_score", (it or {}).get("fit_rms_db", 1e9)), 1e9),
                        _af((it or {}).get("fit_rms_db", 1e9), 1e9),
                    ),
                )
                for idx, it in enumerate(tc_cands_sorted[:3], start=1):
                    tc_rows.append([
                        f"{idx}",
                        str((it or {}).get("hc_mode", "n/a") or "n/a"),
                        "n/a",
                        "n/a",
                        _f3((it or {}).get('fit_rms_db', float('nan'))),
                        _f3((it or {}).get('preselect_score', float('nan'))),
                        _f3((it or {}).get('boost_penalty', float('nan'))),
                        _f3((it or {}).get('asym_penalty_db', float('nan'))),
                        "n/a",
                    ])
            if len(tc_rows) > 1:
                put_collapse(
                    " Target curve top-3",
                    [put_table(tc_rows)],
                    open=False,
                )

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
                    ])
                put_collapse(
                    " Automatic mode top-5 (by rank)",
                    [put_table(top_rows)],
                    open=False,
                )

    #    put_markdown(f"###  {t('rep_header')}")

        def _phase_clamp_new(st: dict) -> str:
            try:
                lim = float((st or {}).get('phase_corr_clamp_deg', 0.0) or 0.0)
                bef = float((st or {}).get('phase_corr_max_before_deg', 0.0) or 0.0)
                clipped = bool((st or {}).get('phase_corr_clipped', False))
                if lim <= 0.0:
                    return "-"
                if clipped:
                    return f"max={bef:.1f} deg -> {lim:.1f} deg"
                return f"max={bef:.1f} deg (limit {lim:.1f} deg)"
            except Exception:
                return "-"

        def _xo_fc_wrapped_new(st: dict) -> str:
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
                    k = f"xo{i}_dphi_wrapped_deg@fc"
                    if k not in st:
                        continue
                    try:
                        v = float(st.get(k))
                    except Exception:
                        continue
                    lbl = f"{int(round(freqs[i-1]))}Hz" if i <= len(freqs) and freqs[i-1] is not None else f"XO{i}"
                    items.append(f"{lbl}:{v:+.1f} deg")
                return " | ".join(items) if items else "-"
            except Exception:
                return "-"

        def _xo_fc_gd_new(st: dict) -> str:
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
                    k = f"xo{i}_dgd_ms@fc"
                    if k not in st:
                        continue
                    try:
                        v = float(st.get(k))
                    except Exception:
                        continue
                    lbl = f"{int(round(freqs[i-1]))}Hz" if i <= len(freqs) and freqs[i-1] is not None else f"XO{i}"
                    items.append(f"{lbl}:{v:+.2f} ms")
                return " | ".join(items) if items else "-"
            except Exception:
                return "-"

        def _xo_phase_model_new(st: dict) -> str:
            try:
                s = (st or {}).get("xo_summary", None)
                if s is None or str(s).strip() == "":
                    return "-"
                return str(s)
            except Exception:
                return "-"

        def _xo_diff_raw_new(st: dict) -> str:
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

        def _xo_fc_gd_badge_new(st: dict) -> str:
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
                    label, bg, fg, title = "LOW", "rgba(46, 125, 50, 0.15)", "rgba(46, 125, 50, 1.0)", "Small XO delta GD at fc (typically subtle)."
                elif worst < 1.5:
                    label, bg, fg, title = "MED", "rgba(255, 143, 0, 0.15)", "rgba(255, 143, 0, 1.0)", "Moderate XO delta GD at fc (often audible improvement with XO phase correction)."
                else:
                    label, bg, fg, title = "HIGH", "rgba(211, 47, 47, 0.15)", "rgba(211, 47, 47, 1.0)", "Large XO delta GD at fc (aggressive crossover / lots of time smear)."
                return (
                    f"<span title='{title}' style='display:inline-block; margin-left:6px; padding:1px 6px; "
                    f"border-radius:10px; font-size:11px; font-weight:600; background:{bg}; color:{fg}; "
                    f"vertical-align:middle;'>{label}</span>"
                )
            except Exception:
                return ""

        def _hpf_diff_raw_new(st: dict) -> str:
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

        def _hpf_model_new(st: dict) -> str:
            try:
                s = (st or {}).get("hpf_summary", None)
                if s is None or str(s).strip() == "":
                    return "-"
                return str(s)
            except Exception:
                return "-"

        def _format_ir_window_new(data: dict) -> str:
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

        def _mixed_blend_new(st: dict, key: str) -> str:
            try:
                v = (st or {}).get(key, None)
                if v is None:
                    return "-"
                return f"{float(v):.1f} Hz"
            except Exception:
                return "-"

        def _gd_limiter_new(st: dict) -> str:
            try:
                s = st or {}
                enabled = bool(s.get("gd_limiter_enabled", s.get("gd_grad_limiter_enabled", False)))
                reason = str(s.get("gd_limiter_reason", s.get("gd_grad_limiter_reason", "unknown")) or "unknown")
                lim = s.get("gd_limiter_limit_ms_per_oct", s.get("gd_grad_limit_ms_per_oct", None))
                lim_txt = "n/a" if lim is None else f"{float(lim):.2f}"
                return f"{'ON' if enabled else 'OFF'} (reason={reason}, limit={lim_txt} ms/oct)"
            except Exception:
                return "n/a"

        def _gd_grad_max_new(st: dict) -> str:
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

        _put_metric_collapse(
            " Filter setup",
            [
                _metric_row("Length", f"{data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)", f"{data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)"),
                _metric_row("Resolution", f"{data['fs']/data['taps']:.2f} Hz", f"{data['fs']/data['taps']:.2f} Hz"),
                _metric_row("IR window", _format_ir_window_new(data), _format_ir_window_new(data)),
                _metric_row("FDW", f"{data['fdw_cycles']}", f"{data['fdw_cycles']}"),
                _metric_row("House curve", f"{data['hc_mode']} - {data.get('hc_source', 'Unknown')}", f"{data['hc_mode']} - {data.get('hc_source', 'Unknown')}"),
                _metric_row("Correction range", f"{data['mag_c_min']}-{data['mag_c_max']} Hz", f"{data['mag_c_min']}-{data['mag_c_max']} Hz"),
                _metric_row("Filter type", f"{data['filter_type']}", f"{data['filter_type']}"),
                _metric_row("HPF model", _hpf_model_new(l_st_f), _hpf_model_new(r_st_f)),
                _metric_row("Mixed blend split", _mixed_blend_new(l_st_f, "mixed_blend_split_hz"), _mixed_blend_new(r_st_f, "mixed_blend_split_hz")),
                _metric_row("Mixed blend transition", _mixed_blend_new(l_st_f, "mixed_blend_transition_hz"), _mixed_blend_new(r_st_f, "mixed_blend_transition_hz")),
                _metric_row("Smoothing view", psl_str, psl_str),
                _metric_row("Leveling algo", str(data.get('lvl_algo', '') or "-"), str(data.get('lvl_algo', '') or "-")),
            ],
            open=False,
        )

        xo_gd_left = _xo_fc_gd_new(l_st_f)
        xo_gd_right = _xo_fc_gd_new(r_st_f)
        xo_gd_badge = _xo_fc_gd_badge_new(l_st_f) or _xo_fc_gd_badge_new(r_st_f)
        _put_metric_collapse(
            " Phase and crossover",
            [
                _metric_row("XO phase model", _xo_phase_model_new(l_st_f), _xo_phase_model_new(r_st_f)),
                _metric_row("XO delta phi@fc (wrapped)", _xo_fc_wrapped_new(l_st_f), _xo_fc_wrapped_new(r_st_f)),
                {
                    "label": "XO delta GD@fc",
                    "left": _metric_cell(put_html(f"{xo_gd_left} {xo_gd_badge}" if xo_gd_badge else xo_gd_left), xo_gd_left),
                    "right": _metric_cell(xo_gd_right, xo_gd_right),
                },
                _metric_row("XO effect (theoretical raw)", _xo_diff_raw_new(l_st_f), _xo_diff_raw_new(r_st_f)),
                _metric_row("HPF effect (theoretical raw)", _hpf_diff_raw_new(l_st_f), _hpf_diff_raw_new(r_st_f)),
                _metric_row("Phase correction clamp", _phase_clamp_new(l_st_f), _phase_clamp_new(r_st_f)),
                _metric_row("GD limiter", _gd_limiter_new(l_st_f), _gd_limiter_new(r_st_f)),
                _metric_row("A/B GD-gradient max", _gd_grad_max_new(l_st_f), _gd_grad_max_new(r_st_f)),
            ],
            open=False,
        )

        if False:
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
            path_rows = [["Item", "Path"]]
            if saved_filters_dir:
                path_rows.append([t("paths_export_folder"), str(saved_filters_dir)])
            if auto_cache_path:
                path_rows.append([t("paths_auto_mode_cache"), str(auto_cache_path)])
            if len(path_rows) > 1:
                path_rows[0] = [t("paths_item"), t("paths_path")]
                put_markdown(f"### {t('paths_title')}")
                put_table(path_rows)
        except Exception:
            pass

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

        done_msg = t('done_msg')
        if run_started_at is not None:
            try:
                total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
                update_status_notices(
                    summary_text=done_msg,
                    info_text="",
                )
                update_status(f"{done_status} | {total_s:.1f} s")
            except Exception:
                update_status_notices(summary_text=done_msg, info_text="")
                update_status(done_status)
        else:
            update_status_notices(summary_text=done_msg, info_text="")
            update_status(done_status)
        set_processbar('bar', 1.0)
    return main
