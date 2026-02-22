import json
import logging
import os
from . import camillafir_plot as plots
from ..config.camillafir_convolver_configs import generate_hlc_config, generate_raspberry_yaml
from ..dsp.smoothing import AFDW_BW_MAX_OCT, AFDW_BW_MIN_OCT
logger = logging.getLogger("CamillaFIR")

TEST_MODE = 1

TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(data.get('enable_afdw', False))
        enable_tdc  = bool(data.get('enable_tdc',  False))
        tdc_strength = float(data.get('tdc_strength', 0.0) or 0.0)
        fdw_cycles = float(data.get('fdw_cycles', 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0
        fdw_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, fdw_oct_width)))
        afdw_min_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, afdw_min_oct_width)))

        df_on = bool(data.get('df_smoothing', False))
        df_ref = 44100.0 / 65536.0
        fsmooth = data.get('filter_smooth', data.get('smoothing_level', 12))
        base_sigma = 60 // (fsmooth / 12 if (fsmooth or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = (float(fs_v) / float(data.get('taps', 65536) or 65536))
        sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Sample rate: {int(fs_v)} Hz\n"
        try:
            psl = str(data.get("plot_smoothing_level", "Psychoacoustic") or "Psychoacoustic").strip()
            psl_display = "CamillaFIR Reference" if "psy" in psl.lower() else psl
            summary_content += f"Plot smoothing: {psl_display}\n"
        except Exception:
            pass


        if enable_afdw:
            summary_content += "FDW mode: Adaptive (A-FDW)\n"
            summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"
            summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width -> {afdw_min_oct_width:.3f})\n"
            summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
        else:
            summary_content += "FDW mode: Fixed\n"
            summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"

        summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
        if enable_tdc:
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength/100.0:.3f})\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"
    except Exception as e:
        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Could not compute effective params: {type(e).__name__}: {e}\n"

    return summary_content

def _append_acoustic_events(summary_content, l_st, r_st):
    for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
        reflections = st.get('reflections') or []
        if reflections:
            summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
            summary_content += (
                "Note: 'Path delta' is an equivalent path-length from dt.\n"
                "Reflections: time-of-flight equivalent extra path.\n"
                "Resonances: not a physical distance.\n"
            )
            summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'dt (ms)':<12} {'Path delta (m)':<14}\n"
            summary_content += "-" * 50 + "\n"
            for rev in reflections:
                freq = float(rev.get('freq', 0) or 0)
                ev_type = str(rev.get('type', 'Event') or 'Event')
                gd_error = float(rev.get('gd_error', 0) or 0)
                dist = float(rev.get('dist', 0) or 0)
                try:
                    et = ev_type.strip().lower()
                except Exception:
                    et = ""
                if "reson" in et:
                    summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {'n/a':<14}\n"
                else:
                    summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {dist:<14}\n"
        summary_content += f"\n=== HEADROOM MANAGEMENT ({side}) ===\n"
        summary_content += f"Auto Gain Margin: {float(st.get('gain_margin_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Auto Gain: {float(st.get('auto_global_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Normalize: {'ON' if bool(st.get('do_normalize', False)) else 'OFF'}\n"
        summary_content += f"Peak Gain (pre-headroom): {float(st.get('peak_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Headroom: {float(st.get('auto_headroom_db', 0.0)):.2f} dB\n"
        summary_content += f"Final Max (filter+auto_gain+headroom): {float(st.get('final_max_db', 0.0)):.2f} dB\n"
        summary_content += f"\n=== BOOST/CUT DIAGNOSTICS ({side}) ===\n"
        _mb_eff = float(st.get('max_boost_db_effective', st.get('max_boost_db', 0.0)) or 0.0)
        _mb_user = float(st.get('max_boost_db_user', st.get('max_boost_db', 0.0)) or 0.0)
        _mb_cap = float(st.get('max_safe_boost_db', 0.0) or 0.0)
        if _mb_cap > 0.0 and _mb_user > _mb_eff + 1e-9:
            summary_content += f"Config: max_boost_db={_mb_eff:.2f} dB (user={_mb_user:.2f}, cap={_mb_cap:.2f}), "
        else:
            summary_content += f"Config: max_boost_db={_mb_eff:.2f} dB, "
        summary_content += f"max_cut_db={float(st.get('max_cut_db', 0.0)):.2f} dB\n"
        summary_content += f"Config: low_bass_cut_hz={float(st.get('low_bass_cut_hz', 0.0)):.1f} Hz, "
        summary_content += f"exc_prot={'ON' if bool(st.get('exc_prot', False)) else 'OFF'}, "
        summary_content += f"exc_freq={float(st.get('exc_freq', 0.0)):.1f} Hz, "
        summary_content += f"max_slope_db_per_oct={float(st.get('max_slope_db_per_oct', 0.0)):.1f}\n"
        summary_content += f"Result (post-clamp): boost_peak={float(st.get('boost_peak_db', 0.0)):.2f} dB, "
        summary_content += f"cut_peak={float(st.get('cut_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_bins', 0))}\n"
        summary_content += f"Net boost peak (post global/headroom): {float(st.get('net_boost_peak_db', 0.0)):.2f} dB\n"
        summary_content += f"Candidate (pre-softclip): boost_peak={float(st.get('boost_candidate_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_candidate_bins', 0))}, "
        summary_content += f"lowbass_boost_bins={int(st.get('boost_candidate_bins_lowbass', 0))}, "
        summary_content += f"excprot_boost_bins={int(st.get('boost_candidate_bins_excprot', 0))}\n"
        summary_content += f"Boost blocked reason: {str(st.get('boost_blocked_reason', 'n/a'))}\n"
        summary_content += f"\n=== CLAMP DIAGNOSTICS ({side}) ===\n"
        summary_content += f"{str(st.get('clamp_summary', 'n/a'))}\n"
        summary_content += (
            f"soft_clip: boost_bins={int(st.get('softclip_boost_bins', 0))}, "
            f"cut_bins={int(st.get('softclip_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('softclip_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('softclip_worst_over_cut_db', 0.0)):.2f} dB\n"
        )
        summary_content += (
            f"hard_clamp: boost_bins={int(st.get('hardclamp_boost_bins', 0))}, "
            f"cut_bins={int(st.get('hardclamp_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('hardclamp_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('hardclamp_worst_over_cut_db', 0.0)):.2f} dB\n"
        )


        probes = st.get("stage_probes") or {}
        if isinstance(probes, dict) and probes:
            summary_content += f"\n=== STAGE CHECKPOINTS ({side}) ===\n"
            summary_content += f"{'Stage':<22} {'BoostPk':>8} {'CutPk':>8} {'BoostBins':>10} {'CutBins':>8} {'NetBoostPk':>11}\n"
            summary_content += "-" * 75 + "\n"
            order = [
                "after_gain_apply",
                "after_lowbass_policy",
                "after_slope",
                "after_fade",
                "pre_softclip",
                "post_softclip",
                "post_hardclamp",
            ]
            for key in order:
                p = probes.get(key)
                if not isinstance(p, dict):
                    continue
                stage = str(p.get("stage", key))
                bpk = float(p.get("boost_peak_db", 0.0) or 0.0)
                cpk = float(p.get("cut_peak_db", 0.0) or 0.0)
                bb  = int(p.get("boost_bins", 0) or 0)
                cb  = int(p.get("cut_bins", 0) or 0)
                nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
                summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"

            summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
            summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

            pk_hz = st.get('bass_first_mode_peak_hz', None)
            pk_sc = st.get('bass_first_mode_peak_score', None)
            if (pk_hz is not None) and (pk_sc is not None):
                summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
            else:
                summary_content += "Mode peak: n/a\n"

            summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"

            rm_max = st.get('bass_first_roommode_max_20_200', None)
            rel_mean = st.get('bass_first_rel_mean_20_200', None)
            rel_min = st.get('bass_first_rel_min_20_200', None)
            conf_eff_mean = st.get('bass_first_conf_eff_mean_20_200', None)
            conf_eff_min = st.get('bass_first_conf_eff_min_20_200', None)
            floor_applied = bool(st.get('bass_first_conf_floor_applied', False))
            if (rm_max is not None) or (rel_mean is not None) or (rel_min is not None) or (conf_eff_mean is not None):
                summary_content += (
                    f"BF masks (20-200): "
                    f"roommode_max={float(rm_max or 0.0):.3f}, "
                    f"rel_mean(raw)={float(rel_mean or 0.0):.3f}, "
                    f"rel_min(raw)={float(rel_min or 0.0):.3f}, "
                    f"conf_eff_mean={float(conf_eff_mean or 0.0):.3f}, "
                    f"conf_eff_min={float(conf_eff_min or 0.0):.3f}, "
                    f"conf_floor_applied={'YES' if floor_applied else 'NO'}\n"
                )

            src = st.get("bass_first_source", None)
            if isinstance(src, str) and src.strip():
                summary_content += f"BassFirst source: {src.strip()}\n"


 

    return summary_content

def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    write_dashboards: bool = True,
    irw_tag: str = "auto",
    ui_dashboards: dict | None = None,
):
    sum_name = f"Summary_{ft_short}_{fs_v}Hz.txt"
    l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.png"
    r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.png"

    summary_content = plots.format_summary_content(data, l_st, r_st)
    try:
        hc_src = str(data.get('hc_source', '') or '').strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except Exception:
        pass
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    try:
        summary_content += "\n=== LEVELING ===\n"
        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            if not isinstance(st, dict):
                continue
            summary_content += f"[{side}]\n"
            summary_content += f"Method: {st.get('offset_method', 'n/a')}\n"
            win = st.get("smart_scan_range", None)
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                try:
                    summary_content += f"Window: {float(win[0]):.0f}-{float(win[1]):.0f} Hz\n"
                except Exception:
                    pass
            try:
                summary_content += f"Offset to measurement: {float(st.get('offset_db', 0.0) or 0.0):+.2f} dB\n"
            except Exception:
                pass
            try:
                summary_content += f"Effective target level: {float(st.get('eff_target_db', 0.0) or 0.0):.2f} dB\n"
            except Exception:
                pass
            tilt = st.get("tilt_slope_db_per_oct", None)
            if tilt is not None:
                try:
                    tilt_f = float(tilt)
                    summary_content += f"Tilt slope: {tilt_f:+.2f} dB/oct\n"
                    if abs(tilt_f) > 1.5:
                        summary_content += (
                            "Warning: Large broadband tilt detected. "
                            "May indicate measurement/target mismatch or strong room tilt.\n"
                        )
                except Exception:
                    pass
            summary_content += "\n"
    except Exception:
        pass

    summary_content = _append_acoustic_events(summary_content, l_st, r_st)



    if 'auto_align' in l_st:
        res = l_st['auto_align']
        summary_content += "\n=== AUTO-ALIGN ===\n"
        summary_content += f"Delay: {res['delay_ms']} ms\n"
        summary_content += f"Distance Diff: {res['distance_cm']} cm\n"
        summary_content += f"Gain Diff: {res['gain_diff_db']} dB\n"

    if TEST_MODE:
        try:
            diag = _build_diagnostics_dict(data, fs_v, l_st, r_st)
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(_json_safe(diag), indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"
        except Exception as e:
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps({
                "schema_version": 1,
                "error": f"diagnostics_json_failed: {type(e).__name__}: {e}"
            }, indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"



    zf.writestr(sum_name, summary_content)

    if bool(write_dashboards):
        
        psl = data.get("plot_smoothing_level", "Psychoacoustic")

        html_l = plots.generate_prediction_plot(
            f_l,
            m_l,
            p_l, l_imp, fs_v, "Left",
            None, l_st, data['mixed_freq'], "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["fs_hz"] = int(fs_v)
            ui_dashboards["left_html"] = str(html_l)
        png_l = plots.generate_combined_plot_mpl(
            f_l, m_l, p_l, l_imp, fs_v, "Left", target_stats=l_st
        )
        if png_l:
            zf.writestr(l_dash_name, png_l)
        else:
            zf.writestr(l_dash_name.replace(".png", ".txt"), str(html_l))

        html_r = plots.generate_prediction_plot(
            f_r,
            m_r,
            p_r, r_imp, fs_v, "Right",
            None, r_st, data['mixed_freq'], "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["right_html"] = str(html_r)
        png_r = plots.generate_combined_plot_mpl(
            f_r, m_r, p_r, r_imp, fs_v, "Right", target_stats=r_st
        )
        if png_r:
            zf.writestr(r_dash_name, png_r)
        else:
            zf.writestr(r_dash_name.replace(".png", ".txt"), str(html_r))

    hlc_cfg = generate_hlc_config(fs_v, ft_short, file_ts, irw_tag=irw_tag)
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg", hlc_cfg)

    if not bool(data.get("multi_rate_opt", False)):
        yaml_content = generate_raspberry_yaml(
            fs_v,
            ft_short,
            file_ts,
            master_gain_db=0.0,
            irw_tag=irw_tag,
        )
        zf.writestr(f"camilladsp_{ft_short}_{fs_v}Hz_{irw_tag}.yml", yaml_content)

