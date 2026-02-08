import os
import sys

# Allow running as a script or frozen entrypoint by forcing package context.
if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "camillafir"

import io
import json
import logging
import re
import zipfile
import typing
import scipy.io.wavfile
import math
from datetime import datetime
from textwrap import dedent
import math
import numpy as np
from pywebio import config, start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.session import set_env
import pywebio.output as pwo
#from pywebio.output import toast
from pywebio.output import put_html
from .config.camillafir_config import load_config, save_config
from .resources.i8n.camillafir_i18n import t
from .ui.camillafir_housecurve import _normalize_hc_mode_key, get_house_curve_by_name, load_target_curve, load_house_curve
from camillafir.io.measurements_loader import load_measurements_lr
from camillafir.io.measurements_txt import parse_measurements_from_path
from .ui.camillafir_ui_helpers import (
    update_mode_desc,
    apply_mode_defaults_to_ui,
    update_taps_auto_info,
    update_lvl_ui,
    apply_tdc_preset,
    apply_afdw_preset,
    put_guide_section,
    _max_boost_help_with_cap,
    _toast,
    _warn_taps_if_over_cap,
    _warn_max_boost_if_over_cap,
)
from .config.camillafir_pipeline import (
    collect_ui_data,
    log_df_smoothing_toggle,
    build_xos_hpf,
    filter_type_short,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
    build_filter_config,
)
from .ui.camillafir_export import _write_fs_outputs
from .dsp import camillafir_dsp as dsp
from .ui import camillafir_plot as plots
import camillafir.config.models as models
from camillafir.config.models import FilterConfig
from .ui.camillafir_modes import apply_mode_to_cfg, MODE_DEFAULTS
from .config.camillafir_convolver_configs import (
    generate_raspberry_yaml,
    generate_hlc_config,
)
from .ui.camillafir_ui import _render_results
from .ui.camillafir_utils import scale_taps_with_fs
from pywebio.pin import pin_update

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")




VERSION = "v2.9.5"
# Change log:
# v.2.9.5 [UI] add low-bass cut toggle + lock Hz field when disabled.
# v.2.9.4 [CFG] Fixed low_bass_cut_hz value not saving correctly in config.
# v.2.9.3 [UI] Fixed typo at psychoacoustic plot smoothing code (1/48 / 1/3 ---> 1/6 / 1/3)
# v.2.9.2 [UI] hard-lock IR windowing to Auto in Basic & Asymmetric filter modes
# v.2.9.1 [UI] Cleared functions (mixed phase & ir-windowing)
# v.2.9.0 [UI] Removed "Symmetric" & "Off" modes from windowing options, leaving only "Auto" (rewind-based) and "Asymmetric" (rewind-based + shift). 
# This simplifies the UI and focuses on the most effective windowing strategies. The "Auto" mode will automatically choose the best windowing based on the impulse response characteristics, while "Asymmetric" can be used to further reduce latency if desired.
# v.2.9.0 [DSP] Fixed HPF magnitude application to ensure it is applied as a real magnitude filter (gain_db += hpf_db) rather than being baked into the target curve. This keeps magnitude and phase consistent and avoids issues with double-HPF effects.
# v.2.8.9 [DSP] Added more safety checks and fixed edge cases in various blocks (leveling, TDC, bass-first, etc.)
# v.2.8.5 [DSP] Added tukey windowing option
# v2.8.4  Github actions now makes running files
# v2.8.2.3 [IO] Fixed ZIP output when multi-rate is enabled:
#               generate a single CamillaDSP .yml using $samplerate$
# v2.8.2.3 [DSP] More precise leveling tilt used in magnitude calculation
# v.2.8.2.2 [UI] updated translations and phase plot
# v.2.8.2.1 changed file structure to more debug-friendly format
# v2.8.2: [UI] improved robustness of file upload parsing from browser & added xo_help translation
# v2.8.1.2" [UI/DSP] bug fix for modes selection, that was not saving ui state correctly
# v2.8.1.1" [UI] small ui-update for modes selection
# v2.8.1: [DSP] fix A-FDW bandwidth limits & UI display
# v2.8.0: [UI] removed html dashboard export (now PNG only)
# v2.7.9: [UI] fix custom house curve upload
# v2.7.8: [IO] fix WAV parsing – phase unwrap
# v2.7.7: [DSP] fix HF phase handling
# v2.7.6: [IO] fix WAV parsing smoothing

PROGRAM_NAME = "CamillaFIR"
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000
MAX_SAFE_TAPS = 131072
TEST_MODE = 1
# =========================
# Test / diagnostics output
# =========================
TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

def _irwin_tag(mode: typing.Any) -> str:
    """
    Short, filename-safe tag for IR export windowing mode.
    UI values: auto / off / rew_sym / rew_asym
    """
    try:
        m = str(mode or "auto").strip().lower()
    except Exception:
        m = "auto"
    if m == "rew_sym":
        return "sym"
    if m == "rew_asym":
        return "asym"
    if m in ("auto", "off"):
        return m
    return "auto"

def process_run():

    from .ui.camillafir_ui import update_status as status_cb

    def _status(msg):
        if callable(status_cb):
            try:
                status_cb(msg)
            except Exception:
                pass
    # 1) UI -> data dict (new unified collector)
    data = collect_ui_data(pin)
    # Sanitize IR export window mode (config.json may contain null)
    _iw = data.get('ir_export_window_mode')
    if not isinstance(_iw, str) or _iw.strip() == "":
        data['ir_export_window_mode'] = 'auto'
    logger.info(f"UI ir_export_window_mode={data.get('ir_export_window_mode')}")

    # Sanitize IR export window shape + Tukey alpha
    try:
        sh = str(data.get('ir_export_window_shape', 'hann') or 'hann').strip().lower()
    except Exception:
        sh = 'hann'
    if sh not in ('hann', 'tukey'):
        sh = 'hann'
    data['ir_export_window_shape'] = sh
    try:
        a = float(data.get('ir_export_tukey_alpha', 0.25))
    except Exception:
        a = 0.25
    if not math.isfinite(a):
        a = 0.25
    data['ir_export_tukey_alpha'] = float(np.clip(a, 0.0, 1.0))

    taps_base = int(float(data.get("taps", 65536) or 65536))
    save_config(data)
    put_processbar('bar')
    put_scope('status_area')
    set_processbar('bar', 0.0)


    # Always warn at START if user requested boost above safety cap
    try:
        mb = float(data.get('max_boost', 0.0) or 0.0)
        if float(MAX_SAFE_BOOST) > 0.0 and mb > float(MAX_SAFE_BOOST) + 1e-9:
            try:
                cap_suffix = t('max_boost_help_cap').format(value=f"{MAX_SAFE_BOOST:.1f}")
            except Exception:
                cap_suffix = f" (capped to {MAX_SAFE_BOOST:.1f} dB)"
            toast(f"{t('max_boost')}: {mb:.1f} dB > {MAX_SAFE_BOOST:.1f} dB{cap_suffix}", duration=6)
    except Exception:
        pass

    # 2) Measurements (upload OR local paths)
    _status(t('stat_reading'))
    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    if f_l is None or f_r is None:
        _toast("Measurement files missing! Load Left/Right or give local.", duration=6, color='red')
        return

    # 3) Target / house curve
    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path
    )
    data['hc_source'] = hc_source
    logger.info(f"House curve source: {hc_source}")
    # 4) XO + HPF
    xos, hpf = build_xos_hpf(data)
    try:
        if xos:
            xo_txt = ", ".join([f"{float(x.get('freq')):.1f}Hz/{int(x.get('slope', int(x.get('order',1))*6))}dB/oct" for x in xos])
            logger.info(f"XO (UI->CFG): {xo_txt}")
        else:
            logger.info("XO (UI->CFG): off")
        if isinstance(hpf, dict) and hpf.get("enabled"):
            hf = float(hpf.get("freq", 0.0) or 0.0)
            ho = int(hpf.get("order", 0) or 0)
            logger.info(f"HPF (UI->CFG): {hf:.1f}Hz/{int(ho*6)}dB/oct")
        else:
            logger.info("HPF (UI->CFG): off")
    except Exception:
        pass
    # 5) (Optional) DF smoothing log
    df_on = log_df_smoothing_toggle(pin, logger)


    # 6) Sample rates list
    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(target_rates, multi_rate_on=multi_rate_on, forced_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ))

    zip_buffer = io.BytesIO()
    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = filter_type_short(data['filter_type'])
    split, zoom = data['mixed_freq'], t('zoom_hint')
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None
    # Debug: UI-selected IR export window parameters (cfg not built yet)
    logger.warning(
        f"EXPORT IR (UI): shape={data.get('ir_export_window_shape')}, "
        f"alpha={data.get('ir_export_tukey_alpha')}"
    )

    # --- IR windowing tag (used in filenames) ---
    # Keep this stable across all outputs in this run.
    val_raw = data.get('ir_export_window_mode', None)
    if not isinstance(val_raw, str) or val_raw.strip() == '':
        val_raw = data.get('ir_window_mode', 'auto')
    irw_mode = str(val_raw or 'auto').strip().lower()
    if irw_mode not in ('auto','off','rew_sym','rew_asym'):
        irw_mode = 'auto'
    data['ir_export_window_mode'] = irw_mode
    irw_tag = _irwin_tag(irw_mode)

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, fs_v in enumerate(target_rates):
            if bool(data.get('multi_rate_opt', False)):
                taps_v = scale_taps_with_fs(fs_v, taps_base)
                logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
            else:
                taps_v = taps_base
            _status(f"Lasketaan {fs_v}Hz...")
            set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))
            data["enable_residual_pass"] = True

            cfg = build_filter_config(
                FilterConfig_cls=FilterConfig,
                fs_v=fs_v,
                taps_v = taps_v,
                data=data,
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin=pin,
            )

            logger.info(
                f"[{fs_v} Hz] EXPORT IR cfg: shape={cfg.ir_export_window_shape}, "
                f"alpha={cfg.ir_export_tukey_alpha}"
            )
            setattr(cfg, 'ir_export_window_mode', irw_mode)

            # IR export window shape (Hann/Tukey) + alpha (0..1)
            try:
                setattr(cfg, 'ir_export_window_shape', str(data.get('ir_export_window_shape', 'hann') or 'hann').strip().lower())
                setattr(cfg, 'ir_export_tukey_alpha', float(data.get('ir_export_tukey_alpha', 0.25) or 0.25))
            except Exception:
                pass

           # IR window length values come from UI keys (ms). DSP expects *_ms fields.
            try:
                setattr(cfg, 'ir_window', float(data.get('ir_window', getattr(cfg, 'ir_window', 0.0)) or 0.0))
                setattr(cfg, 'ir_window_left', float(data.get('ir_window_left', getattr(cfg, 'ir_window_left', 0.0)) or 0.0))
            except Exception:
                pass

            # --- Safety cap for boost (CamillaFIR philosophy: never allow "surprise" boosts) ---
            # max_boost_db is a user-visible knob, but we additionally cap it with MAX_SAFE_BOOST
            # to prevent accidental large boosts from unstable measurements / target mismatch.
            try:
                _user_mb = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
                setattr(cfg, "max_boost_db_user", _user_mb)
                setattr(cfg, "max_safe_boost_db", float(MAX_SAFE_BOOST))
                if _user_mb > 0.0 and float(MAX_SAFE_BOOST) > 0.0:
                    _eff_mb = min(_user_mb, float(MAX_SAFE_BOOST))
                    if _eff_mb < _user_mb - 1e-9:
                        logger.info(
                            f"Safety cap: max_boost_db user={_user_mb:.2f} dB -> effective={_eff_mb:.2f} dB "
                            f"(MAX_SAFE_BOOST={float(MAX_SAFE_BOOST):.2f} dB)"
                        )
                    setattr(cfg, "max_boost_db", float(_eff_mb))
            except Exception:
                pass



            is_wav = detect_is_wav_source(data, pin)
            try:
                setattr(cfg, "is_wav_source", bool(is_wav))
            except Exception:
                pass



            log_df_smoothing_toggle(cfg, df_on)
            _status(f"{t('stat_calc')} {int(fs_v)} Hz")
            if bool(getattr(cfg, "stereo_link", False)):
                l_imp, l_st, r_imp, r_st = dsp.generate_filter_pair(
                    f_l, m_l, p_l,
                    f_r, m_r, p_r,
                    cfg
                )
            else:
                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

            # ------------------------------------------------------------------
            # RT60 reliability tagging for scoring:
            # - WAV/IR path: RT60 is IR-derived => higher reliability
            # - TXT/REW FR path: RT60 is proxy/estimate => lower reliability
            # This is used by the new Acoustic Score formula (bonus is weighted).
            # ------------------------------------------------------------------
            try:
                rt_rel = 1.0 if bool(is_wav) else 0.25
                rt_src = "WAV" if bool(is_wav) else "TXT/REW"
                if isinstance(l_st, dict):
                    l_st["rt60_reliability"] = float(rt_rel)
                    l_st["rt60_source"] = rt_src
                if isinstance(r_st, dict):
                    r_st["rt60_reliability"] = float(rt_rel)
                    r_st["rt60_source"] = rt_src
            except Exception:
                pass


            l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
            r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)
            # Build comparison grid per sample-rate (needed for correct UI scoring with WAV)
            if bool(data.get("comparison_mode", False)):
                try:
                    l_st = plots._make_comparison_stats(l_st, int(fs_v), int(taps_v))
                    r_st = plots._make_comparison_stats(r_st, int(fs_v), int(taps_v))
                except Exception as e:
                    logger.warning(f"Comparison-mode stats failed: {e}")

            # ------------------------------------------------------------------
            # Time alignment
            #
            # TXT-compatible behavior: if generate_filter() produced explicit
            # delay estimates (delay_samples), prefer those for alignment.
            # This avoids the "peak-pick" method drifting when the main impulse
            # peak is not stable (common with heavy LF energy / long tails).
            #
            # If delay_samples are missing, fall back to the legacy peak-pick.
            # ------------------------------------------------------------------
            if data['align_opt']:
                d_s = None

                # Prefer delay_samples (TXT-compatible)
                try:
                    dl = l_st.get('delay_samples', None) if isinstance(l_st, dict) else None
                    dr = r_st.get('delay_samples', None) if isinstance(r_st, dict) else None
                    if dl is not None and dr is not None:
                        dl_i = int(round(float(dl)))
                        dr_i = int(round(float(dr)))
                        d_s = dl_i - dr_i
                except Exception:
                    d_s = None

                # Fallback: align by impulse peak
                if d_s is None:
                    d_s = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))

                if d_s > 0:
                    r_imp = np.roll(r_imp, d_s)
                elif d_s < 0:
                    l_imp = np.roll(l_imp, -d_s)

            # UI "results" view: show the same fs as the (single) dashboard fs in multi-rate.
            if fs_v == dash_fs:
                l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp

            

            if 'delay_samples' in l_st and 'delay_samples' in r_st:
                diff_samples = r_st['delay_samples'] - l_st['delay_samples']
                delay_ms = round((diff_samples / fs_v) * 1000, 3)
                distance_cm = round((delay_ms / 1000) * 34300, 2)
                gain_diff = round(l_st['offset_db'] - r_st['offset_db'], 2)
                l_st['auto_align'] = {'delay_ms': delay_ms, 'distance_cm': distance_cm, 'gain_diff_db': gain_diff}

            wav_l, wav_r = io.BytesIO(), io.BytesIO()
            scipy.io.wavfile.write(wav_l, fs_v, l_imp.astype(np.float32))
            scipy.io.wavfile.write(wav_r, fs_v, r_imp.astype(np.float32))
            zf.writestr(f"L_{ft_short}_{fs_v}Hz_{file_ts}_{irw_tag}.wav", wav_l.getvalue())
            zf.writestr(f"R_{ft_short}_{fs_v}Hz_{file_ts}_{irw_tag}.wav", wav_r.getvalue())

            _write_fs_outputs(
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
                write_dashboards=(not multi_rate_on) or (int(fs_v) == int(dash_fs)),
                irw_tag=irw_tag,
            )

        # Multi-rate: write ONE CamillaDSP YAML (uses $samplerate$ in FIR filenames)
        if bool(data.get("multi_rate_opt", False)):
            yaml_content = generate_raspberry_yaml(
                int(data.get("fs") or 44100),
                ft_short,
                file_ts,
                master_gain_db=float(data.get('gain', 0.0) or 0.0),
                irw_tag=irw_tag,
            )
            zf.writestr(f"camilladsp_{ft_short}_{irw_tag}.yml", yaml_content)

    # --- Save ZIP into filters/ directory ---
    filters_dir = os.path.join(os.getcwd(), "filters")
    os.makedirs(filters_dir, exist_ok=True)

    fname = f"CamillaFIR_{ft_short}_{irw_tag}_{ts}.zip"
    out_path = os.path.join(filters_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Saved: {os.path.abspath(out_path)}"
    except Exception:
        save_msg = "Zip saving failed."


    # --- Ensure UI has stats even if fs selection didn't hit (e.g. WAV/local path quirks) ---
    if l_st_f is None:
        l_st_f = l_st
    if r_st_f is None:
        r_st_f = r_st
    if l_imp_f is None:
        l_imp_f = l_imp
    if r_imp_f is None:
        r_imp_f = r_imp

    # --- Ensure UI scoring has filter_mags (so Measured != Filtered) ---
    try:
        fs_sel = int(data.get('fs') or 44100)
    except Exception:
        fs_sel = 44100
    _inject_filter_mags_for_ui(l_st_f, l_imp_f, fs_sel)
    _inject_filter_mags_for_ui(r_st_f, r_imp_f, fs_sel)

    logger.info(f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
                f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis',[]))}/{len(l_st_f.get('cmp_measured_mags',[]))}/{len(l_st_f.get('cmp_target_mags',[]))}")

    _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer)

def _ui_pick(stats, key):
    """
    UI helper: pick comparison-grid data if analysis_mode == 'comparison'
    """
    if not stats:
        return None
    mode = str(stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _pick_cmp(stats, key):
    """
    Return comparison-mode arrays for UI scoring if available.
    """
    if not stats:
        return None
    if str(stats.get("analysis_mode", "native")).lower() == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)

def view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
    """UI-only smoothing for plots (does NOT affect DSP math).

    plot_smoothing_level:
      - "Psychoacoustic" => REW-like crossfade (heavy LF, light HF), view-only
      - int N            => standard 1/N octave smoothing (view-only)
    """
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    if f.size < 8 or m.size != f.size:
        return mags

    try:
        apply_smoothing_std = dsp.apply_smoothing_std
    except Exception:
        return mags

    psl = plot_smoothing_level

    # Psychoacoustic (string selector)
    if isinstance(psl, str) and ("psy" in psl.lower()):
        try:
            dummy = np.zeros_like(m)
            m_heavy, _ = apply_smoothing_std(f, m, dummy, 1/3.0)
            m_light, _ = apply_smoothing_std(f, m, dummy, 1/48.0)

            ff = np.maximum(f, 1.0)
            lo, hi = 200.0, 2000.0
            w = (np.log10(ff) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            w = np.clip(w, 0.0, 1.0)
            return (1.0 - w) * m_heavy + w * m_light
        except Exception:
            return m

    # Standard (numeric selector => 1/N octave)
    try:
        n = float(psl if not isinstance(psl, str) else 48.0)
        if not np.isfinite(n) or n <= 0:
            n = 48.0
        oct_frac = 1.0 / n
    except Exception:
        oct_frac = 1.0 / 48.0

    try:
        m_sm, _ = apply_smoothing_std(f, m, np.zeros_like(m), float(oct_frac))
        return m_sm
    except Exception:
        return m


def _ensure_scoring_keys(st, f_in, m_in, hc_f, hc_m):
    """
    Ensure UI scoring keys exist in stats dict (WAV/TXT safe).
    - freq_axis, measured_mags, target_mags, confidence_mask
    """
    try:
        if st is None:
            return st

        f = np.asarray(f_in or [], dtype=float)
        m = np.asarray(m_in or [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        # target mags (fallback from house curve if missing)
        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f or [], dtype=float)
                hm = np.asarray(hc_m or [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

        # confidence mask (fallback to ones if missing)
        if st.get("confidence_mask") is None:
            if f.size > 1:
                st["confidence_mask"] = np.ones_like(f, dtype=float)
        return st
    except Exception:
        return st

_HOUSE_FREQS = np.array([
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
], dtype=float)

def _resample_to_freq_axis(freqs_dst: np.ndarray, arr: np.ndarray, freqs_src: np.ndarray) -> np.ndarray:
    """Safe 1D interpolation in log-frequency domain."""
    if arr.size == 0 or freqs_src.size == 0 or freqs_dst.size == 0:
        return arr
    # clip to valid region
    f1 = np.maximum(freqs_src.astype(float), 1.0)
    f2 = np.maximum(freqs_dst.astype(float), 1.0)
    lf1 = np.log10(f1)
    lf2 = np.log10(f2)
    # Ensure monotonic source
    order = np.argsort(lf1)
    lf1 = lf1[order]
    a1 = arr.astype(float)[order]
    return np.interp(lf2, lf1, a1, left=a1[0], right=a1[-1])


def calculate_target_match(st):
    """Calculates how well the corrected response follows the target curve (0-100%)."""
    if not st:
        return 0.0

    freqs = np.asarray(_ui_pick(st, 'freq_axis') or [], dtype=float)
    meas  = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt  = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if freqs.size == 0 or meas.size == 0 or target.size == 0:
        return 0.0

    # WAV-polulla filter_mags voi puuttua -> tulkitaan 0 dB korjaukseksi
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)
    # If filter mags are missing (common in some UI paths), treat as 0 dB correction
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)

    # If WAV measurement: measured/target are dense FFT grid, but filter may be on 19-point house grid.
    # Resample target/filter to the measurement freq_axis when shapes differ.
    if target.size != freqs.size:
        # common case: target on house grid
        if target.size == _HOUSE_FREQS.size:
            target = _resample_to_freq_axis(freqs, target, _HOUSE_FREQS)
        else:
            # last resort: truncate
            n = min(freqs.size, meas.size, target.size)
            freqs, meas, target = freqs[:n], meas[:n], target[:n]

    if filt.size != freqs.size:
        if filt.size == _HOUSE_FREQS.size:
            filt = _resample_to_freq_axis(freqs, filt, _HOUSE_FREQS)
        else:
            n = min(freqs.size, meas.size, filt.size, target.size)
            freqs, meas, target, filt = freqs[:n], meas[:n], target[:n], filt[:n]

    # RMS virhe (dB) korjatusta vasteesta
    diff = (meas + filt) - target
    rms = float(np.sqrt(np.mean(diff * diff)))

    # Sama muunnos kuin Summaryssä (sigmoidi)
    m0 = 3.2   # dB @ 50%
    s0 = 0.9   # jyrkkyys
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))



def _avg_confidence_pct(st: dict) -> float:
    """
    UI helper: returns average confidence in percent.
    Supports comparison-mode keys (cmp_avg_confidence / cmp_confidence_mask).
    """
    if not st:
        return 0.0
    mode = str(st.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        v = st.get("cmp_avg_confidence", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        cm = np.asarray(st.get("cmp_confidence_mask", []) or [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    # native
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm = np.asarray(st.get("confidence_mask", []) or [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


def calculate_target_match_unfiltered(st: dict) -> float:
    """
    Target match for *unfiltered* response (measured vs target).
    Uses the same sigmoid mapping as calculate_target_match().
    """
    if not st:
        return 0.0
    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    if meas.size == 0 or target.size == 0:
        return 0.0
    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]
    diff = meas - target
    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))

def _inject_filter_mags_for_ui(st: dict, filt_ir, fs: int):
    """Ensure st has filter_mags on the same freq_axis as measured, so UI can score 'Filtered' correctly.

    Some pipelines didn't store filter_mags into stats; then UI 'Measured' and 'Filtered' collapse to the same value.
    This computes |FFT(filter_ir)| and interpolates it to st['freq_axis'] (or cmp_freq_axis if in comparison mode),
    storing it as (cmp_)filter_mags in dB.
    """
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        if st.get(key_g) is not None:
            return

        f_axis = np.asarray(st.get(key_f, []) or [], dtype=float)
        if f_axis.size < 4:
            return

        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 8:
            return

        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return

        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

        f_min = float(np.min(f_fft))
        f_max = float(np.max(f_fft))
        f_q = np.clip(f_axis, f_min, f_max)
        st[key_g] = np.interp(f_q, f_fft, g_db).tolist()
    except Exception:
        return


def calculate_score(st, is_predicted=False):
    """UI score (0..99) for Measured / Filtered.

    Note: this is *not* the same as Target Curve Match.
    It combines:
      - target match (sigmoid RMS mapping)
      - acoustic confidence
      - optional RT60 room-quality bonus/penalty (scaled by reliability)
    """
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if meas.size == 0 or target.size == 0:
        return float(np.clip(conf, 0.0, 99.0))

    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]

    if is_predicted:
        if filt.size >= n:
            filt = filt[:n]
        elif filt.size > 0:
            filt = np.pad(filt, (0, n - filt.size), mode='edge')
        else:
            filt = np.zeros(n, dtype=float)
        diff = (meas + filt) - target
    else:
        diff = meas - target

    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    match_pct = float(np.clip(match_pct, 0.0, 100.0))

    base = 0.55 * match_pct + 0.35 * conf  # 0..90

    rt_bonus = 0.0
    try:
        rt = float(st.get('rt60_val', None)) if st.get('rt60_val', None) is not None else None
    except Exception:
        rt = None
    try:
        rel = float(st.get('rt60_reliability', 0.0) or 0.0)
    except Exception:
        rel = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))

    if rt is not None and rt > 0:
        if rt <= 0.35:
            rt_bonus = ((0.35 - rt) / 0.25) * 15.0
        elif rt >= 0.55:
            rt_bonus = -min(15.0, ((rt - 0.55) / 0.35) * 15.0)
        rt_bonus *= rel

    events = st.get('cmp_reflections', st.get('reflections', [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(8.0, float(len(events)) * 1.0) * penalty_mult

    score = base + rt_bonus - event_penalty
    return float(np.clip(score, 0.0, 99.0))


# Build PyWebIO app from extracted UI module
from .ui.camillafir_ui import build_app as _build_ui_app
main = _build_ui_app(
    process_run=process_run,
    PROGRAM_NAME=PROGRAM_NAME,
    VERSION=VERSION,
    MAX_SAFE_BOOST=MAX_SAFE_BOOST,
)


if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
