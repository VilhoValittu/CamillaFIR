import os
import sys

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "camillafir"

import io
import logging
import zipfile
import typing
import scipy.io.wavfile
import math
import time
from datetime import datetime
import numpy as np
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from .config.camillafir_config import save_config
from .resources.i8n.camillafir_i18n import t
from .ui.camillafir_housecurve import load_house_curve
from camillafir.io.measurements_loader import load_measurements_lr
from camillafir.io.measurements_txt import parse_measurements_from_path
from .ui.system_health import (
    compute_health,
    toast_health_gate_result,
    toast_measurement_files_missing,
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
from .dsp.target_match import target_match_from_stats
from .ui import camillafir_plot as plots
from camillafir.config.models import FilterConfig
from .ui.camillafir_modes import apply_mode_to_cfg
from .config.camillafir_convolver_configs import (
    generate_raspberry_yaml,
)
from .ui.camillafir_ui import _render_results
from .ui.camillafir_utils import scale_taps_with_fs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

VERSION = "v.3.1.1.2"
PROGRAM_NAME = "CamillaFIR"
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000
MAX_SAFE_TAPS = 131072
TEST_MODE = 0
TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

def resolve_static_dir() -> str | None:
    """
    Palauttaa PyWebIO:n static_dir-polun, josta paikallinen Plotly JS loytyy.

    Tarkistaa ensin PyInstaller-ympariston (`sys._MEIPASS/assets`) ja sen
    jalkeen lahdekoodin resurssipolun (`resources/plotly`).
    """
    candidates = []

    if hasattr(sys, "_MEIPASS"):
        try:
            candidates.append(os.path.join(sys._MEIPASS, "assets"))  # type: ignore[attr-defined]
        except Exception:
            pass

    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "plotly"))

    for d in candidates:
        try:
            if d and os.path.isfile(os.path.join(d, "plotly.min.js")):
                return d
        except Exception:
            continue
    return None

def _irwin_tag(mode: typing.Any) -> str:
    """
    Normalisoi IR-ikkunointitilan lyhyeksi tiedostonimitunnisteeksi.

    Tunnetut arvot muunnetaan muotoon `sym`, `asym`, `auto` tai `off`.
    Virheellinen syote palautuu arvoon `auto`.
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

def _resolve_ui_stats_fs(ui_stats_fs: typing.Any, selected_fs: typing.Any) -> int:
    """
    Valitsee UI-statistiikalle oikean sample raten.

    Priorisoi dashboardille valitun analyysinopeuden (`ui_stats_fs`) ja
    kaatuu turvallisesti UI-valintaan (`selected_fs`) tai 44100 Hz:iin.
    """
    for cand in (ui_stats_fs, selected_fs, 44100):
        try:
            fs_i = int(cand)
        except Exception:
            continue
        if fs_i > 0:
            return fs_i
    return 44100

def _shift_zeropad_1d(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Siirtaa 1D-signaalia kokonaisilla naytteilla nollataytolla.

    Positiivinen siirto lisaa viivetta (siirto oikealle), negatiivinen
    aikaistaa signaalia (siirto vasemmalle) ilman wrap-around-kaytosta.
    """
    arr = np.asarray(x)
    n = int(arr.size)
    s = int(shift)
    if n == 0 or s == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if s > 0:
        if s < n:
            out[s:] = arr[:-s]
    else:
        s = -s
        if s < n:
            out[:-s] = arr[s:]
    return out


def _postpolish_wav_filter_ir(
    ir: np.ndarray,
    fs: int,
    *,
    mag_c_min: float,
    mag_c_max: float,
    trans_width: float,
) -> np.ndarray:
    """
    Viimeistelee WAV-lahteesta lasketun FIR-impulssin ennen vientia.

    Tasaa korjausalueen ylarajan ymparilla esiintyvaa ripplea taajuustasossa,
    sailyttaa vaiheen ja skaalaa ulostulon takaisin alkuperaiseen huippuun.
    """
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    fs_i = int(fs) if fs else 0
    if n < 64 or fs_i <= 0:
        return x

    cmin = float(mag_c_min if np.isfinite(mag_c_min) else 0.0)
    cmax = float(mag_c_max if np.isfinite(mag_c_max) else 0.0)
    tw = float(trans_width if np.isfinite(trans_width) else 0.0)
    if cmax <= max(1.0, cmin):
        return x
    if tw <= 0.0:
        tw = max(50.0, 0.4 * cmax)

    f_lo = max(cmin, cmax - 0.95 * tw)
    f_hi = min(float(fs_i) * 0.5, cmax + 1.45 * tw)
    if f_hi <= f_lo:
        return x

    h = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_i))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-14))
    ph = np.angle(h)

    df = float(np.median(np.diff(freqs))) if freqs.size > 2 else 1.0
    sigma_bins = max(2.0, 8.0 / max(df, 1e-9))
    half = int(max(3, round(4.0 * sigma_bins)))
    kx = np.arange(-half, half + 1, dtype=float)
    kk = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kk /= np.sum(kk)
    mag_sm = np.convolve(mag_db, kk, mode="same")

    zone = (freqs >= f_lo) & (freqs <= f_hi)
    if int(np.count_nonzero(zone)) < 8:
        return x

    w = np.zeros_like(freqs, dtype=float)
    span = max(1e-9, float(f_hi - f_lo))
    zz = (freqs[zone] - f_lo) / span
    w[zone] = 0.5 - 0.5 * np.cos(np.pi * np.clip(zz, 0.0, 1.0))

    mix = 0.95
    mag_out = mag_db + (mag_sm - mag_db) * (mix * w)

    h2 = np.power(10.0, mag_out / 20.0) * np.exp(1j * ph)
    y = np.fft.irfft(h2, n=n)

    p0 = float(np.max(np.abs(x)))
    p1 = float(np.max(np.abs(y)))
    if p0 > 0.0 and p1 > 0.0:
        y = y * (p0 / p1)
    return y.astype(float, copy=False)


def process_run():
    from .ui.camillafir_ui import update_status as status_cb

    run_started_at = time.perf_counter()
    perf_stats = {
        "read_s": 0.0,
        "dsp_s": 0.0,
        "zip_png_s": 0.0,
    }
    per_fs_stats = {}

    def _elapsed_seconds() -> float:
        try:
            return max(0.0, float(time.perf_counter() - run_started_at))
        except Exception:
            return 0.0

    def _status(msg):
        if callable(status_cb):
            try:
                status_cb(f"{msg} | {_elapsed_seconds():.1f} s")
            except Exception:
                pass

    data = collect_ui_data(pin)
    data["program_version"] = VERSION

    try:
        mode = str(data.get("mode") or "BASIC").strip().upper()
        hr = compute_health(data, mode)
        if toast_health_gate_result(hr, mode):
            return
    except Exception:
        pass



    _iw = data.get('ir_export_window_mode')
    if not isinstance(_iw, str) or _iw.strip() == "":
        data['ir_export_window_mode'] = 'auto'
    logger.info(f"UI ir_export_window_mode={data.get('ir_export_window_mode')}")

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
    _t_read = time.perf_counter()
    _status(t('stat_reading'))
    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    perf_stats["read_s"] += max(0.0, float(time.perf_counter() - _t_read))
    if f_l is None or f_r is None:
        toast_measurement_files_missing()
        return

    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path
    )
    data['hc_source'] = hc_source
    logger.info(f"House curve source: {hc_source}")
    try:
        if hc_f is not None and hc_m is not None:
            logger.info(
                f"HC: n={len(hc_f)} f=[{hc_f[0]:.2f}..{hc_f[-1]:.2f}] "
                f"m=[{float(np.min(hc_m)):.2f}..{float(np.max(hc_m)):.2f}] mean={float(np.mean(hc_m)):.2f}"
            )
    except Exception:
        pass
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
    df_on = log_df_smoothing_toggle(pin, logger)


    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(target_rates, multi_rate_on=multi_rate_on, forced_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ))
    zip_dashboards_on = False

    zip_buffer = io.BytesIO()
    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = filter_type_short(data['filter_type'])
    split, zoom = data['mixed_freq'], t('zoom_hint')
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None
    ui_stats_fs = None
    ui_dashboards = {}
    logger.warning(
        f"EXPORT IR (UI): shape={data.get('ir_export_window_shape')}, "
        f"alpha={data.get('ir_export_tukey_alpha')}"
    )

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

            try:
                mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
            except Exception:
                mode_u = "BASIC"
            try:
                apply_mode_to_cfg(cfg, mode_u, apply_defaults=False)
            except Exception as e:
                logger.warning(f"Mode clamp apply failed ({mode_u}): {e}")

            logger.info(
                f"[{fs_v} Hz] EXPORT IR cfg: shape={cfg.ir_export_window_shape}, "
                f"alpha={cfg.ir_export_tukey_alpha}"
            )
            setattr(cfg, 'ir_export_window_mode', irw_mode)

            try:
                setattr(cfg, 'ir_export_window_shape', str(data.get('ir_export_window_shape', 'hann') or 'hann').strip().lower())
                setattr(cfg, 'ir_export_tukey_alpha', float(data.get('ir_export_tukey_alpha', 0.25) or 0.25))
            except Exception:
                pass

            try:
                setattr(cfg, 'ir_window', float(data.get('ir_window', getattr(cfg, 'ir_window', 500.0)) or 500.0))
                setattr(cfg, 'ir_window_left', float(data.get('ir_window_left', getattr(cfg, 'ir_window_left', 120.0)) or 120.0))
            except Exception:
                pass

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
            _t_dsp = time.perf_counter()
            if bool(getattr(cfg, "stereo_link", False)):
                l_imp, l_st, r_imp, r_st = dsp.generate_filter_pair(
                    f_l, m_l, p_l,
                    f_r, m_r, p_r,
                    cfg
                )
            else:
                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)
            _dsp_dt = max(0.0, float(time.perf_counter() - _t_dsp))
            perf_stats["dsp_s"] += _dsp_dt
            _fs_k = int(fs_v)
            _slot = per_fs_stats.setdefault(_fs_k, {})
            _slot["dsp_s"] = float(_slot.get("dsp_s", 0.0)) + _dsp_dt

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
            if bool(data.get("comparison_mode", False)):
                try:
                    l_st = plots._make_comparison_stats(l_st, int(fs_v), int(taps_v))
                    r_st = plots._make_comparison_stats(r_st, int(fs_v), int(taps_v))
                except Exception as e:
                    logger.warning(f"Comparison-mode stats failed: {e}")

            d_s = None
            align_method = "peak"
            d_peak = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))

            d_delay = None
            try:
                dl = l_st.get('delay_samples', None) if isinstance(l_st, dict) else None
                dr = r_st.get('delay_samples', None) if isinstance(r_st, dict) else None
                if dl is not None and dr is not None:
                    dl_i = int(round(float(dl)))
                    dr_i = int(round(float(dr)))
                    d_delay = dr_i - dl_i
            except Exception:
                d_delay = None

            if d_delay is None:
                d_s = d_peak
                align_method = "peak"
            else:
                d_s = int(d_delay)
                align_method = "delay_samples"

                try:
                    guard_samples = 8
                    if abs(int(d_delay) - int(d_peak)) > int(guard_samples):
                        d_s = int(d_peak)
                        align_method = "peak_guard"
                        logger.info(
                            f"Alignment guard: delay_samples={int(d_delay)} vs peak={int(d_peak)} "
                            f"(>{guard_samples} samp) -> using peak"
                        )
                except Exception:
                    pass

            if d_s > 0:
                r_imp = _shift_zeropad_1d(r_imp, d_s)
            elif d_s < 0:
                l_imp = _shift_zeropad_1d(l_imp, -d_s)

            _wav_like_fft_grid = False
            try:
                _fx = np.asarray(f_l if f_l is not None else [], dtype=float)
                if _fx.size > 1024 and abs(float(_fx[0])) < 1e-9:
                    _df = float(np.median(np.diff(_fx[: min(int(_fx.size), 4096)])))
                    if np.isfinite(_df) and (0.0 < _df < 2.0):
                        _wav_like_fft_grid = True
            except Exception:
                _wav_like_fft_grid = False

            if bool(is_wav) or bool(_wav_like_fft_grid):
                try:
                    mc_min = float(getattr(cfg, "mag_c_min", data.get("mag_c_min", 10.0)) or 10.0)
                    mc_max = float(getattr(cfg, "mag_c_max", data.get("mag_c_max", 230.0)) or 230.0)
                    tr_w = float(getattr(cfg, "trans_width", data.get("trans_width", 100.0)) or 100.0)

                    l_imp = _postpolish_wav_filter_ir(
                        l_imp,
                        int(fs_v),
                        mag_c_min=mc_min,
                        mag_c_max=mc_max,
                        trans_width=tr_w,
                    )
                    r_imp = _postpolish_wav_filter_ir(
                        r_imp,
                        int(fs_v),
                        mag_c_min=mc_min,
                        mag_c_max=mc_max,
                        trans_width=tr_w,
                    )
                    logger.info(
                        f"WAV final IR polish applied at {int(fs_v)} Hz "
                        f"(zone approx {max(mc_min, mc_max - 0.95*tr_w):.0f}-{mc_max + 1.45*tr_w:.0f} Hz, "
                        f"is_wav={bool(is_wav)}, wav_like_fft_grid={bool(_wav_like_fft_grid)})"
                    )
                except Exception as e:
                    logger.warning(f"WAV final IR polish failed: {e}")

            if fs_v == dash_fs:
                l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp
                ui_stats_fs = int(fs_v)

            

            if isinstance(l_st, dict) and isinstance(r_st, dict):
                try:
                    delay_ms = round((float(d_s) / fs_v) * 1000, 3)
                    distance_cm = round((delay_ms / 1000) * 34300, 2)
                    gain_diff = round(float(l_st.get('offset_db', 0.0)) - float(r_st.get('offset_db', 0.0)), 2)
                    l_st['auto_align'] = {
                        'delay_ms': delay_ms,
                        'distance_cm': distance_cm,
                        'gain_diff_db': gain_diff,
                        'method': str(align_method),
                    }
                except Exception:
                    pass

            _t_zip = time.perf_counter()
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
                write_dashboards=zip_dashboards_on and ((not multi_rate_on) or (int(fs_v) == int(dash_fs))),
                irw_tag=irw_tag,
                ui_dashboards=ui_dashboards if int(fs_v) == int(dash_fs) else None,
            )
            _zip_dt = max(0.0, float(time.perf_counter() - _t_zip))
            perf_stats["zip_png_s"] += _zip_dt
            _slot["zip_png_s"] = float(_slot.get("zip_png_s", 0.0)) + _zip_dt

        if bool(data.get("multi_rate_opt", False)):
            yaml_content = generate_raspberry_yaml(
                int(data.get("fs") or 44100),
                ft_short,
                file_ts,
                master_gain_db=0.0,
                irw_tag=irw_tag,
            )
            zf.writestr(f"camilladsp_{ft_short}_{irw_tag}.yml", yaml_content)

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


    if l_st_f is None:
        l_st_f = l_st
    if r_st_f is None:
        r_st_f = r_st
    if l_imp_f is None:
        l_imp_f = l_imp
    if r_imp_f is None:
        r_imp_f = r_imp

    try:
        fs_sel = int(data.get('fs') or 44100)
    except Exception:
        fs_sel = 44100
    fs_ui_stats = _resolve_ui_stats_fs(ui_stats_fs, fs_sel)
    _inject_filter_mags_for_ui(l_st_f, l_imp_f, fs_ui_stats)
    _inject_filter_mags_for_ui(r_st_f, r_imp_f, fs_ui_stats)

    logger.info(f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
                f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis',[]))}/{len(l_st_f.get('cmp_measured_mags',[]))}/{len(l_st_f.get('cmp_target_mags',[]))}")

    _render_results(
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
        dash_html_l=ui_dashboards.get("left_html"),
        dash_html_r=ui_dashboards.get("right_html"),
        run_started_at=run_started_at,
        perf_stats=perf_stats,
        per_fs_stats=per_fs_stats,
        saved_filters_dir=os.path.abspath(filters_dir),
    )

def _ui_pick(stats, key):
    """
    Hakee UI:lle oikean arvon natiivi- tai vertailutilasta.

    Jos analyysitila on `comparison`, funktio priorisoi `cmp_<key>`-avaimen.
    Muussa tapauksessa palauttaa tavallisen `<key>`-arvon.
    """
    if not stats:
        return None
    mode = str(stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _pick_cmp(stats, key):
    """Yhteensopiva valitsin vertailutilan (`cmp_`) ja natiivin datan valille."""
    if not stats:
        return None
    if str(stats.get("analysis_mode", "native")).lower() == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)

def view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
    """
    Tuottaa kuvaajia varten tasoitetun amplitudikayran (vain UI-nakymaan).

    Tukee kahta tilaa:
    - `Psychoacoustic`: REW-tyylinen painotettu LF/HF-yhdistelma
    - numeerinen N: standardi 1/N-oktaavitasoitus
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
    Varmistaa, etta pisteytykseen tarvittavat avaimet ovat stats-sanakirjassa.

    Taydentaa puuttuvat kentat (`freq_axis`, `measured_mags`, `target_mags`,
    `confidence_mask`) turvallisilla oletuksilla annetuista syotteista.
    """
    try:
        if st is None:
            return st

        f = np.asarray(f_in if f_in is not None else [], dtype=float)
        m = np.asarray(m_in if m_in is not None else [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f if hc_f is not None else [], dtype=float)
                hm = np.asarray(hc_m if hc_m is not None else [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

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
    """
    Resamplaa 1D-kayran uudelle taajuusakselille log-taajuusinterpoloinnilla.

    Tarkoitettu tilanteisiin, joissa target- tai filter-kayran pisteistus
    poikkeaa mittausakselista.
    """
    if arr.size == 0 or freqs_src.size == 0 or freqs_dst.size == 0:
        return arr
    f1 = np.maximum(freqs_src.astype(float), 1.0)
    f2 = np.maximum(freqs_dst.astype(float), 1.0)
    lf1 = np.log10(f1)
    lf2 = np.log10(f2)
    order = np.argsort(lf1)
    lf1 = lf1[order]
    a1 = arr.astype(float)[order]
    return np.interp(lf2, lf1, a1, left=a1[0], right=a1[-1])


def calculate_target_match(st):
    """
    Laskee korjatun vasteen tavoitevastaavuuden prosentteina (0-100).

    Vertailu tehdaan kayrille `(measured + filter)` vs. `target`, jonka
    RMS-virhe muunnetaan sigmoidilla prosenttipisteiksi.
    """
    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=True,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    return float(match_pct) if match_pct is not None else 0.0



def _avg_confidence_pct(st: dict) -> float:
    """
    Palauttaa keskimaaraisen confidence-arvon prosentteina.

    Tukee seka natiivi- (`avg_confidence` / `confidence_mask`) etta
    vertailutilan (`cmp_avg_confidence` / `cmp_confidence_mask`) avaimia.
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
        cm_src = st.get("cmp_confidence_mask", None)
        cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm_src = st.get("confidence_mask", None)
    cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


def calculate_target_match_unfiltered(st: dict) -> float:
    """
    Laskee tavoitevastaavuuden suodattamattomalle vasteelle.

    Vertailu tehdaan suoraan kayrille `measured` vs. `target` samalla
    sigmoidikartoituksella kuin suodatetussa target-match-laskennassa.
    """
    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    return float(match_pct) if match_pct is not None else 0.0

def _inject_filter_mags_for_ui(st: dict, filt_ir, fs: int):
    """
    Laskee ja injektoi `filter_mags`-kayran stats-rakenteeseen UI:ta varten.

    Jos putki ei ole tallentanut filter-kayraa valmiiksi, funktio muodostaa
    sen `filt_ir`-impulssista FFT:lla ja interpoloi aktiiviselle taajuusakselille.
    """
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        if st.get(key_g) is not None:
            return

        f_src = st.get(key_f, None)
        f_axis = np.asarray(f_src if f_src is not None else [], dtype=float)
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
    """
    Laskee UI:n kokonaispisteet (0..99) mitatulle tai ennustetulle vasteelle.

    Piste koostuu target-matchista, confidence-arvosta, RT60-bonuksesta
    (luotettavuudella painotettuna) seka heijastuslistan penalteista.
    """
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=bool(is_predicted),
        use_confidence=True,
        use_smart_scan_range=True,
    )
    if match_pct is None:
        return float(np.clip(conf, 0.0, 99.0))
    match_pct = float(match_pct)

    base = 0.55 * match_pct + 0.35 * conf

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


from .ui.camillafir_ui import build_app as _build_ui_app
main = _build_ui_app(
    process_run=process_run,
    PROGRAM_NAME=PROGRAM_NAME,
    VERSION=VERSION,
    MAX_SAFE_BOOST=MAX_SAFE_BOOST,
)


if __name__ == '__main__':
    start_server(
        main,
        port=8080,
        debug=True,
        auto_open_webbrowser=True,
        static_dir=resolve_static_dir(),
    )
