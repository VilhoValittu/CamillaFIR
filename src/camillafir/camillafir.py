import os
import sys

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "camillafir"

import logging
import typing
import re
import unicodedata
import math
import time
from datetime import datetime
import numpy as np
from pywebio import start_server
from pywebio.output import put_processbar, put_scope, set_processbar
from pywebio.pin import pin
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
)
from .ui.camillafir_export import build_export_zip, save_export_bundle
from .ui import camillafir_plot as plots
from .engine import build_config, run_pipeline, summarize_run
from .dsp import camillafir_dsp as dsp
from .dsp.target_match import target_match_from_stats
from .ui.camillafir_ui import _render_results
from .ui.camillafir_utils import scale_taps_with_fs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

VERSION = "v.3.4.0"
PROGRAM_NAME = "CamillaFIR"
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000
MAX_SAFE_TAPS = 131072
TEST_MODE = 0
TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"
AUTO_MODE_TRIALS = 100

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


def _slugify_filename_token(value: typing.Any, *, default: str = "target", max_len: int = 48) -> str:
    """Muuntaa tekstin turvalliseksi tiedostonimiosaksi."""
    try:
        raw = str(value or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return default

    try:
        txt = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    except Exception:
        txt = raw
    txt = re.sub(r"[^A-Za-z0-9]+", "-", txt).strip("-").lower()
    if not txt:
        return default
    if len(txt) > int(max_len):
        txt = txt[: int(max_len)].rstrip("-")
    return txt or default


def _pick_target_curve_label(data: dict) -> str:
    """Valitsee target curven nimen vientitiedostonimiin."""
    try:
        up = data.get("hc_custom_file")
        if isinstance(up, dict):
            for k in ("filename", "name", "file_name"):
                v = up.get(k)
                if isinstance(v, str) and v.strip():
                    return os.path.splitext(os.path.basename(v.strip()))[0]
    except Exception:
        pass

    try:
        p = str(data.get("local_path_house") or "").strip()
    except Exception:
        p = ""
    if p:
        return os.path.splitext(os.path.basename(p))[0]

    try:
        hc_mode = str(data.get("hc_mode") or "").strip()
    except Exception:
        hc_mode = ""
    if hc_mode:
        return hc_mode

    try:
        src = str(data.get("hc_source") or "").strip()
    except Exception:
        src = ""
    if src:
        return src
    return "Target"

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


def _auto_safe_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _auto_collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []


def _auto_pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _auto_dsp_quality_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    real_rms = _auto_pick_metric(
        st,
        (
            "real_mag_error_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)
    dbg["real_rms"] = real_rms

    ripple_rms = _auto_pick_metric(
        st,
        (
            "ripple_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)
    dbg["ripple_rms"] = ripple_rms

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        penalty += 0.60 * max(0.0, float(gd_grad_max) - 12.0)
    dbg["gd_grad_max"] = gd_grad_max

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ),
    )
    if pre_ringing_db is not None:
        penalty += 0.70 * max(0.0, float(pre_ringing_db) + 40.0)
    dbg["pre_ringing_db"] = pre_ringing_db

    pre_post_ratio = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ),
        nonneg=True,
    )
    if pre_post_ratio is not None:
        penalty += 30.0 * max(0.0, float(pre_post_ratio) - 0.015)
    dbg["ir_pre_post_ratio"] = pre_post_ratio

    phase_boundary_mdb = _auto_pick_metric(
        st,
        (
            "phase_boundary_peak_mdb",
            "phase_corr_boundary_peak_mdb",
        ),
        abs_value=True,
        nonneg=True,
    )
    if phase_boundary_mdb is not None:
        penalty += 0.015 * max(0.0, float(phase_boundary_mdb) - 120.0)
    dbg["phase_boundary_peak_mdb"] = phase_boundary_mdb

    return float(max(0.0, penalty)), dbg


def _auto_score_result(result) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = plots.calc_ai_summary_from_stats(l_st)
    r_ai = plots.calc_ai_summary_from_stats(r_st)

    l_score = _auto_safe_float(l_ai.get("score"), 0.0)
    r_score = _auto_safe_float(r_ai.get("score"), 0.0)
    avg_score = (l_score + r_score) / 2.0
    lr_delta = abs(l_score - r_score)

    net_boost_max = max(
        _auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        _auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    events_total = int(len(_auto_collect_reflections(l_st)) + len(_auto_collect_reflections(r_st)))
    dsp_pen_l, dsp_dbg_l = _auto_dsp_quality_penalty(l_st)
    dsp_pen_r, dsp_dbg_r = _auto_dsp_quality_penalty(r_st)
    dsp_penalty = 0.5 * (float(dsp_pen_l) + float(dsp_pen_r))

    boost_pen = 1.5 * max(0.0, net_boost_max - 3.0)
    event_pen = 0.5 * float(events_total)
    lr_pen = 0.25 * lr_delta
    rank_score = float(np.clip(avg_score - boost_pen - event_pen - lr_pen - dsp_penalty, 0.0, 100.0))

    return {
        "rank_score": float(rank_score),
        "avg_score": float(avg_score),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "events_total": int(events_total),
        "dsp_penalty": float(dsp_penalty),
        "dsp_penalty_l": float(dsp_pen_l),
        "dsp_penalty_r": float(dsp_pen_r),
        "dsp_dbg_l": dict(dsp_dbg_l),
        "dsp_dbg_r": dict(dsp_dbg_r),
    }


def _auto_rank_key(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _build_auto_mode_candidates(base_data: dict, *, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_eff = max(1, int(n_trials))

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_linear = ("linear" in ft) and (not is_mixed)
    mixed_center = _auto_safe_float(base_data.get("mixed_freq", 180.0), 180.0)
    if not np.isfinite(mixed_center) or mixed_center <= 0.0:
        mixed_center = 180.0
    phase_center = _auto_safe_float(base_data.get("phase_limit", 600.0), 600.0)
    if not np.isfinite(phase_center) or phase_center <= 0.0:
        phase_center = 600.0

    out: list[dict] = [{}]
    for _ in range(max(0, n_eff - 1)):
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(rng.uniform(8.0, 16.0)), 2),
            "tdc_strength": round(float(rng.uniform(35.0, 75.0)), 1),
            "tdc_max_reduction_db": round(float(rng.uniform(6.0, 12.0)), 1),
            "tdc_slope_db_per_oct": float(rng.choice(np.array([3.0, 4.0, 5.0, 6.0, 8.0]))),
            "reg_strength": round(float(rng.uniform(15.0, 45.0)), 1),
            "max_slope_db_per_oct": float(rng.choice(np.array([8.0, 10.0, 12.0, 14.0, 16.0]))),
            "max_boost": round(float(rng.uniform(3.0, 6.0)), 2),
            "mag_c_max": round(float(rng.uniform(170.0, 300.0)), 1),
            "trans_width": round(float(rng.uniform(70.0, 150.0)), 1),
            "filter_smooth": int(rng.choice(np.array([6, 12, 24, 48, 96]))),
            "bass_first_mode_max_hz": round(float(rng.uniform(150.0, 220.0)), 1),
            "low_bass_cut_hz": round(float(rng.uniform(20.0, 45.0)), 1),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(loc=mixed_center, scale=35.0), 80.0, 320.0)), 1)
        if is_linear:
            cand["phase_limit"] = round(float(np.clip(rng.normal(loc=phase_center, scale=140.0), 150.0, 1400.0)), 1)
        out.append(cand)
    return out


def _run_auto_mode_search(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int = AUTO_MODE_TRIALS,
) -> dict | None:
    seed = int(20260302 + int(fs_v) * 17 + int(taps_v))
    candidates = _build_auto_mode_candidates(base_data, n_trials=int(n_trials), seed=seed)

    best_result = None
    best_metrics = None
    best_preset = None
    scored = []

    for idx, preset in enumerate(candidates, start=1):
        trial_data = dict(base_data or {})
        trial_data.update(preset or {})
        trial_data["comparison_mode"] = True
        trial_measurements = dict(measurements or {})
        trial_measurements["ui_data"] = trial_data

        try:
            cfg = build_config(
                trial_data,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin=pin_obj,
                max_safe_boost=float(MAX_SAFE_BOOST),
            )
            try:
                setattr(cfg, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
                setattr(cfg, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
            except Exception:
                pass

            result = run_pipeline(cfg, trial_measurements)
            result.metrics["summary"] = summarize_run(result)
            metrics = _auto_score_result(result)
            metrics["trial"] = int(idx)
            scored.append({"metrics": metrics, "preset": dict(preset or {})})

            if best_metrics is None or _auto_rank_key(metrics) < _auto_rank_key(best_metrics):
                best_result = result
                best_metrics = metrics
                best_preset = dict(preset or {})
        except Exception as exc:
            logger.warning(f"Automatic mode trial {idx}/{len(candidates)} failed: {type(exc).__name__}: {exc}")

        if callable(status_cb):
            best_txt = "n/a" if not best_metrics else f"{_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}"
            status_cb(f"CamillaFIR automatic mode: {idx}/{len(candidates)} trials (best {best_txt}/100)")

    if best_result is None or best_metrics is None:
        return None

    top = sorted(scored, key=lambda x: _auto_rank_key(x.get("metrics", {})))[:5]
    return {
        "best_result": best_result,
        "best_metrics": dict(best_metrics),
        "best_preset": dict(best_preset or {}),
        "top": top,
        "trials_total": int(len(candidates)),
        "trials_ok": int(len(scored)),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }


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
    data["bass_adaptive_isolation_mode"] = True
    # Target-lean trial tuning for bass adaptive smoothing.
    data["bass_smooth_sigma_scale"] = 1.20
    data["bass_smooth_conf_floor"] = 0.25
    data["bass_smooth_w_gamma"] = 2.40
    data["bass_smooth_w_max"] = 0.45
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
    target_curve_name = _pick_target_curve_label(data)
    target_curve_tag = _slugify_filename_token(target_curve_name, default="target")
    data["target_curve_name"] = target_curve_name
    data["target_curve_tag"] = target_curve_tag
    logger.info(f"House curve source: {hc_source}")
    logger.info(f"Export target curve tag: {target_curve_tag} (from '{target_curve_name}')")
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
    log_df_smoothing_toggle(pin, logger)


    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(target_rates, multi_rate_on=multi_rate_on, forced_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ))
    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    auto_mode_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    zip_dashboards_on = False

    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = filter_type_short(data['filter_type'])
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None
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

    is_wav_source = detect_is_wav_source(data, pin)
    data["_is_wav_source"] = bool(is_wav_source)

    results_by_fs = []
    measurements = {
        "f_l": np.asarray(f_l, dtype=float),
        "m_l": np.asarray(m_l, dtype=float),
        "p_l": np.asarray(p_l, dtype=float),
        "f_r": np.asarray(f_r, dtype=float),
        "m_r": np.asarray(m_r, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "hc_f": hc_f,
        "hc_m": hc_m,
        "ui_data": data,
        "is_wav_source": bool(is_wav_source),
    }

    if auto_mode_enabled:
        try:
            data["comparison_mode"] = True
            auto_search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
            if bool(data.get("multi_rate_opt", False)):
                auto_search_taps = int(scale_taps_with_fs(auto_search_fs, taps_base))
            else:
                auto_search_taps = int(taps_base)
            set_processbar('bar', 0.10)
            _status(
                f"CamillaFIR automatic mode: searching best preset "
                f"({AUTO_MODE_TRIALS} trials @ {auto_search_fs} Hz)"
            )
            auto_res = _run_auto_mode_search(
                base_data=dict(data),
                measurements=measurements,
                fs_v=int(auto_search_fs),
                taps_v=int(auto_search_taps),
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin_obj=pin,
                status_cb=_status,
                n_trials=int(AUTO_MODE_TRIALS),
            )
            if isinstance(auto_res, dict):
                best_preset = dict(auto_res.get("best_preset", {}) or {})
                best_metrics = dict(auto_res.get("best_metrics", {}) or {})
                if best_preset:
                    data.update(best_preset)
                    measurements["ui_data"] = data
                data["_auto_mode_meta"] = {
                    "enabled": True,
                    "trials_total": int(auto_res.get("trials_total", AUTO_MODE_TRIALS)),
                    "trials_ok": int(auto_res.get("trials_ok", 0)),
                    "search_fs": int(auto_res.get("search_fs", auto_search_fs)),
                    "search_taps": int(auto_res.get("search_taps", auto_search_taps)),
                    "best_metrics": best_metrics,
                    "best_preset": best_preset,
                    "top": list(auto_res.get("top", []) or []),
                }
                logger.info(
                    "Automatic mode best: "
                    f"rank={_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}/100, "
                    f"avg={_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
                    f"dsp_pen={_auto_safe_float(best_metrics.get('dsp_penalty'), 0.0):.3f}, "
                    f"boost={_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
                    f"events={int(best_metrics.get('events_total', 0) or 0)}"
                )
            else:
                logger.warning("Automatic mode could not produce a valid best preset; using current settings.")
            set_processbar('bar', 0.18)
        except Exception as exc:
            logger.warning(f"Automatic mode failed: {type(exc).__name__}: {exc}")

    for i, fs_v in enumerate(target_rates):
        if bool(data.get('multi_rate_opt', False)):
            taps_v = scale_taps_with_fs(fs_v, taps_base)
            logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
        else:
            taps_v = taps_base

        _status(f"{t('stat_calc')} {fs_v}Hz...")
        set_processbar('bar', 0.2 + 0.6 * (i / len(target_rates)))
        data["enable_residual_pass"] = bool(data.get("enable_residual_pass", False))

        cfg = build_config(
            data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin=pin,
            max_safe_boost=float(MAX_SAFE_BOOST),
        )
        try:
            setattr(cfg, "bass_smooth_w_gamma", float(data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg, "bass_smooth_w_max", float(data.get("bass_smooth_w_max", 0.45)))
        except Exception:
            pass

        _status(f"{t('stat_calc')} {int(fs_v)} Hz")
        _t_dsp = time.perf_counter()
        result = run_pipeline(cfg, measurements)
        result.metrics["summary"] = summarize_run(result)
        _dsp_dt = max(0.0, float(time.perf_counter() - _t_dsp))

        perf_stats["dsp_s"] += _dsp_dt
        _fs_k = int(fs_v)
        _slot = per_fs_stats.setdefault(_fs_k, {})
        _slot["dsp_s"] = float(_slot.get("dsp_s", 0.0)) + _dsp_dt
        results_by_fs.append(result)

        if int(fs_v) == int(dash_fs):
            l_st_f, r_st_f = result.l_st, result.r_st
            l_imp_f, r_imp_f = result.l_ir, result.r_ir

    if not results_by_fs:
        toast_measurement_files_missing()
        return

    _t_zip = time.perf_counter()
    zip_buffer, ui_dashboards, zip_perf = build_export_zip(
        data=data,
        results=results_by_fs,
        ft_short=ft_short,
        file_ts=file_ts,
        irw_tag=irw_tag,
        write_dashboards=zip_dashboards_on,
        dash_fs=int(dash_fs),
    )
    _zip_dt = max(0.0, float(time.perf_counter() - _t_zip))
    perf_stats["zip_png_s"] += max(float(zip_perf.get("zip_png_s", 0.0) or 0.0), _zip_dt)
    for fs_k, st in (zip_perf.get("per_fs_stats", {}) or {}).items():
        slot = per_fs_stats.setdefault(int(fs_k), {})
        slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + float(st.get("zip_png_s", 0.0) or 0.0)

    fname, saved_filters_dir, _save_msg = save_export_bundle(
        zip_buffer,
        ft_short=ft_short,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        ts=ts,
    )

    if l_st_f is None or r_st_f is None or l_imp_f is None or r_imp_f is None:
        fallback = results_by_fs[-1]
        l_st_f, r_st_f = fallback.l_st, fallback.r_st
        l_imp_f, r_imp_f = fallback.l_ir, fallback.r_ir

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
        saved_filters_dir=saved_filters_dir,
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
    - `Psychoacoustic`: REW-tyylinen painotettu LF/HF-yhdistelma psyko=CamillaFIR Reference
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
            from .dsp.smoothing import psychoacoustic_smoothing
            return psychoacoustic_smoothing(f, m)
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
        st[f"{key_g}_source"] = "ir_fft_final"
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
