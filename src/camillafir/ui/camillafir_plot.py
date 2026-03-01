import os
import sys
import io, scipy.signal, scipy.fft, scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
from ..dsp.smoothing import apply_smoothing_std, psychoacoustic_smoothing
from ..dsp.quality_metrics import _mag_error_db, _rms
from ..dsp.target_match import target_match_from_stats as _target_match_from_stats_ssot
from ..resources.i8n.camillafir_i18n import t
PHASE_SMOOTH_OCT = 5.5
GD_SMOOTH_OCT    = 3.0


def _float_allow_zero(v, default: float) -> float:
    """Sisainen apufunktio: float allow zero."""
    if v is None:
        return float(default)
    if isinstance(v, str) and v.strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

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


def smooth_complex(freqs, spec, oct_frac=1.0):
    """Kasittelee signaalia tai dataa: smooth complex."""
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    return real_s + 1j * imag_s

def calculate_clean_gd(freqs, complex_resp):
    """Laskee: calculate clean gd."""
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=8)

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(lo)

def calc_acoustic_score(conf_pct: float, match_pct: float, rt60_s: float | None = None, rt60_rel: float | None = None) -> float:
    """Laskee: calc acoustic score."""

    conf = _clamp(conf_pct, 0.0, 100.0)
    match = _clamp(match_pct, 0.0, 100.0)

    base = 0.55 * match + 0.35 * conf

    rt_bonus_eff = 0.0
    try:
        if rt60_s is not None:
            rt60 = float(rt60_s)
            if rt60 > 0:
                rel = 1.0 if rt60_rel is None else _clamp(float(rt60_rel), 0.0, 1.0)
                rt_bonus = 15.0 * _clamp((0.35 - rt60) / 0.25, 0.0, 1.0)
                rt_bonus_eff = rt_bonus * rel
    except Exception:
        rt_bonus_eff = 0.0

    return _clamp(base + rt_bonus_eff, 0.0, 100.0)

_calc_acoustic_score = calc_acoustic_score




def calc_ai_summary_from_stats(stats: dict) -> dict:
    """Laskee: calc ai summary from stats."""
    stats = stats or {}
    conf = float(stats.get('cmp_avg_confidence', stats.get('avg_confidence', 0.0)) or 0.0)
    rms, match = calc_target_match_from_stats(stats)
    if match is None:
        return {"conf": conf, "rms": None, "match": None, "score": None}
    rt60 = stats.get("rt60_val", None)
    rt_rel = stats.get("rt60_reliability", None)
    score = calc_acoustic_score(conf, float(match), rt60_s=rt60, rt60_rel=rt_rel)
    return {
        "conf": conf,
        "rms": float(rms) if rms is not None else None,
        "match": float(match),
        "score": score,
        "rt60": float(rt60) if rt60 is not None else None,
        "rt60_rel": float(rt_rel) if rt_rel is not None else None,
    }

def _calc_target_match(stats):
    """Sisainen apufunktio: calc target match."""
    return _target_match_from_stats_ssot(
        stats or {},
        include_filter=True,
        use_confidence=True,
        use_smart_scan_range=True,
    )

def calc_target_match_from_stats(stats: dict):
    """Laskee: calc target match from stats."""
    try:
        return _calc_target_match(stats or {})
    except Exception:
        return None, None


def format_dsp_quality_report_block(settings, l_stats, r_stats):
    """Rakenna Summaryyn DSP Quality Report -blokki (L/R)."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    def _safe_float(v):
        try:
            x = float(v)
            if np.isfinite(x):
                return float(x)
        except Exception:
            pass
        return None

    def _as_arr(v):
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
        if a.size == 0:
            return np.asarray([], dtype=float)
        return a

    def _gd_grad_max_value(st):
        for k in (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _gd_grad_max_hz(st):
        for k in (
            "gd_grad_limiter_peak_hz",
            "gd_limiter_max_grad_hz",
            "gd_grad_limiter_max_grad_hz",
            "gd_limiter_max_grad_after_hz",
            "gd_grad_limiter_max_grad_after_hz",
            "gd_limiter_max_grad_before_hz",
            "gd_grad_limiter_max_grad_before_hz",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _gd_max_value(st):
        for k in (
            "gd_max_ms",
            "max_gd_ms",
            "mixed_excess_delay_before_ms",
            "xo_diff_raw_max_gd_ms",
            "hpf_diff_raw_max_gd_ms",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return abs(v)
        refs = st.get("reflections", []) or []
        vals = []
        for ref in refs:
            if isinstance(ref, dict):
                g = _safe_float(ref.get("gd_error", None))
                if g is not None:
                    vals.append(abs(g))
        if vals:
            return float(max(vals))
        return None

    def _pre_ringing_db(st):
        if bool(st.get("pre_energy_metric_suspect", False)):
            return None
        for k in (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _pre_post_ratio(st, pre_db):
        if bool(st.get("pre_energy_metric_suspect", False)):
            return None
        for k in (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None and v >= 0.0:
                return v
        if pre_db is None:
            return None
        try:
            r = float(10.0 ** (float(pre_db) / 10.0))
            return r if np.isfinite(r) else None
        except Exception:
            return None

    def _pre_metric_info(st):
        suspect = bool(st.get("pre_energy_metric_suspect", False))
        note = str(st.get("pre_energy_metric_note", "") or "").strip()
        if suspect and not note:
            note = "pre/post < 1e-10 (likely zeroed or split issue)"
        return suspect, note

    def _active_axes(st):
        mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
        if mode == "comparison":
            f = _as_arr(st.get("cmp_freq_axis", None))
            t = _as_arr(st.get("cmp_target_mags", None))
            mm = _as_arr(st.get("cmp_mag_mask", None))
            offset_db = _safe_float(st.get("cmp_offset_db", 0.0))
            if offset_db is None:
                offset_db = 0.0
            m_raw = _as_arr(st.get("cmp_measured_mags_raw", None))
            m_corr = _as_arr(st.get("cmp_measured_mags", None))
            g_pred = _as_arr(st.get("cmp_predicted_filter_mags", None))
            g_real = _as_arr(st.get("cmp_realized_filter_mags", None))
            g_legacy = _as_arr(st.get("cmp_filter_mags", None))
            g_legacy_src = str(st.get("cmp_filter_mags_source", "") or "").strip().lower()
            if m_raw.size < 8 and m_corr.size >= 8:
                m_raw = np.asarray(m_corr, dtype=float) + float(offset_db)
            if g_pred.size < 8 and g_legacy.size >= 8 and g_legacy_src != "ir_fft_final":
                g_pred = g_legacy
            if g_real.size < 8 and g_legacy.size >= 8 and g_legacy_src == "ir_fft_final":
                g_real = g_legacy
            if g_pred.size < 8 and g_legacy.size >= 8:
                g_pred = g_legacy
            if g_real.size < 8 and g_legacy.size >= 8:
                g_real = g_legacy
            return f, m_raw, t, g_pred, g_real, mm, float(offset_db)

        f = _as_arr(st.get("freq_axis", None))
        t = _as_arr(st.get("target_mags", None))
        mm = _as_arr(st.get("mag_mask", st.get("mask_c", None)))
        offset_db = _safe_float(st.get("offset_db", 0.0))
        if offset_db is None:
            offset_db = 0.0
        m_raw = _as_arr(st.get("measured_mags_raw", None))
        m_corr = _as_arr(st.get("measured_mags", None))
        g_pred = _as_arr(st.get("predicted_filter_mags", None))
        g_real = _as_arr(st.get("realized_filter_mags", None))
        g_legacy = _as_arr(st.get("filter_mags", None))
        g_legacy_src = str(st.get("filter_mags_source", "") or "").strip().lower()
        if m_raw.size < 8 and m_corr.size >= 8:
            m_raw = np.asarray(m_corr, dtype=float) + float(offset_db)
        if g_pred.size < 8 and g_legacy.size >= 8 and g_legacy_src != "ir_fft_final":
            g_pred = g_legacy
        if g_real.size < 8 and g_legacy.size >= 8 and g_legacy_src == "ir_fft_final":
            g_real = g_legacy
        if g_pred.size < 8 and g_legacy.size >= 8:
            g_pred = g_legacy
        if g_real.size < 8 and g_legacy.size >= 8:
            g_real = g_legacy
        return f, m_raw, t, g_pred, g_real, mm, float(offset_db)

    def _phase_limit_hz(st):
        for k in ("phase_limit_hz", "phase_limit"):
            v = _safe_float(st.get(k, None))
            if v is not None and v > 0.0:
                return float(v)
        v = _safe_float(settings.get("phase_limit", None))
        if v is not None and v > 0.0:
            return float(v)
        return None

    def _phase_boundary_peak_mdb(freqs, filt_db, valid_mask, phase_lim_hz):
        if phase_lim_hz is None:
            return None, None
        f = np.asarray(freqs, dtype=float)
        g = np.asarray(filt_db, dtype=float)
        valid = np.asarray(valid_mask, dtype=bool)
        if f.size < 32 or g.size != f.size or valid.size != f.size:
            return None, None
        lim = float(phase_lim_hz)
        lo = max(20.0, 0.75 * lim)
        hi = min(float(np.max(f[np.isfinite(f)])) if np.any(np.isfinite(f)) else lim, 1.20 * lim)
        if hi <= (lo + 1.0):
            return None, None
        sel = valid & (f >= lo) & (f <= hi)
        if np.count_nonzero(sel) < 12:
            return None, None
        fv = f[sel]
        gv = g[sel]
        sigma = float(np.clip(float(gv.size) / 18.0, 1.0, 14.0))
        try:
            trend = scipy.ndimage.gaussian_filter1d(gv, sigma=sigma, mode="nearest")
            resid = gv - trend
            idx = int(np.argmax(np.abs(resid)))
            return float(np.abs(resid[idx]) * 1000.0), float(fv[idx])
        except Exception:
            return None, None

    def _collect(st):
        out = {
            "pred_mag_error_rms": None,
            "pred_mag_error_max": None,
            "pred_mag_error_max_hz": None,
            "pred_mag_error_rms_20_200": None,
            "pred_mag_error_max_20_200": None,
            "pred_mag_error_rms_200_2000": None,
            "real_mag_error_rms": None,
            "real_mag_error_max": None,
            "real_mag_error_max_hz": None,
            "real_mag_error_rms_20_200": None,
            "real_mag_error_max_20_200": None,
            "real_mag_error_rms_200_2000": None,
            "mid_refit_enabled": bool(st.get("mid_refit_enabled", False)),
            "mid_refit_reason": str(st.get("mid_refit_reason", "") or ""),
            "mid_refit_k": _safe_float(st.get("mid_refit_k", None)),
            "mid_refit_err_rms_before": _safe_float(st.get("mid_refit_err_rms_before", None)),
            "mid_refit_err_rms_after": _safe_float(st.get("mid_refit_err_rms_after", None)),
            "mid_refit_delta_rms": _safe_float(st.get("mid_refit_delta_rms", None)),
            "mid_refit_conf_avg_200_2000": _safe_float(st.get("mid_refit_conf_avg_200_2000", None)),
            "ripple_rms": None,
            "gd_max": _gd_max_value(st),
            "gd_grad_max": _gd_grad_max_value(st),
            "gd_grad_max_hz": _gd_grad_max_hz(st),
            "phase_boundary_peak_mdb": None,
            "phase_boundary_peak_hz": None,
            "bass_adaptive_enabled": bool(st.get("bass_adaptive_smoothing_enabled", False)),
            "bass_adaptive_conf_source": str(st.get("bass_adaptive_conf_source", "") or ""),
            "bass_adaptive_isolation_mode": bool(st.get("bass_adaptive_isolation_mode", False)),
            "bass_adaptive_sigma_scale": _safe_float(st.get("bass_adaptive_smoothing_sigma_scale", None)),
            "bass_adaptive_conf_floor": _safe_float(st.get("bass_adaptive_smoothing_conf_floor", None)),
            "bass_adaptive_w_gamma": _safe_float(st.get("bass_adaptive_smoothing_w_gamma", None)),
            "bass_adaptive_w_max": _safe_float(st.get("bass_adaptive_smoothing_w_max", None)),
            "bass_adaptive_avg_w": _safe_float(st.get("bass_adaptive_smoothing_avg_w_20_200", None)),
            "bass_adaptive_delta_rms_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_rms_db_20_200", None)),
            "bass_adaptive_delta_max_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_max_db_20_200", None)),
            "bass_adaptive_delta_max_hz_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_max_hz_20_200", None)),
            "bass_adaptive_delta_basis": str(st.get("bass_adaptive_smoothing_delta_basis", "") or ""),
            "bass_adaptive_effectiveness_pct": None,
            "post_to_ir_delta_rms_20_200": _safe_float(st.get("post_to_ir_delta_rms_20_200_db", None)),
            "post_to_ir_delta_max_20_200": _safe_float(st.get("post_to_ir_delta_max_20_200_db", None)),
            "post_to_ir_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_delta_max_hz_20_200", None)),
            "post_to_ir_delta_offset_20_200": _safe_float(st.get("post_to_ir_delta_offset_20_200_db", None)),
            "post_to_ir_shape_delta_rms_20_200": _safe_float(st.get("post_to_ir_shape_delta_rms_20_200_db", None)),
            "post_to_ir_shape_delta_max_20_200": _safe_float(st.get("post_to_ir_shape_delta_max_20_200_db", None)),
            "post_to_ir_shape_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_shape_delta_max_hz_20_200", None)),
            "post_to_ir_staged_delta_rms_20_200": _safe_float(st.get("post_to_ir_staged_delta_rms_20_200_db", None)),
            "post_to_ir_staged_delta_max_20_200": _safe_float(st.get("post_to_ir_staged_delta_max_20_200_db", None)),
            "post_to_ir_staged_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_staged_delta_max_hz_20_200", None)),
            "post_to_ir_staged_delta_offset_20_200": _safe_float(st.get("post_to_ir_staged_delta_offset_20_200_db", None)),
            "post_to_ir_staged_shape_delta_rms_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_rms_20_200_db", None)),
            "post_to_ir_staged_shape_delta_max_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_max_20_200_db", None)),
            "post_to_ir_staged_shape_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_max_hz_20_200", None)),
            "ir_realized_level_match_enabled": bool(st.get("ir_realized_level_match_enabled", False)),
            "ir_realized_level_match_applied": bool(st.get("ir_realized_level_match_applied", False)),
            "ir_realized_level_match_reason": str(st.get("ir_realized_level_match_reason", "") or ""),
            "ir_realized_level_match_mid_lo_hz": _safe_float(st.get("ir_realized_level_match_mid_lo_hz", None)),
            "ir_realized_level_match_mid_hi_hz": _safe_float(st.get("ir_realized_level_match_mid_hi_hz", None)),
            "ir_realized_level_match_delta_db_raw": _safe_float(st.get("ir_realized_level_match_delta_db_raw", None)),
            "ir_realized_level_match_delta_db_applied": _safe_float(st.get("ir_realized_level_match_delta_db_applied", None)),
            "ir_realized_level_match_delta_db_after": _safe_float(st.get("ir_realized_level_match_delta_db_after", None)),
            "ir_realized_level_match_scale": _safe_float(st.get("ir_realized_level_match_scale", None)),
            "post_to_ir_delta_rms_magc": _safe_float(st.get("post_to_ir_delta_rms_magc_db", None)),
            "post_to_ir_delta_max_magc": _safe_float(st.get("post_to_ir_delta_max_magc_db", None)),
            "post_to_ir_delta_max_hz_magc": _safe_float(st.get("post_to_ir_delta_max_hz_magc", None)),
            "bass_boost_cap_enabled": bool(st.get("bass_boost_cap_enabled", False)),
            "bass_boost_cap_avg_extra_db_20_200": _safe_float(st.get("bass_boost_cap_avg_extra_db_20_200", None)),
            "bass_boost_cap_max_extra_db_20_200": _safe_float(st.get("bass_boost_cap_max_extra_db_20_200", None)),
            "bass_boost_post_restore_applied": bool(st.get("bass_boost_post_restore_applied", False)),
            "bass_boost_post_restore_strength": _safe_float(st.get("bass_boost_post_restore_strength", None)),
            "bass_boost_post_restore_bins": _safe_float(st.get("bass_boost_post_restore_bins", None)),
            "bass_boost_post_restore_delta_rms_20_200": _safe_float(st.get("bass_boost_post_restore_delta_rms_20_200", None)),
            "bass_boost_post_restore_delta_max_20_200": _safe_float(st.get("bass_boost_post_restore_delta_max_20_200", None)),
            "conf_pull_bass_boost_floor_hz": _safe_float(st.get("conf_pull_post_bass_boost_floor_hz", None)),
            "conf_pull_bass_boost_floor_min": _safe_float(st.get("conf_pull_post_bass_boost_floor_min", None)),
            "conf_pull_bass_boost_restore": _safe_float(st.get("conf_pull_post_bass_boost_restore", None)),
            "conf_pull_bass_boost_restore_mean_eff": _safe_float(st.get("conf_pull_post_bass_boost_restore_mean_eff", None)),
            "pre_ringing_db": None,
            "ir_pre_post_ratio": None,
            "pre_metric_suspect": False,
            "pre_metric_note": "",
        }
        f, m_raw, t, g_pred, g_real, mm, offset_db = _active_axes(st)
        if min(f.size, m_raw.size, t.size, g_pred.size) < 8:
            pre_db = _pre_ringing_db(st)
            out["pre_ringing_db"] = pre_db
            out["ir_pre_post_ratio"] = _pre_post_ratio(st, pre_db)
            out["pre_metric_suspect"], out["pre_metric_note"] = _pre_metric_info(st)
            return out

        cmin = _safe_float(st.get("mag_c_min", settings.get("mag_c_min", 20.0)))
        cmax = _safe_float(st.get("mag_c_max", settings.get("mag_c_max", 20000.0)))
        if cmin is None:
            cmin = 20.0
        if cmax is None or cmax <= cmin:
            cmax = float(np.max(f)) if f.size else (cmin + 1.0)

        def _compute_error_bundle(freqs, measured_raw, target_db, gain_db, mag_mask, *, include_ripple=False):
            res = {
                "rms_magc": None,
                "max_magc": None,
                "max_hz_magc": None,
                "rms_20_200": None,
                "max_20_200": None,
                "rms_200_2000": None,
                "ripple_rms": None,
                "valid": None,
            }
            n_loc = int(min(freqs.size, measured_raw.size, target_db.size, gain_db.size))
            if n_loc < 8:
                return res
            f_loc = np.asarray(freqs[:n_loc], dtype=float)
            m_loc = np.asarray(measured_raw[:n_loc], dtype=float)
            t_loc = np.asarray(target_db[:n_loc], dtype=float)
            g_loc = np.asarray(gain_db[:n_loc], dtype=float)
            if mag_mask.size >= n_loc:
                mm_loc = np.asarray(mag_mask[:n_loc], dtype=float)
            else:
                mm_loc = np.asarray([], dtype=float)

            valid = (
                np.isfinite(f_loc)
                & np.isfinite(m_loc)
                & np.isfinite(t_loc)
                & np.isfinite(g_loc)
                & (f_loc > 0.0)
            )
            err = _mag_error_db(t_loc, m_loc, g_loc, float(offset_db))

            def _band_stats(lo_hz: float, hi_hz: float):
                m_band = valid & (f_loc >= float(lo_hz)) & (f_loc <= float(hi_hz))
                if np.count_nonzero(m_band) < 8:
                    return None, None, None
                ev_loc = np.asarray(err[m_band], dtype=float)
                fv_loc = np.asarray(f_loc[m_band], dtype=float)
                rms_loc = _rms(ev_loc)
                if not np.isfinite(rms_loc):
                    rms_loc = None
                max_loc = None
                hz_loc = None
                if ev_loc.size:
                    idx_loc = int(np.argmax(np.abs(ev_loc)))
                    max_loc = float(np.abs(ev_loc[idx_loc]))
                    if fv_loc.size > idx_loc:
                        hz_loc = float(fv_loc[idx_loc])
                return rms_loc, max_loc, hz_loc

            rms_20_200, max_20_200, _ = _band_stats(20.0, 200.0)
            rms_200_2000, _, _ = _band_stats(200.0, 2000.0)
            res["rms_20_200"] = rms_20_200
            res["max_20_200"] = max_20_200
            res["rms_200_2000"] = rms_200_2000

            if mm_loc.size == n_loc:
                band = np.asarray(mm_loc, dtype=float) > 0.5
            else:
                band = (f_loc >= float(cmin)) & (f_loc <= float(cmax))
            mask = valid & band
            if np.count_nonzero(mask) < 8:
                mask = valid & (f_loc >= float(cmin)) & (f_loc <= float(cmax))
            if np.count_nonzero(mask) >= 8:
                ev = np.asarray(err[mask], dtype=float)
                fv = np.asarray(f_loc[mask], dtype=float)
                rms_magc = _rms(ev)
                res["rms_magc"] = rms_magc if np.isfinite(rms_magc) else None
                if ev.size:
                    idx = int(np.argmax(np.abs(ev)))
                    res["max_magc"] = float(np.abs(ev[idx]))
                    res["max_hz_magc"] = float(fv[idx]) if fv.size else None
                if include_ripple:
                    try:
                        sigma = max(1.0, float(ev.size) / 64.0)
                        ev_sm = scipy.ndimage.gaussian_filter1d(ev, sigma=sigma)
                        rp = ev - ev_sm
                        rp_rms = _rms(rp)
                        res["ripple_rms"] = rp_rms if np.isfinite(rp_rms) else None
                    except Exception:
                        res["ripple_rms"] = None

            res["valid"] = valid
            return res

        pred = _compute_error_bundle(f, m_raw, t, g_pred, mm, include_ripple=True)
        out["pred_mag_error_rms"] = pred["rms_magc"]
        out["pred_mag_error_max"] = pred["max_magc"]
        out["pred_mag_error_max_hz"] = pred["max_hz_magc"]
        out["pred_mag_error_rms_20_200"] = pred["rms_20_200"]
        out["pred_mag_error_max_20_200"] = pred["max_20_200"]
        out["pred_mag_error_rms_200_2000"] = pred["rms_200_2000"]
        out["ripple_rms"] = pred["ripple_rms"]

        real = _compute_error_bundle(f, m_raw, t, g_real, mm, include_ripple=False)
        out["real_mag_error_rms"] = real["rms_magc"]
        out["real_mag_error_max"] = real["max_magc"]
        out["real_mag_error_max_hz"] = real["max_hz_magc"]
        out["real_mag_error_rms_20_200"] = real["rms_20_200"]
        out["real_mag_error_max_20_200"] = real["max_20_200"]
        out["real_mag_error_rms_200_2000"] = real["rms_200_2000"]

        n_phase = int(min(f.size, g_real.size))
        if n_phase < 8:
            n_phase = int(min(f.size, g_pred.size))
            g_phase = np.asarray(g_pred[:n_phase], dtype=float)
        else:
            g_phase = np.asarray(g_real[:n_phase], dtype=float)
        f_phase = np.asarray(f[:n_phase], dtype=float)
        valid_phase = np.isfinite(f_phase) & np.isfinite(g_phase) & (f_phase > 0.0)
        phase_lim = _phase_limit_hz(st)
        pb_mdb, pb_hz = _phase_boundary_peak_mdb(f_phase, g_phase, valid_phase, phase_lim)
        out["phase_boundary_peak_mdb"] = pb_mdb
        out["phase_boundary_peak_hz"] = pb_hz

        pre_db = _pre_ringing_db(st)
        out["pre_ringing_db"] = pre_db
        out["ir_pre_post_ratio"] = _pre_post_ratio(st, pre_db)
        out["pre_metric_suspect"], out["pre_metric_note"] = _pre_metric_info(st)
        try:
            dmax = _safe_float(out.get("bass_adaptive_delta_max_20_200", None))
            emax = _safe_float(out.get("pred_mag_error_max_20_200", None))
            if dmax is not None and emax is not None and emax > 1e-9:
                out["bass_adaptive_effectiveness_pct"] = float((dmax / emax) * 100.0)
        except Exception:
            pass
        return out

    def _fmt(v, unit="", prec=2):
        if v is None:
            return "n/a"
        try:
            x = float(v)
            if not np.isfinite(x):
                return "n/a"
            return f"{x:.{int(prec)}f}{unit}"
        except Exception:
            return "n/a"

    def _fmt_ratio(v):
        if v is None:
            return "n/a"
        try:
            x = float(v)
            if not np.isfinite(x):
                return "n/a"
            return f"{x:.4g}"
        except Exception:
            return "n/a"

    lq = _collect(l_stats)
    rq = _collect(r_stats)
    debug_report = bool(settings.get("quality_report_debug", True)) # Raportin laatu-debug
    def _fmt_onoff(v):
        return "ON" if bool(v) else "OFF"
    def _fmt_src(v):
        s = str(v or "").strip()
        return s if s else "n/a"
    lines = [
        "",
        "--- DSP Quality Report ---",
        f"Predicted mag error RMS within mag_c band:      L {_fmt(lq['pred_mag_error_rms'], ' dB')} | R {_fmt(rq['pred_mag_error_rms'], ' dB')}",
        f"Predicted mag error max within mag_c band:      L {_fmt(lq['pred_mag_error_max'], ' dB')} | R {_fmt(rq['pred_mag_error_max'], ' dB')}",
        f"Predicted mag error max within mag_c band @ Hz: L {_fmt(lq['pred_mag_error_max_hz'], '', 1)} | R {_fmt(rq['pred_mag_error_max_hz'], '', 1)}",
        f"Predicted mag error RMS @ 20-200 Hz:            L {_fmt(lq['pred_mag_error_rms_20_200'], ' dB')} | R {_fmt(rq['pred_mag_error_rms_20_200'], ' dB')}",
        f"Predicted mag error max @ 20-200 Hz:            L {_fmt(lq['pred_mag_error_max_20_200'], ' dB')} | R {_fmt(rq['pred_mag_error_max_20_200'], ' dB')}",
        f"Predicted mag error RMS @ 200-2000 Hz:          L {_fmt(lq['pred_mag_error_rms_200_2000'], ' dB')} | R {_fmt(rq['pred_mag_error_rms_200_2000'], ' dB')}",
        f"Realized mag error RMS within mag_c band:       L {_fmt(lq['real_mag_error_rms'], ' dB')} | R {_fmt(rq['real_mag_error_rms'], ' dB')}",
        f"Realized mag error max within mag_c band:       L {_fmt(lq['real_mag_error_max'], ' dB')} | R {_fmt(rq['real_mag_error_max'], ' dB')}",
        f"Realized mag error max within mag_c band @ Hz:  L {_fmt(lq['real_mag_error_max_hz'], '', 1)} | R {_fmt(rq['real_mag_error_max_hz'], '', 1)}",
        f"Realized mag error RMS @ 20-200 Hz:             L {_fmt(lq['real_mag_error_rms_20_200'], ' dB')} | R {_fmt(rq['real_mag_error_rms_20_200'], ' dB')}",
        f"Realized mag error RMS @ 200-2000 Hz:           L {_fmt(lq['real_mag_error_rms_200_2000'], ' dB')} | R {_fmt(rq['real_mag_error_rms_200_2000'], ' dB')}",
        f"Ripple RMS:              L {_fmt(lq['ripple_rms'], ' dB')} | R {_fmt(rq['ripple_rms'], ' dB')}",
        f"GD max (ms):             L {_fmt(lq['gd_max'], '', 2)} | R {_fmt(rq['gd_max'], '', 2)}",
        f"GD gradient max (ms/oct): L {_fmt(lq['gd_grad_max'], '', 2)} | R {_fmt(rq['gd_grad_max'], '', 2)}",
        f"GD gradient max @ Hz:    L {_fmt(lq['gd_grad_max_hz'], '', 1)} | R {_fmt(rq['gd_grad_max_hz'], '', 1)}",
        f"Phase boundary peak (mdB): L {_fmt(lq['phase_boundary_peak_mdb'], '', 2)} | R {_fmt(rq['phase_boundary_peak_mdb'], '', 2)}",
        f"Phase boundary peak @ Hz:  L {_fmt(lq['phase_boundary_peak_hz'], '', 1)} | R {_fmt(rq['phase_boundary_peak_hz'], '', 1)}",
        f"Pre-ringing dB:          L {_fmt(lq['pre_ringing_db'], ' dB')} | R {_fmt(rq['pre_ringing_db'], ' dB')}",
        f"IR pre/post energy ratio: L {_fmt_ratio(lq['ir_pre_post_ratio'])} | R {_fmt_ratio(rq['ir_pre_post_ratio'])}",
    ]
    if debug_report:
        lines += [
            f"Mid re-fit 200-2000: L {_fmt_onoff(lq['mid_refit_enabled'])} (k {_fmt(lq['mid_refit_k'], '', 2)}, conf {_fmt(lq['mid_refit_conf_avg_200_2000'], '', 2)}) | R {_fmt_onoff(rq['mid_refit_enabled'])} (k {_fmt(rq['mid_refit_k'], '', 2)}, conf {_fmt(rq['mid_refit_conf_avg_200_2000'], '', 2)})",
            f"Mid re-fit err RMS 200-2000 (dB): L {_fmt(lq['mid_refit_err_rms_before'], '', 2)} -> {_fmt(lq['mid_refit_err_rms_after'], '', 2)} (delta {_fmt(lq['mid_refit_delta_rms'], '', 2)}) | R {_fmt(rq['mid_refit_err_rms_before'], '', 2)} -> {_fmt(rq['mid_refit_err_rms_after'], '', 2)} (delta {_fmt(rq['mid_refit_delta_rms'], '', 2)})",
            f"Mid re-fit reason: L {_fmt_src(lq['mid_refit_reason'])} | R {_fmt_src(rq['mid_refit_reason'])}",
            f"Bass adaptive smoothing:  L {_fmt_onoff(lq['bass_adaptive_enabled'])} | R {_fmt_onoff(rq['bass_adaptive_enabled'])}",
            f"Bass adaptive conf source: L {_fmt_src(lq['bass_adaptive_conf_source'])} | R {_fmt_src(rq['bass_adaptive_conf_source'])}",
            f"Bass adaptive isolation mode: L {_fmt_onoff(lq['bass_adaptive_isolation_mode'])} | R {_fmt_onoff(rq['bass_adaptive_isolation_mode'])}",
            f"Bass adaptive params: L sigma {_fmt(lq['bass_adaptive_sigma_scale'], '', 2)} / conf {_fmt(lq['bass_adaptive_conf_floor'], '', 2)} / gamma {_fmt(lq['bass_adaptive_w_gamma'], '', 2)} / wmax {_fmt(lq['bass_adaptive_w_max'], '', 2)} | R sigma {_fmt(rq['bass_adaptive_sigma_scale'], '', 2)} / conf {_fmt(rq['bass_adaptive_conf_floor'], '', 2)} / gamma {_fmt(rq['bass_adaptive_w_gamma'], '', 2)} / wmax {_fmt(rq['bass_adaptive_w_max'], '', 2)}",
            f"Bass adaptive avg w 20-200: L {_fmt(lq['bass_adaptive_avg_w'], '', 3)} | R {_fmt(rq['bass_adaptive_avg_w'], '', 3)}",
            f"Bass adaptive delta RMS 20-200 (dB): L {_fmt(lq['bass_adaptive_delta_rms_20_200'], '', 2)} | R {_fmt(rq['bass_adaptive_delta_rms_20_200'], '', 2)}",
            f"Bass adaptive delta max 20-200 (dB): L {_fmt(lq['bass_adaptive_delta_max_20_200'], '', 2)} | R {_fmt(rq['bass_adaptive_delta_max_20_200'], '', 2)}",
            f"Bass adaptive delta max @ Hz (20-200): L {_fmt(lq['bass_adaptive_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['bass_adaptive_delta_max_hz_20_200'], '', 1)}",
            f"Bass adaptive effectiveness (% of max err 20-200): L {_fmt(lq['bass_adaptive_effectiveness_pct'], '%', 1)} | R {_fmt(rq['bass_adaptive_effectiveness_pct'], '%', 1)}",
            f"Bass adaptive delta basis: L {_fmt_src(lq['bass_adaptive_delta_basis'])} | R {_fmt_src(rq['bass_adaptive_delta_basis'])}",
            f"Post->IR delta RMS 20-200 (dB): L {_fmt(lq['post_to_ir_delta_rms_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_delta_rms_20_200'], '', 2)}",
            f"Post->IR delta max @ Hz (20-200): L {_fmt(lq['post_to_ir_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_delta_max_hz_20_200'], '', 1)}",
            f"Post->IR offset 20-200 (dB): L {_fmt(lq['post_to_ir_delta_offset_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_delta_offset_20_200'], '', 2)}",
            f"Post->IR shape delta RMS/max 20-200 (dB): L {_fmt(lq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_fmt(lq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_fmt(rq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR delta RMS 20-200 (dB): L {_fmt(lq['post_to_ir_staged_delta_rms_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_staged_delta_rms_20_200'], '', 2)}",
            f"Post+staging->IR delta max @ Hz (20-200): L {_fmt(lq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR offset 20-200 (dB): L {_fmt(lq['post_to_ir_staged_delta_offset_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_staged_delta_offset_20_200'], '', 2)}",
            f"Post+staging->IR shape delta RMS/max 20-200 (dB): L {_fmt(lq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_fmt(lq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_fmt(rq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)}",
            f"IR mid-band level match: L {_fmt_onoff(lq['ir_realized_level_match_applied'])} ({_fmt_src(lq['ir_realized_level_match_reason'])}) | R {_fmt_onoff(rq['ir_realized_level_match_applied'])} ({_fmt_src(rq['ir_realized_level_match_reason'])})",
            f"IR mid-band level match delta dB: L raw {_fmt(lq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_fmt(lq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_fmt(lq['ir_realized_level_match_delta_db_after'], '', 2)} | R raw {_fmt(rq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_fmt(rq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_fmt(rq['ir_realized_level_match_delta_db_after'], '', 2)}",
            f"IR mid-band level match scale/band: L x{_fmt(lq['ir_realized_level_match_scale'], '', 4)} @ {_fmt(lq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_fmt(lq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz | R x{_fmt(rq['ir_realized_level_match_scale'], '', 4)} @ {_fmt(rq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_fmt(rq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz",
            f"Bass boost cap: L {_fmt_onoff(lq['bass_boost_cap_enabled'])} | R {_fmt_onoff(rq['bass_boost_cap_enabled'])}",
            f"Bass boost cap extra 20-200 (dB): L avg {_fmt(lq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_fmt(lq['bass_boost_cap_max_extra_db_20_200'], '', 2)} | R avg {_fmt(rq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_fmt(rq['bass_boost_cap_max_extra_db_20_200'], '', 2)}",
            f"Bass boost post-restore: L {_fmt_onoff(lq['bass_boost_post_restore_applied'])} (set {_fmt(lq['bass_boost_post_restore_strength'], '', 2)}, bins {_fmt(lq['bass_boost_post_restore_bins'], '', 0)}) | R {_fmt_onoff(rq['bass_boost_post_restore_applied'])} (set {_fmt(rq['bass_boost_post_restore_strength'], '', 2)}, bins {_fmt(rq['bass_boost_post_restore_bins'], '', 0)})",
            f"Bass boost post-restore delta 20-200 (dB): L rms {_fmt(lq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_fmt(lq['bass_boost_post_restore_delta_max_20_200'], '', 2)} | R rms {_fmt(rq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_fmt(rq['bass_boost_post_restore_delta_max_20_200'], '', 2)}",
            f"Conf-pull bass boost floor: L {_fmt(lq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_fmt(lq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz | R {_fmt(rq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_fmt(rq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz",
            f"Conf-pull bass boost restore: L set {_fmt(lq['conf_pull_bass_boost_restore'], '', 2)} / eff {_fmt(lq['conf_pull_bass_boost_restore_mean_eff'], '', 2)} | R set {_fmt(rq['conf_pull_bass_boost_restore'], '', 2)} / eff {_fmt(rq['conf_pull_bass_boost_restore_mean_eff'], '', 2)}",
        ]
    if bool(lq.get("pre_metric_suspect", False)) or bool(rq.get("pre_metric_suspect", False)):
        l_note = str(lq.get("pre_metric_note", "") or "suspect").strip()
        r_note = str(rq.get("pre_metric_note", "") or "suspect").strip()
        lines.append(f"Pre-energy metric sanity: L {l_note} | R {r_note}")
    return lines




def format_summary_content(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    from datetime import datetime
    import numpy as np

    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}
    program_version = str(settings.get("program_version", "") or "").strip()

    def _safe_float(v, default=0.0):
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            pass
        return float(default)

    def _fmt_score(v):
        return "n/a" if v is None else f"{float(v):.1f}/100"

    def _fmt_match(match_pct, rms_db):
        if match_pct is None or rms_db is None:
            return "n/a"
        return f"{float(match_pct):.1f}% (RMS {float(rms_db):.2f} dB)"

    def _fmt_range(rng):
        if not isinstance(rng, (list, tuple)) or len(rng) < 2:
            return "n/a"
        try:
            return f"{float(rng[0]):.0f}-{float(rng[1]):.0f} Hz"
        except Exception:
            return "n/a"

    def _phase_clamp_line(side: str, st: dict) -> str:
        lim = st.get("phase_corr_clamp_deg", None)
        bef = st.get("phase_corr_max_before_deg", None)
        if lim is None or bef is None:
            return f"{side}: n/a"
        return f"{side}: max {float(bef):.1f} deg -> clamp {float(lim):.1f} deg"

    def _gd_grad_max_value(st: dict):
        keys = (
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        )
        for k in keys:
            try:
                v = float(st.get(k, None))
                if np.isfinite(v):
                    return float(v)
            except Exception:
                continue
        return None

    def _fmt_gd_grad_max(st: dict) -> str:
        v = _gd_grad_max_value(st)
        return "n/a" if v is None else f"{float(v):.2f} ms/oct"

    def _gd_limiter_line(side: str, st: dict) -> str:
        try:
            enabled = bool(st.get("gd_limiter_enabled", st.get("gd_grad_limiter_enabled", False)))
            reason = str(st.get("gd_limiter_reason", st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
            limit_v = st.get("gd_limiter_limit_ms_per_oct", st.get("gd_grad_limit_ms_per_oct", None))
            grad_before = st.get(
                "gd_limiter_max_grad_before_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_before_ms_per_oct", None),
            )
            grad_after = st.get(
                "gd_limiter_max_grad_after_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_after_ms_per_oct", _gd_grad_max_value(st)),
            )

            lim_txt = "n/a"
            if limit_v is not None:
                try:
                    lim_txt = f"{float(limit_v):.2f} ms/oct"
                except Exception:
                    lim_txt = "n/a"

            grad_txt = "n/a"
            try:
                gb = float(grad_before) if grad_before is not None else None
            except Exception:
                gb = None
            try:
                ga = float(grad_after) if grad_after is not None else None
            except Exception:
                ga = None
            if gb is not None and np.isfinite(gb) and ga is not None and np.isfinite(ga):
                grad_txt = f"{gb:.2f} -> {ga:.2f} ms/oct"
            elif ga is not None and np.isfinite(ga):
                grad_txt = f"{ga:.2f} ms/oct"

            return (
                f"{side}: {'ON' if enabled else 'OFF'} "
                f"(reason={reason}, limit={lim_txt}, GD-gradient max {grad_txt})"
            )
        except Exception:
            return f"{side}: n/a"

    def _afdw_line(side: str, st: dict) -> str:
        mode = str(st.get("fdw_mode", "") or "").strip().lower()
        if mode == "fixed":
            cyc = st.get("fdw_fixed_cycles", settings.get("fdw_cycles", None))
            bw = st.get("fdw_fixed_bw_oct", None)
            try:
                cyc_txt = f"{float(cyc):.2f}" if cyc is not None else "n/a"
            except Exception:
                cyc_txt = "n/a"
            try:
                bw_txt = f"{float(bw):.4f}" if bw is not None else "n/a"
            except Exception:
                bw_txt = "n/a"
            return f"{side}: FIXED | cycles={cyc_txt}, BW={bw_txt} oct (A-FDW OFF)"

        active = bool(st.get("afdw_active", False)) or bool(settings.get("enable_afdw", False))
        if not active:
            return f"{side}: OFF"
        mn = st.get("afdw_bw_min_oct", None)
        me = st.get("afdw_bw_mean_oct", None)
        mx = st.get("afdw_bw_max_oct", None)
        if mn is None or me is None or mx is None:
            return f"{side}: ON (effective bandwidth not available)"
        return f"{side}: ON | BW min/mean/max = {float(mn):.4f}/{float(me):.4f}/{float(mx):.4f} oct"

    def _fmt_bands(bands):
        if not bands:
            return "-"
        picks = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
        keys = [float(k) for k in bands.keys()]
        out = []
        for p in picks:
            k = min(keys, key=lambda x: abs(x - p))
            if k in bands:
                val = bands[k]
            elif str(k) in bands:
                val = bands[str(k)]
            else:
                kk = min(bands.keys(), key=lambda x: abs(float(x) - p))
                val = bands[kk]
                k = float(kk)
            out.append(f"{k:.0f}Hz:{float(val):.2f}s")
        return " | ".join(out) if out else "-"

    def _worst_event(st: dict) -> str:
        refs = st.get("reflections", []) or []
        if not refs:
            return "None"
        try:
            w = max(refs, key=lambda x: float(x.get("gd_error", 0.0) or 0.0))
        except Exception:
            return "None"
        freq = _safe_float(w.get("freq", 0.0), 0.0)
        gd_ms = _safe_float(w.get("gd_error", 0.0), 0.0)
        typ = str(w.get("type", "Event") or "Event")
        return f"{typ} at {freq:.0f} Hz ({gd_ms:.2f} ms)"

    def _calc_acoustic_score(conf_pct, match_pct, rt60_s=None, rt60_reliability=None):
        try:
            return globals()["calc_acoustic_score"](
                float(conf_pct),
                float(match_pct),
                rt60_s=rt60_s,
                rt60_rel=rt60_reliability,
            )
        except Exception:
            conf_pct = float(np.clip(float(conf_pct), 0.0, 100.0))
            match_pct = float(np.clip(float(match_pct), 0.0, 100.0))
            return float(np.clip(0.60 * match_pct + 0.40 * conf_pct, 0.0, 100.0))

    l_rt = _safe_float(l_stats.get("rt60_val", 0.0), 0.0)
    r_rt = _safe_float(r_stats.get("rt60_val", 0.0), 0.0)
    l_band_avg = _safe_float(l_stats.get("rt60_band_avg", 0.0), 0.0)
    r_band_avg = _safe_float(r_stats.get("rt60_band_avg", 0.0), 0.0)
    l_conf = _safe_float(l_stats.get("cmp_avg_confidence", l_stats.get("avg_confidence", 0.0)), 0.0)
    r_conf = _safe_float(r_stats.get("cmp_avg_confidence", r_stats.get("avg_confidence", 0.0)), 0.0)
    l_rms, l_match = _calc_target_match(l_stats)
    r_rms, r_match = _calc_target_match(r_stats)
    l_rms_raw, l_match_raw = _target_match_from_stats_ssot(
        l_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    r_rms_raw, r_match_raw = _target_match_from_stats_ssot(
        r_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )

    l_score = None
    if l_match is not None:
        l_score = _calc_acoustic_score(
            l_conf,
            l_match,
            l_stats.get("rt60_val", None),
            l_stats.get("rt60_reliability", None),
        )
    r_score = None
    if r_match is not None:
        r_score = _calc_acoustic_score(
            r_conf,
            r_match,
            r_stats.get("rt60_val", None),
            r_stats.get("rt60_reliability", None),
        )

    lines = [
        "=== CamillaFIR - Filter Generation Summary ===",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if program_version:
        lines.append(f"Version: {program_version}")
    lines += [
        "",
        "--- Executive Summary ---",
        f"Acoustic Score: L {_fmt_score(l_score)} | R {_fmt_score(r_score)}",
        f"Target Match:   L {_fmt_match(l_match, l_rms)} | R {_fmt_match(r_match, r_rms)}",
        f"Confidence:     L {l_conf:.1f}% | R {r_conf:.1f}%",
        f"RT60 Wideband:  L {l_rt:.2f}s | R {r_rt:.2f}s",
        f"Worst Event:    L {_worst_event(l_stats)} | R {_worst_event(r_stats)}",
        "",
        "--- Core Settings ---",
    ]

    keys = [
        "mode",
        "fs",
        "taps",
        "filter_type",
        "mixed_freq",
        "mag_c_min",
        "mag_c_max",
        "max_boost",
        "max_cut_db",
        "max_slope_db_per_oct",
        "hpf_enable",
        "hpf_freq",
        "hpf_slope",
        "enable_tdc",
        "tdc_strength",
        "enable_afdw",
        "comparison_mode",
        "stereo_link",
        "bass_first_ai",
    ]
    for k in keys:
        if k in settings:
            lines.append(f"{k}: {settings.get(k)}")

    lines.append("\n--- Analysis Mode ---")
    lines.append(
        f"Analysis mode: L {str(l_stats.get('analysis_mode', 'native'))} | "
        f"R {str(r_stats.get('analysis_mode', 'native'))}"
    )
    if str(l_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (L): fs={_safe_float(l_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(l_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )
    if str(r_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (R): fs={_safe_float(r_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(r_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )

    max_cut_db = _safe_float(settings.get("max_cut_db", 15.0), 15.0)
    max_slope = _safe_float(settings.get("max_slope_db_per_oct", 12.0), 12.0)
    max_slope_boost = _safe_float(settings.get("max_slope_boost_db_per_oct", 0.0), 0.0) or max_slope
    max_slope_cut = _safe_float(settings.get("max_slope_cut_db_per_oct", 0.0), 0.0) or max_slope
    low_bass_cut_hz = _float_allow_zero(settings.get("low_bass_cut_hz", 40.0), 40.0)

    lines.append("\n--- Correction Guards ---")
    lines.append(f"Max cut: -{max_cut_db:.1f} dB")
    if abs(max_slope_boost - max_slope_cut) > 1e-9:
        lines.append(f"Slope: boost {max_slope_boost:.1f} dB/oct | cut {max_slope_cut:.1f} dB/oct")
    else:
        lines.append(f"Max slope: {max_slope:.1f} dB/oct")
    lines.append(f"Low-bass cut policy: <{low_bass_cut_hz:.1f} Hz (cuts only)")

    lines.append("\n--- Temporal Decay Control (TDC) ---")
    tdc_enabled = bool(settings.get("enable_tdc", False))
    lines.append(f"TDC enabled: {'YES' if tdc_enabled else 'NO'}")
    if tdc_enabled:
        lines.append(f"TDC strength: {_safe_float(settings.get('tdc_strength', 0), 0):.0f} %")
        lines.append(f"TDC max reduction: {_safe_float(settings.get('tdc_max_reduction_db', 0), 0):.1f} dB")
        slope = _safe_float(settings.get("tdc_slope_db_per_oct", 0), 0)
        if slope > 0:
            lines.append(f"TDC slope limit: {slope:.1f} dB/oct")

    lines.append("\n--- A-FDW ---")
    lines.append(_afdw_line("Left", l_stats))
    lines.append(_afdw_line("Right", r_stats))

    lines.append("\n--- XO and Phase ---")
    lines.append(f"XO phase model: L {l_stats.get('xo_summary', '-')} | R {r_stats.get('xo_summary', '-')}")
    lines.append(_phase_clamp_line("L", l_stats))
    lines.append(_phase_clamp_line("R", r_stats))
    lines.append(_gd_limiter_line("L", l_stats))
    lines.append(_gd_limiter_line("R", r_stats))
    lines.append(f"A/B GD-gradient max: L {_fmt_gd_grad_max(l_stats)} | R {_fmt_gd_grad_max(r_stats)}")

    lines.append("\n--- RT60 and Confidence ---")
    lines.append(f"RT60 wideband: L {l_rt:.2f}s | R {r_rt:.2f}s")
    if (l_band_avg > 0.0) or (r_band_avg > 0.0):
        lines.append(f"RT60 band average (125-4kHz): L {l_band_avg:.2f}s | R {r_band_avg:.2f}s")
    l_bands = l_stats.get("rt60_bands", {}) or {}
    r_bands = r_stats.get("rt60_bands", {}) or {}
    if l_bands or r_bands:
        lines.append(f"Band RT60 L: {_fmt_bands(l_bands)}")
        lines.append(f"Band RT60 R: {_fmt_bands(r_bands)}")
    lines.append(f"Confidence: L {l_conf:.1f}% | R {r_conf:.1f}%")

    l_om = l_stats.get("cmp_offset_method", l_stats.get("offset_method", "")) or "-"
    r_om = r_stats.get("cmp_offset_method", r_stats.get("offset_method", "")) or "-"
    lines.append(f"Offset method: L {l_om} | R {r_om}")
    l_win = l_stats.get("cmp_smart_scan_range", l_stats.get("smart_scan_range", None))
    r_win = r_stats.get("cmp_smart_scan_range", r_stats.get("smart_scan_range", None))
    lines.append(f"Level window: L {_fmt_range(l_win)} | R {_fmt_range(r_win)}")

    lines.append("\n--- Target Curve Match ---")
    lines.append(f"Left:  {_fmt_match(l_match, l_rms)}")
    lines.append(f"Right: {_fmt_match(r_match, r_rms)}")
    lines.append(
        "Debug raw->pred: "
        f"L {_fmt_match(l_match_raw, l_rms_raw)} -> {_fmt_match(l_match, l_rms)} | "
        f"R {_fmt_match(r_match_raw, r_rms_raw)} -> {_fmt_match(r_match, r_rms)}"
    )

    lines += format_dsp_quality_report_block(settings, l_stats, r_stats)

    lines.append("\n--- Alignment and Peaks ---")
    lines.append(f"L peak (pre-norm): {_safe_float(l_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"R peak (pre-norm): {_safe_float(r_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"Global offset applied: {_safe_float(l_stats.get('offset_db', 0), 0):.2f} dB")
    lines.append(f"Auto gain margin setting: {_safe_float(settings.get('gain', 0.0), 0.0):.2f} dB")
    lines.append(
        f"Applied auto gain: L {_safe_float(l_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB | "
        f"R {_safe_float(r_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB"
    )

    return "\n".join(lines)
_format_summary_content_legacy = format_summary_content

def _make_comparison_stats(stats: dict, ref_fs: int = 44100, ref_taps: int = 65536) -> dict:
    """Sisainen apufunktio: make comparison stats."""
    stats = stats or {}
    out = copy.deepcopy(stats)

    if str(out.get("analysis_mode", "native")).lower() == "comparison" and ("cmp_freq_axis" in out):
        return out
    f = out.get("freq_axis", None)
    m = out.get("measured_mags", None)
    t = out.get("target_mags", None)
    g = out.get("filter_mags", None)
    c = out.get("confidence_mask", None)
    mm = out.get("mag_mask", out.get("mask_c", None))

    if f is None or m is None or t is None:
        return out  
    
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if g is not None:
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    if c is not None:
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    if mm is not None:
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)


    try:
        f = np.asarray(f, dtype=float)
        m = np.asarray(m, dtype=float)
        t = np.asarray(t, dtype=float)
        g = np.asarray(g, dtype=float) if g is not None else None
        c = np.asarray(c, dtype=float) if c is not None else None
        mm = np.asarray(mm, dtype=float) if mm is not None else None
    except Exception:
        return out

    if f.ndim != 1 or f.size < 32 or m is None or t is None:
        return out
    if (m.ndim != 1) or (t.ndim != 1) or (m.size != f.size) or (t.size != f.size):
        return out
    if (g is not None) and ((g.ndim != 1) or (g.size != f.size)):
        g = None
    if (c is not None) and ((c.ndim != 1) or (c.size != f.size)):
        c = None
    if (mm is not None) and ((mm.ndim != 1) or (mm.size != f.size)):
        mm = None

    nfft = int(ref_taps)
    if nfft < 1024:
        nfft = 1024
    if (nfft % 2) != 0:
        nfft += 1
    fmax = min(float(ref_fs) / 2.0, float(np.max(f)))
    if fmax <= 10.0:
        return out

    freq_cmp = np.linspace(0.0, fmax, nfft // 2 + 1)

    def _interp(y):
        y = np.asarray(y, dtype=float)
        if y.shape != f.shape:
            return None
        return np.interp(freq_cmp, f, y)

    m_cmp = _interp(m)
    t_cmp = _interp(t)
    g_cmp = _interp(g) if g is not None and g.shape == f.shape else None
    c_cmp = _interp(c) if c is not None and c.shape == f.shape else None
    mm_cmp = _interp(mm) if mm is not None and mm.shape == f.shape else None
    if m_cmp is None or t_cmp is None:
        return out
    

    rng = out.get("smart_scan_range", None)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        fmin, fmax_rng = float(rng[0]), float(rng[1])
    else:
        fmin, fmax_rng = 200.0, 5000.0
    mask = (freq_cmp >= fmin) & (freq_cmp <= fmax_rng)
    if np.count_nonzero(mask) >= 20:
        cmp_offset_db = float(np.median((m_cmp - t_cmp)[mask]))
    else:
        cmp_offset_db = 0.0
    m_cmp = m_cmp - cmp_offset_db

    out["analysis_mode"] = "comparison"
    out["cmp_ref_fs"] = int(ref_fs)
    out["cmp_ref_taps"] = int(ref_taps)
    out["cmp_freq_axis"] = freq_cmp.tolist()
    out["cmp_measured_mags"] = m_cmp.tolist()
    out["cmp_target_mags"] = t_cmp.tolist()
    if g_cmp is not None:
        out["cmp_filter_mags"] = g_cmp.tolist()
    if c_cmp is not None:
        out["cmp_confidence_mask"] = np.clip(c_cmp, 0.0, 1.0).tolist()
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)
    if mm_cmp is not None:
        out["cmp_mag_mask"] = (np.asarray(mm_cmp, dtype=float) > 0.5).astype(float).tolist()
    bw = out.get("afdw_bw_oct", None)
    bw_cmp = _interp(bw) if bw is not None and np.asarray(bw).shape == f.shape else None
    if bw_cmp is not None:
        out["cmp_afdw_bw_oct"] = np.clip(bw_cmp, 1.0/96.0, 1.0/3.0).tolist()
        out["cmp_offset_db"] = float(cmp_offset_db)

    if "smart_scan_range" in out and isinstance(out["smart_scan_range"], (list, tuple)) and len(out["smart_scan_range"]) == 2:
        out["cmp_smart_scan_range"] = [float(out["smart_scan_range"][0]), float(out["smart_scan_range"][1])]

    if c_cmp is not None:
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)


    if "eff_target_db" in stats and stats.get("eff_target_db") is not None:
        try:
            v = float(stats.get("eff_target_db"))
            if np.isfinite(v):
                out["eff_target_db"] = v
                out["cmp_eff_target_db"] = v
        except Exception:
            pass

    if "target_level_db_window" in stats:
        out["cmp_target_level_db_window"] = stats.get("target_level_db_window")

    return out

def format_summary_content(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    if bool(settings.get("comparison_mode", False)):
        l_stats = _make_comparison_stats(l_stats, 44100, 65536)
        r_stats = _make_comparison_stats(r_stats, 44100, 65536)

    return _format_summary_content_legacy(settings, l_stats, r_stats)

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
        spec_sm_phase = smooth_complex(f_lin, total_spec, PHASE_SMOOTH_OCT)
        ph_sm = (np.rad2deg(np.angle(spec_sm_phase)) + 180) % 360 - 180

        spec_sm_gd = smooth_complex(f_lin, total_spec, GD_SMOOTH_OCT)
        gd_sm = calculate_clean_gd(f_lin, spec_sm_gd)
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

        fig = make_subplots(
            rows=5, cols=1, vertical_spacing=0.045,
            subplot_titles=(
                "<b>Magnitude & Alignment</b>",
                "<b>Phase</b>",
                "<b>Group Delay</b>",
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
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"Level reference ({win_label})",
                    line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
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
                thr = 0.35
                bad = np.asarray(c_mask, dtype=float) < float(thr)
                if bad.size == c_freqs.size and bad.size > 8:
                    in_seg = False
                    seg_start = None
                    for fx, is_bad in zip(c_freqs, bad):
                        if is_bad and not in_seg:
                            in_seg = True
                            seg_start = float(fx)
                        elif (not is_bad) and in_seg:
                            in_seg = False
                            seg_end = float(fx)
                            if seg_start is not None and seg_end > seg_start:
                                fig.add_shape(
                                    type="rect", xref="x", yref="y",
                                    x0=seg_start, x1=seg_end,
                                    y0=avg_t-40, y1=avg_t+60,
                                    fillcolor="rgba(255, 0, 0, 0.06)", layer="below", line_width=0,
                                    row=1, col=1
                                )
                    if in_seg and seg_start is not None:
                        seg_end = float(c_freqs[-1])
                        if seg_end > seg_start:
                            fig.add_shape(
                                type="rect", xref="x", yref="y",
                                x0=seg_start, x1=seg_end,
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

        fig.add_trace(go.Scatter(x=f_vis, y=ph_vis, name="Phase", line=dict(color='orange'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_vis, y=gd_vis, name="Group Delay", line=dict(color='orange'), showlegend=False), row=3, col=1)

        # ---- Filter panel: show BOTH baked/exported and compensated views (pro-level clarity) ----
        # Exported (baked): what you actually load into DSP (IR FFT).
        idx_filter_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis, y=filt_vis_export,
                name="Filter dB (exported)",
                line=dict(color='red', width=1.2),
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
                line=dict(color='red', width=1.2, dash="dot"),
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
                        text=f"Auto gain: {ag_txt:+.2f} dB | Headroom: {ah_txt:+.2f} dB",
                       showarrow=False,
                        align="left",
                        font=dict(size=12),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(0,0,0,0.15)",
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
                    bgcolor="white",  #tausta
                    bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1,
                    font=dict(size=12, color="black"),
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
                    font=dict(size=11)
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
                                opacity=0.6,
                                line=dict(width=2),
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
                row=5,
                col=1
            )


        t_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in (1, 2, 3, 4, 5):
            fig.update_xaxes(matches="x", row=r, col=1)
            fig.update_xaxes(type="log", range=[np.log10(2), np.log10(20000)], tickvals=t_vals, row=r, col=1)


        fig.update_yaxes(range=[avg_t-20, avg_t+30], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_yaxes(range=[-30, 12], row=4, col=1)
        if bw_vis is not None and len(bw_vis) > 0:
            bw_lo = max(1.0/96.0, float(np.min(bw_vis)) * 0.9)
            bw_hi = min(1.0/3.0,  float(np.max(bw_vis)) * 1.1)
            if bw_hi - bw_lo < 1e-6:
                bw_lo, bw_hi = (1.0/96.0, 1.0/3.0)
            fig.update_yaxes(range=[bw_lo, bw_hi], row=5, col=1) 
        else:
            fig.update_yaxes(range=[1.0/96.0, 1.0/3.0], row=5, col=1)

        fig.update_yaxes(title_text="oct", row=5, col=1)

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            template="plotly_white",
            title_text=f"{title} Analysis",
            uirevision="keep"
        )
        
        if create_full_html:
            if _plotly_js_path():
                js_mode = "assets/plotly.min.js"
            else:
                js_mode = "cdn"
        else:
            js_mode = True

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
    """Rakentaa tai generoi: generate combined plot mpl."""
    try:
        n_fft = len(filt_ir); f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs); h_filt = scipy.fft.rfft(filt_ir)
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        m_lin = np.interp(f_lin, orig_freqs, orig_mags); p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
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
        ax3.semilogx(f_lin, calculate_clean_gd(f_lin, total_spec), 'orange')
        ax4.semilogx(f_lin, 20*np.log10(np.abs(h_filt)+1e-12), 'r')
        
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xscale('log'); ax.set_xlim(20, 20000); ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"Visualization error ({title}): {e}")
        return b""
    
