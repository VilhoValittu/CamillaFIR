from __future__ import annotations

import numpy as np
import scipy.ndimage


def apply_confidence_weighted_target_pull(
    target_db,
    measured_db,
    confidence_mask,
    *,
    conf_floor: float = 0.07,
    conf_ceil: float = 0.95,
    freq_axis=None,
    freq_limit_hz: float | None = 400.0,
    gamma_cut: float = 0.70,
    gamma_boost: float = 1.20,
    return_telemetry: bool = False,
):
    try:
        t = np.asarray(target_db, dtype=float)
        m = np.asarray(measured_db, dtype=float)
        c = np.asarray(confidence_mask, dtype=float) if confidence_mask is not None else None
    except (TypeError, ValueError):
        out = np.asarray(target_db, dtype=float)
        return (out, None) if return_telemetry else out
    if c is None or t.size < 8 or t.shape != m.shape or t.shape != c.shape or conf_ceil <= conf_floor + 1e-9:
        return t
    if freq_limit_hz is not None and freq_axis is not None:
        try:
            f = np.asarray(freq_axis, dtype=float)
        except (TypeError, ValueError):
            f = None
        pull_mask = (f > 0.0) & (f <= float(freq_limit_hz)) if f is not None and f.shape == t.shape else None
    else:
        pull_mask = None
    c = np.clip(c, conf_floor, conf_ceil)
    w = np.clip((c - conf_floor) / (conf_ceil - conf_floor), 0.0, 1.0)
    is_cut = t < m
    gc = float(gamma_cut) if np.isfinite(float(gamma_cut)) and float(gamma_cut) > 0 else 1.0
    gb = float(gamma_boost) if np.isfinite(float(gamma_boost)) and float(gamma_boost) > 0 else 1.0
    w_eff = np.clip(np.where(is_cut, w ** gc, w ** gb), 0.0, 1.0)
    out = (w_eff * t) + ((1.0 - w_eff) * m)
    if pull_mask is not None:
        out = np.where(pull_mask, out, t)
    if not return_telemetry:
        return out
    try:
        pm = np.ones_like(w_eff, dtype=bool) if pull_mask is None else np.asarray(pull_mask, dtype=bool)
        pull_strength = np.clip(1.0 - w_eff, 0.0, 1.0)
        return out, {"w_eff": w_eff, "pull_mask": pm, "pull_strength": pull_strength}
    except (TypeError, ValueError):
        return out, {"w_eff": w_eff, "pull_mask": None, "pull_strength": None}


def _limit_gd_gradient_ms_per_oct(
    freq_axis,
    phase_rad,
    *,
    mask=None,
    max_grad_ms_per_oct=30.0,
    f_min=20.0,
    f_max=250.0,
    grad_smooth_sigma=0.8,
    soft_limit=True,
):
    f = np.asarray(freq_axis, dtype=float)
    ph = np.asarray(phase_rad, dtype=float)
    if f.size < 16 or ph.size != f.size:
        return ph
    m = (np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(f, dtype=bool))
    m = m & np.isfinite(f) & np.isfinite(ph) & (f > 0.0) & (f >= float(f_min)) & (f <= float(f_max))
    if not np.any(m):
        return ph
    idx = np.where(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    ff = f[i0:i1 + 1]
    pp = ph[i0:i1 + 1]
    if ff.size < 16 or not np.all(np.diff(ff) > 0.0):
        return ph
    pp_u = np.unwrap(pp)
    omega = 2.0 * np.pi * ff
    gd_ms = np.nan_to_num((-np.gradient(pp_u, omega)) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    log2f = np.log2(np.maximum(ff, 1e-12))
    base_lim = float(max(0.1, max_grad_ms_per_oct))
    lim_arr = np.where(
        ff < 100.0,
        base_lim * 0.67,
        np.where(ff < 500.0, base_lim, base_lim * 1.5),
    )
    lim_scalar = float(np.min(lim_arr))
    gd_l = gd_ms.copy()
    try:
        sigma = float(grad_smooth_sigma) if float(grad_smooth_sigma) > 0.0 else 0.6
    except (TypeError, ValueError, OverflowError):
        sigma = 0.6
    sigma = float(max(0.25, sigma))
    for _ in range(14):
        gd_grad_now = np.nan_to_num(np.gradient(gd_l, log2f), nan=0.0, posinf=0.0, neginf=0.0)
        if gd_grad_now.size == 0 or float(np.max(np.abs(gd_grad_now / lim_arr))) <= 1.001:
            break
        gd_l = scipy.ndimage.gaussian_filter1d(gd_l, sigma=sigma, mode="nearest")
        sigma *= 1.25
    gd_grad = np.nan_to_num(np.gradient(gd_l, log2f), nan=0.0, posinf=0.0, neginf=0.0)
    gd_grad_l = lim_arr * np.tanh(gd_grad / lim_arr) if soft_limit else np.clip(gd_grad, -lim_arr, lim_arr)
    gd_new = np.empty_like(gd_l)
    gd_new[0] = float(gd_l[0])
    for k in range(1, ff.size):
        gd_new[k] = gd_new[k - 1] + gd_grad_l[k - 1] * (log2f[k] - log2f[k - 1])
    dphi_df = -2.0 * np.pi * (gd_new / 1000.0)
    phi_new = np.empty_like(pp_u)
    phi_new[0] = float(pp_u[0])
    dff = np.diff(ff)
    for k in range(1, ff.size):
        phi_new[k] = phi_new[k - 1] + 0.5 * (dphi_df[k - 1] + dphi_df[k]) * dff[k - 1]
    out = ph.copy()
    out[i0:i1 + 1] = np.nan_to_num(phi_new, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _stage_probe(stage_name, freq_axis, arr_db, mask_c, global_gain_db=0.0, auto_headroom_db=0.0, logger_obj=None):
    try:
        out = {"stage": str(stage_name), "boost_peak_db": 0.0, "cut_peak_db": 0.0, "boost_bins": 0, "cut_bins": 0, "net_boost_peak_db": 0.0}
        if arr_db is None or mask_c is None or not np.any(mask_c):
            return out
        vv = np.asarray(arr_db, dtype=float)[np.asarray(mask_c, dtype=bool)]
        out["boost_peak_db"] = float(np.max(vv)) if vv.size else 0.0
        out["cut_peak_db"] = float(np.min(vv)) if vv.size else 0.0
        out["boost_bins"] = int(np.sum(vv > 1e-6))
        out["cut_bins"] = int(np.sum(vv < -1e-6))
        out["net_boost_peak_db"] = float(out["boost_peak_db"] + float(global_gain_db) + float(auto_headroom_db))
        if logger_obj is not None:
            logger_obj.info(
                f"StageProbe[{out['stage']}]: boost_peak={out['boost_peak_db']:.2f} dB, cut_peak={out['cut_peak_db']:.2f} dB, "
                f"boost_bins={out['boost_bins']}, cut_bins={out['cut_bins']}, net_boost_peak={out['net_boost_peak_db']:.2f} dB"
            )
        return out
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        return {"stage": str(stage_name), "boost_peak_db": 0.0, "cut_peak_db": 0.0, "boost_bins": 0, "cut_bins": 0, "net_boost_peak_db": 0.0}


def apply_hpf_to_mags(freqs, mags, cutoff, order):
    if cutoff <= 0 or order <= 0:
        return mags
    f = np.asarray(freqs, dtype=float)
    if f.size > 1 and f[0] == 0.0:
        f = f.copy()
        f[0] = f[1] if f[1] > 0 else 1e-6
    with np.errstate(divide="ignore"):
        attenuation = -10 * np.log10(1 + (cutoff / (f + 1e-12)) ** (2 * order))
    return mags + attenuation


def apply_lpf_to_mags(freqs, mags, cutoff, order):
    if cutoff <= 0 or order <= 0:
        return mags
    f = np.asarray(freqs, dtype=float)
    if f.size > 1 and f[0] == 0.0:
        f = f.copy()
        f[0] = f[1] if f[1] > 0 else 1e-6
    with np.errstate(divide="ignore"):
        attenuation = -10 * np.log10(1 + (f / (cutoff + 1e-12)) ** (2 * order))
    return mags + attenuation


def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)
