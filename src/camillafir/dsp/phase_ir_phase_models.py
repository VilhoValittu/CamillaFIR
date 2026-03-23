from __future__ import annotations

import numpy as np
import scipy.ndimage

from .phase_ir_utils import _mixed_excess_weight, _smoothstep01


def apply_mixed_excess_mask(freq_axis, excess, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(excess, dtype=float)
    phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0) or 1000.0)
    full_hz = float(getattr(cfg, "low_freq_full_correction_hz", getattr(cfg, "mixed_split_freq", 300.0)) or 300.0)
    none_hz = float(getattr(cfg, "high_freq_no_correction_hz", phase_lim_hz) or phase_lim_hz)
    if phase_lim_hz > 0.0:
        none_hz = min(none_hz, phase_lim_hz)
    if none_hz <= (full_hz + 1.0):
        none_hz = full_hz + 1.0
    w = _mixed_excess_weight(f, full_hz, none_hz)
    if phase_lim_hz > 0.0:
        w *= ((f > 0) & (f <= phase_lim_hz)).astype(float)
    strength = float(np.clip(float(getattr(cfg, "excess_phase_strength", 0.9) or 0.0), 0.0, 1.0))
    w *= strength
    try:
        if isinstance(st, dict):
            st["mixed_phase_strength"] = float(strength)
            st["mixed_phase_full_correction_hz"] = float(full_hz)
            st["mixed_phase_no_correction_hz"] = float(none_hz)
    except (TypeError, ValueError):
        pass
    return x * w


def linear_excess_weight(freq_axis: np.ndarray, phase_lim_hz: float) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    w = np.zeros_like(f, dtype=float)
    if f.size == 0:
        return w
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 20.0):
        return w
    f0 = 20.0
    f2 = float(max(f0 + 1.0, f_lim))
    f1_hi = max(81.0, 0.88 * f2)
    f1 = float(np.clip(0.55 * f2, 80.0, f1_hi))
    if f2 <= (f1 + 1.0):
        f2 = f1 + 1.0
    w0 = 0.30
    w1 = 0.16
    band = np.isfinite(f) & (f > 0.0) & (f <= f2)
    if not np.any(band):
        return w
    ff = f[band]
    ww = np.empty_like(ff, dtype=float)
    seg1 = ff <= f1
    if np.any(seg1):
        ww[seg1] = w0 + (w1 - w0) * _smoothstep01((ff[seg1] - f0) / (f1 - f0))
    seg2 = ~seg1
    if np.any(seg2):
        ww[seg2] = w1 * (1.0 - _smoothstep01((ff[seg2] - f1) / (f2 - f1)))
    w[band] = np.clip(ww, 0.0, 1.0)
    return w


def smooth_linear_boundary(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(extra_phase, dtype=float)
    if f.size < 16 or x.size != f.size:
        return x
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 30.0):
        return x
    try:
        sigma_bins = float(getattr(cfg, "phase_boundary_smooth_sigma_bins", 1.2) or 1.2)
    except (AttributeError, TypeError, ValueError):
        sigma_bins = 1.2
    sigma_bins = float(np.clip(sigma_bins if np.isfinite(sigma_bins) else 1.2, 0.0, 6.0))
    if sigma_bins <= 1e-6:
        return x
    f_start = float(max(30.0, 0.70 * f_lim))
    f_end = float(f_lim)
    if f_end <= (f_start + 1.0):
        return x
    y = scipy.ndimage.gaussian_filter1d(x, sigma=sigma_bins, mode="nearest")
    out = (1.0 - _smoothstep01(np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0))) * x + _smoothstep01(np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0)) * y
    try:
        if isinstance(st, dict):
            st["phase_boundary_smooth_enabled"] = True
            st["phase_boundary_smooth_sigma_bins"] = float(sigma_bins)
            st["phase_boundary_smooth_start_hz"] = float(f_start)
            st["phase_boundary_smooth_end_hz"] = float(f_end)
    except (TypeError, ValueError):
        pass
    return out


def enforce_linear_tail_decay(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(extra_phase, dtype=float)
    if f.size < 16 or x.size != f.size:
        return x
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 30.0):
        return x
    try:
        enabled = bool(getattr(cfg, "phase_tail_monotonic_enable", True))
    except (AttributeError, TypeError, ValueError):
        enabled = True
    if not enabled:
        return x
    try:
        f_start_ratio = float(getattr(cfg, "phase_tail_start_ratio", 0.72) or 0.72)
    except (AttributeError, TypeError, ValueError):
        f_start_ratio = 0.72
    f_start_ratio = float(np.clip(f_start_ratio if np.isfinite(f_start_ratio) else 0.72, 0.50, 0.92))
    f_start = float(max(30.0, f_start_ratio * f_lim))
    idx = np.flatnonzero(np.isfinite(f) & (f >= f_start) & (f <= f_lim))
    if idx.size < 8:
        return x
    try:
        sigma_abs = float(getattr(cfg, "phase_tail_abs_smooth_sigma_bins", 2.5) or 2.5)
    except (AttributeError, TypeError, ValueError):
        sigma_abs = 2.5
    sigma_abs = float(np.clip(sigma_abs if np.isfinite(sigma_abs) else 2.5, 0.0, 8.0))
    try:
        cosine_strength = float(getattr(cfg, "phase_tail_cosine_strength", 0.85) or 0.85)
    except (AttributeError, TypeError, ValueError):
        cosine_strength = 0.85
    cosine_strength = float(np.clip(cosine_strength if np.isfinite(cosine_strength) else 0.85, 0.0, 1.0))
    out = x.copy()
    x_tail = out[idx]
    abs_tail = np.abs(x_tail)
    if sigma_abs > 1e-9:
        abs_tail = scipy.ndimage.gaussian_filter1d(abs_tail, sigma=sigma_abs, mode="nearest")
    mono = abs_tail.copy()
    for i in range(1, mono.size):
        if mono[i] > mono[i - 1]:
            mono[i] = mono[i - 1]
    if mono.size >= 2 and cosine_strength > 1e-6:
        t = np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)
        cos_env = float(max(mono[0], 0.0)) * (0.5 + 0.5 * np.cos(np.pi * t))
        mono = np.maximum((1.0 - cosine_strength) * mono + cosine_strength * np.minimum(mono, cos_env), 0.0)
    head_n = int(max(3, mono.size // 6))
    sign0 = float(np.sign(np.median(x_tail[:head_n]))) or float(np.sign(x_tail[0])) or 1.0
    mono *= np.clip(0.5 + 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)), 0.0, 1.0)
    out[idx] = sign0 * mono
    try:
        if isinstance(st, dict):
            st["phase_tail_monotonic_enabled"] = True
            st["phase_tail_monotonic_start_hz"] = float(f_start)
            st["phase_tail_monotonic_end_hz"] = float(f_lim)
            st["phase_tail_monotonic_sigma_abs_bins"] = float(sigma_abs)
            st["phase_tail_monotonic_start_ratio"] = float(f_start_ratio)
            st["phase_tail_cosine_strength"] = float(cosine_strength)
    except (TypeError, ValueError):
        pass
    return out


def linear_to_minphase_blend_mask(freq_axis: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    m = np.zeros_like(f, dtype=float)
    if f.size == 0:
        return m
    try:
        f_end = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_end = 0.0
    if (not np.isfinite(f_end)) or (f_end <= 20.0):
        return m
    try:
        start_ratio = float(getattr(cfg, "linear_phase_blend_start_ratio", 0.65) or 0.65)
    except (AttributeError, TypeError, ValueError):
        start_ratio = 0.65
    start_ratio = float(np.clip(start_ratio if np.isfinite(start_ratio) else 0.65, 0.25, 0.95))
    f_start = float(max(20.0, start_ratio * f_end))
    if f_end <= (f_start + 1.0):
        return np.where(f >= f_end, 1.0, 0.0).astype(float)
    x = np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0)
    m = _smoothstep01(x)
    m = np.where(f <= f_start, 0.0, m)
    m = np.where(f >= f_end, 1.0, m)
    try:
        if isinstance(st, dict):
            st["linear_phase_blend_start_hz"] = float(f_start)
            st["linear_phase_blend_end_hz"] = float(f_end)
            st["linear_phase_blend_start_ratio"] = float(start_ratio)
    except (TypeError, ValueError):
        pass
    return m


def merge_minphase_and_excess(min_u, excess_masked) -> np.ndarray:
    return np.asarray(min_u, dtype=float) + np.asarray(excess_masked, dtype=float)
