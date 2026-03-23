from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage

from .phase import combine_mixed_phase
from .phase_ir_ir import _build_complex_spectrum, _ifft_to_ir
from .phase_ir_utils import _max_abs_group_delay_ms, _mixed_excess_weight, _pre_ringing_db, _smoothstep01


@dataclass
class _PhaseComponents:
    raw_u: np.ndarray
    ref_u: np.ndarray
    excess_u: np.ndarray
    min_phase: np.ndarray
    theo_xo: np.ndarray
    conf_mask: np.ndarray | None
    total_mag: np.ndarray
    n_fft: int
    is_mixed: bool
    mixed_split_hz: float
    mixed_transition_hz: float
    use_bassfirst: bool
    afdw_on: bool
    logger: Any
    limit_gd_gradient_ms_per_oct_fn: Any
    low_phase: np.ndarray | None = None
    extra_phase: np.ndarray | None = None
    phase_mask: np.ndarray | None = None


def _unwrap_phases(raw_phase, min_phase) -> tuple[np.ndarray, np.ndarray]:
    raw_u = np.unwrap(np.asarray(raw_phase, dtype=float))
    min_u = np.unwrap(np.asarray(min_phase, dtype=float))
    return raw_u, min_u


def _compute_excess_phase(raw_phase, ref_phase) -> np.ndarray:
    raw = np.asarray(raw_phase, dtype=float)
    ref = np.asarray(ref_phase, dtype=float)
    # Keep excess phase branch-stable: if raw/ref unwrap to different 2*pi branches,
    # direct subtraction can inject large artificial zig-zag into mixed-phase correction.
    return np.angle(np.exp(1j * (raw - ref)))


def _apply_mixed_excess_mask(freq_axis, excess, cfg, st) -> np.ndarray:
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

    strength = float(getattr(cfg, "excess_phase_strength", 0.9) or 0.0)
    strength = float(np.clip(strength, 0.0, 1.0))
    w *= strength

    try:
        if isinstance(st, dict):
            st["mixed_phase_strength"] = float(strength)
            st["mixed_phase_full_correction_hz"] = float(full_hz)
            st["mixed_phase_no_correction_hz"] = float(none_hz)
    except (TypeError, ValueError):
        pass
    return x * w


def _linear_excess_weight(freq_axis: np.ndarray, phase_lim_hz: float) -> np.ndarray:
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
        x1 = (ff[seg1] - f0) / (f1 - f0)
        s1 = _smoothstep01(x1)
        ww[seg1] = w0 + (w1 - w0) * s1
    seg2 = ~seg1
    if np.any(seg2):
        x2 = (ff[seg2] - f1) / (f2 - f1)
        s2 = _smoothstep01(x2)
        ww[seg2] = w1 * (1.0 - s2)
    w[band] = np.clip(ww, 0.0, 1.0)
    return w


def _smooth_linear_boundary(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
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
    if not np.isfinite(sigma_bins):
        sigma_bins = 1.2
    sigma_bins = float(np.clip(sigma_bins, 0.0, 6.0))
    if sigma_bins <= 1e-6:
        return x

    f_start = float(max(30.0, 0.70 * f_lim))
    f_end = float(f_lim)
    if f_end <= (f_start + 1.0):
        return x

    y = scipy.ndimage.gaussian_filter1d(x, sigma=sigma_bins, mode="nearest")
    t = np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0)
    w = _smoothstep01(t)
    out = (1.0 - w) * x + w * y
    try:
        if isinstance(st, dict):
            st["phase_boundary_smooth_enabled"] = True
            st["phase_boundary_smooth_sigma_bins"] = float(sigma_bins)
            st["phase_boundary_smooth_start_hz"] = float(f_start)
            st["phase_boundary_smooth_end_hz"] = float(f_end)
    except (TypeError, ValueError):
        pass
    return out


def _enforce_linear_tail_decay(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
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
    if not np.isfinite(f_start_ratio):
        f_start_ratio = 0.72
    f_start_ratio = float(np.clip(f_start_ratio, 0.50, 0.92))
    f_start = float(max(30.0, f_start_ratio * f_lim))
    sel = np.isfinite(f) & (f >= f_start) & (f <= f_lim)
    idx = np.flatnonzero(sel)
    if idx.size < 8:
        return x

    try:
        sigma_abs = float(getattr(cfg, "phase_tail_abs_smooth_sigma_bins", 2.5) or 2.5)
    except (AttributeError, TypeError, ValueError):
        sigma_abs = 2.5
    if not np.isfinite(sigma_abs):
        sigma_abs = 2.5
    sigma_abs = float(np.clip(sigma_abs, 0.0, 8.0))
    try:
        cosine_strength = float(getattr(cfg, "phase_tail_cosine_strength", 0.85) or 0.85)
    except (AttributeError, TypeError, ValueError):
        cosine_strength = 0.85
    if not np.isfinite(cosine_strength):
        cosine_strength = 0.85
    cosine_strength = float(np.clip(cosine_strength, 0.0, 1.0))

    out = x.copy()
    x_tail = out[idx]
    abs_tail = np.abs(x_tail)
    if sigma_abs > 1e-9:
        abs_tail = scipy.ndimage.gaussian_filter1d(abs_tail, sigma=sigma_abs, mode="nearest")

    # Enforce non-increasing correction magnitude toward phase limit.
    mono = abs_tail.copy()
    for i in range(1, mono.size):
        if mono[i] > mono[i - 1]:
            mono[i] = mono[i - 1]

    # Blend toward a deterministic cosine envelope to avoid narrow spikes/dips.
    if mono.size >= 2 and cosine_strength > 1e-6:
        t = np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)
        cos_env = float(max(mono[0], 0.0)) * (0.5 + 0.5 * np.cos(np.pi * t))
        mono = (1.0 - cosine_strength) * mono + cosine_strength * np.minimum(mono, cos_env)
        mono = np.maximum(mono, 0.0)

    # Keep one sign branch across the whole tail to avoid sign-flip cusp.
    head_n = int(max(3, mono.size // 6))
    sign0 = float(np.sign(np.median(x_tail[:head_n])))
    if sign0 == 0.0:
        sign0 = float(np.sign(x_tail[0]))
    if sign0 == 0.0:
        sign0 = 1.0

    # Additional soft fade to guarantee near-zero at the end bin.
    t = np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)
    fade = 0.5 + 0.5 * np.cos(np.pi * t)
    mono = mono * np.clip(fade, 0.0, 1.0)

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


def _linear_to_minphase_blend_mask(freq_axis: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
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
    if not np.isfinite(start_ratio):
        start_ratio = 0.65
    start_ratio = float(np.clip(start_ratio, 0.25, 0.95))
    f_start = float(max(20.0, start_ratio * f_end))
    if f_end <= (f_start + 1.0):
        m = np.where(f >= f_end, 1.0, 0.0).astype(float)
        return m

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


def _merge_minphase_and_excess(min_u, excess_masked) -> np.ndarray:
    return np.asarray(min_u, dtype=float) + np.asarray(excess_masked, dtype=float)


def _max_abs_gd_gradient_ms_per_oct(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[float, float | None]:
    m = _gd_grad_metrics(freq_axis, phase_rad, mask=mask)
    return float(m["max_ms_per_oct"]), m["at_hz"]


def _gd_grad_metrics(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    out = {
        "max_ms_per_oct": 0.0,
        "at_hz": None,
        "used_x_axis": "log2(f)",
        "df_min": None,
        "df_max": None,
        "phase_wrapped": False,
        "units_note": "gd_ms=-dphi/d(2*pi*f)*1e3; gd_grad=np.gradient(gd_ms, log2(f))",
    }
    try:
        f = np.asarray(freq_axis, dtype=float)
        p_raw = np.asarray(phase_rad, dtype=float)
        if f.size < 16 or p_raw.size != f.size:
            return out

        sel = np.isfinite(f) & np.isfinite(p_raw) & (f > 0.0)
        if mask is not None:
            sel &= np.asarray(mask, dtype=bool)
        if np.count_nonzero(sel) < 4:
            return out

        ff = f[sel]
        pp = np.unwrap(p_raw[sel])
        if ff.size < 4 or not np.all(np.diff(ff) > 0.0):
            return out

        dff = np.diff(ff)
        dff = dff[np.isfinite(dff) & (dff > 0.0)]
        if dff.size:
            out["df_min"] = float(np.min(dff))
            out["df_max"] = float(np.max(dff))

        omega = 2.0 * np.pi * ff
        gd_ms = (-np.gradient(pp, omega)) * 1000.0
        gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)

        log2f = np.log2(np.maximum(ff, 1e-9))
        gd_grad = np.gradient(gd_ms, log2f)
        gd_grad = np.nan_to_num(gd_grad, nan=0.0, posinf=0.0, neginf=0.0)
        if gd_grad.size == 0:
            return out
        idx = int(np.argmax(np.abs(gd_grad)))
        out["max_ms_per_oct"] = float(np.max(np.abs(gd_grad)))
        out["at_hz"] = float(ff[idx]) if ff.size else None
        out["phase_wrapped"] = False
        return out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return out


def _gd_grad_limiter(
    ir,
    cfg,
    st,
    *,
    freq_axis=None,
    phase_mask=None,
    use_bassfirst=False,
    afdw_on=False,
    limiter_fn=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    in_phase = np.asarray(ir, dtype=float).copy()
    out = in_phase.copy()
    info = {
        "enabled": False,
        "applied": False,
        "limit_ms_per_oct": None,
        "reason": "limit<=0",
        "max_grad_before_ms_per_oct": None,
        "max_grad_after_ms_per_oct": None,
        "max_grad_before_hz": None,
        "max_grad_after_hz": None,
        "used_x_axis": "log2(f)",
        "df_min": None,
        "df_max": None,
        "phase_wrapped": False,
        "units_note": "gd_ms=-dphi/d(2*pi*f)*1e3; gd_grad=np.gradient(gd_ms, log2(f))",
        "limit_input": None,
    }
    try:
        lim_cfg = float(
            getattr(
                cfg,
                "gd_grad_limit_ms_per_oct",
                getattr(cfg, "gd_limiter_limit_ms_per_oct", 20.0),
            )
            or 0.0
        )
    except (AttributeError, TypeError, ValueError):
        lim_cfg = 0.0
    if not np.isfinite(lim_cfg):
        lim_cfg = 0.0
    lim_cfg = float(max(0.0, lim_cfg))
    info["limit_input"] = float(lim_cfg)
    info["limit_ms_per_oct"] = float(lim_cfg) if lim_cfg > 0.0 else 0.0
    enable = bool(lim_cfg > 0.0)
    info["enabled"] = enable
    info["reason"] = "applied" if enable else "limit<=0"

    if freq_axis is not None:
        try:
            _before = _gd_grad_metrics(
                np.asarray(freq_axis, dtype=float),
                in_phase,
                mask=phase_mask,
            )
            info["max_grad_before_ms_per_oct"] = float(_before.get("max_ms_per_oct", 0.0) or 0.0)
            info["max_grad_before_hz"] = _before.get("at_hz", None)
            info["used_x_axis"] = str(_before.get("used_x_axis", "log2(f)") or "log2(f)")
            info["df_min"] = _before.get("df_min", None)
            info["df_max"] = _before.get("df_max", None)
            info["phase_wrapped"] = bool(_before.get("phase_wrapped", False))
            info["units_note"] = str(_before.get("units_note", info["units_note"]) or info["units_note"])
        except (TypeError, ValueError, FloatingPointError, IndexError):
            info["max_grad_before_ms_per_oct"] = None
            info["max_grad_before_hz"] = None

    if enable:
        if (limiter_fn is None) or (freq_axis is None):
            info["enabled"] = False
            info["reason"] = "missing data"
        else:
            try:
                f_arr = np.asarray(freq_axis, dtype=float)
                valid_f = np.isfinite(f_arr) & (f_arr > 0.0)
                if phase_mask is not None:
                    valid_f &= np.asarray(phase_mask, dtype=bool)
                if np.count_nonzero(valid_f) >= 4:
                    f_lo = float(np.min(f_arr[valid_f]))
                    f_hi = float(np.max(f_arr[valid_f]))
                else:
                    f_lo = float(np.min(f_arr[np.isfinite(f_arr) & (f_arr > 0.0)]))
                    f_hi = float(np.max(f_arr[np.isfinite(f_arr) & (f_arr > 0.0)]))
                try:
                    gd_sigma = float(getattr(cfg, "gd_grad_smooth_sigma", 0.8) or 0.8)
                except (AttributeError, TypeError, ValueError):
                    gd_sigma = 0.8
                if not np.isfinite(gd_sigma):
                    gd_sigma = 0.8
                out = limiter_fn(
                    f_arr,
                    in_phase,
                    mask=phase_mask,
                    max_grad_ms_per_oct=float(lim_cfg),
                    f_min=float(f_lo),
                    f_max=float(f_hi),
                    grad_smooth_sigma=float(max(0.0, gd_sigma)),
                    soft_limit=True,
                )
                info["applied"] = True
                info["reason"] = "applied"
            except (TypeError, ValueError, FloatingPointError, IndexError):
                info["enabled"] = False
                info["reason"] = "missing data"

    if freq_axis is not None:
        try:
            _after = _gd_grad_metrics(
                np.asarray(freq_axis, dtype=float),
                out,
                mask=phase_mask,
            )
            info["max_grad_after_ms_per_oct"] = float(_after.get("max_ms_per_oct", 0.0) or 0.0)
            info["max_grad_after_hz"] = _after.get("at_hz", None)
        except (TypeError, ValueError, FloatingPointError, IndexError):
            info["max_grad_after_ms_per_oct"] = None
            info["max_grad_after_hz"] = None
    elif info["max_grad_before_ms_per_oct"] is not None:
        info["max_grad_after_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
        info["max_grad_after_hz"] = info["max_grad_before_hz"]

    # Safety: limiter must not worsen GD-gradient metric.
    try:
        gb = info["max_grad_before_ms_per_oct"]
        ga = info["max_grad_after_ms_per_oct"]
        if (
            info.get("applied", False)
            and gb is not None
            and ga is not None
            and np.isfinite(float(gb))
            and np.isfinite(float(ga))
            and (float(ga) > (float(gb) * 1.001))
        ):
            out = in_phase.copy()
            info["applied"] = False
            info["enabled"] = False
            info["reason"] = "missing data"
            info["reverted_non_monotonic"] = True
            info["max_grad_after_ms_per_oct"] = float(gb)
            info["max_grad_after_hz"] = info["max_grad_before_hz"]
    except (TypeError, ValueError, FloatingPointError):
        pass

    try:
        if isinstance(st, dict):
            st["gd_limiter_enabled"] = bool(info["enabled"])
            st["gd_limiter_limit_ms_per_oct"] = info["limit_ms_per_oct"]
            st["gd_limiter_reason"] = str(info["reason"])
            st["gd_limiter_applied"] = bool(info["applied"])
            st["gd_limiter_max_grad_before_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
            st["gd_limiter_max_grad_after_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_limiter_max_grad_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_limiter_max_grad_before_hz"] = info["max_grad_before_hz"]
            st["gd_limiter_max_grad_after_hz"] = info["max_grad_after_hz"]
            st["gd_limiter_max_grad_hz"] = info["max_grad_after_hz"]
            st["gd_grad_limiter_enabled"] = bool(info["enabled"])
            st["gd_grad_limit_ms_per_oct"] = info["limit_ms_per_oct"]
            st["gd_grad_limiter_reason"] = str(info["reason"])
            st["gd_grad_limiter_applied"] = bool(info["applied"])
            st["gd_grad_limiter_max_grad_before_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
            st["gd_grad_limiter_max_grad_after_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_grad_limiter_max_grad_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_grad_limiter_max_grad_before_hz"] = info["max_grad_before_hz"]
            st["gd_grad_limiter_max_grad_after_hz"] = info["max_grad_after_hz"]
            st["gd_grad_limiter_max_grad_hz"] = info["max_grad_after_hz"]

            # Canonical telemetry keys for report/export consumers.
            st["gd_grad_limiter_enabled"] = bool(info["enabled"])
            st["gd_grad_limiter_limit_ms_per_oct"] = float(info["limit_ms_per_oct"] or 0.0)
            st["gd_grad_limiter_before_max_ms_per_oct"] = (
                None if info["max_grad_before_ms_per_oct"] is None else float(info["max_grad_before_ms_per_oct"])
            )
            st["gd_grad_limiter_after_max_ms_per_oct"] = (
                None if info["max_grad_after_ms_per_oct"] is None else float(info["max_grad_after_ms_per_oct"])
            )
            st["gd_grad_limiter_peak_hz"] = (
                None
                if info["max_grad_after_hz"] is None
                else float(info["max_grad_after_hz"])
            )
            st["gd_grad_limiter_reason"] = str(info["reason"])

            # Detailed diagnostics for limiter health checks.
            st["gd_grad_before_max_ms_per_oct"] = (
                None if info["max_grad_before_ms_per_oct"] is None else float(info["max_grad_before_ms_per_oct"])
            )
            st["gd_grad_before_at_hz"] = (
                None if info["max_grad_before_hz"] is None else float(info["max_grad_before_hz"])
            )
            st["gd_grad_after_max_ms_per_oct"] = (
                None if info["max_grad_after_ms_per_oct"] is None else float(info["max_grad_after_ms_per_oct"])
            )
            st["gd_grad_after_at_hz"] = (
                None if info["max_grad_after_hz"] is None else float(info["max_grad_after_hz"])
            )
            st["gd_grad_used_x_axis"] = str(info.get("used_x_axis", "log2(f)") or "log2(f)")
            st["gd_grad_df_min"] = None if info.get("df_min", None) is None else float(info["df_min"])
            st["gd_grad_df_max"] = None if info.get("df_max", None) is None else float(info["df_max"])
            st["gd_grad_phase_wrapped"] = bool(info.get("phase_wrapped", False))
            st["gd_grad_units_note"] = str(info.get("units_note", ""))
            st["gd_grad_limit_input"] = (
                None if info.get("limit_input", None) is None else float(info["limit_input"])
            )
            st["gd_grad_limiter_reverted_non_monotonic"] = bool(info.get("reverted_non_monotonic", False))
    except (TypeError, ValueError):
        pass
    return out, info


def _apply_phase_model(freq_axis, cfg, st, phase_components: _PhaseComponents) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    is_mixed = bool(phase_components.is_mixed)
    min_p = np.asarray(phase_components.min_phase, dtype=float)
    theo_xo = np.asarray(phase_components.theo_xo, dtype=float)
    excess_phase = np.asarray(phase_components.excess_u, dtype=float)
    logger = phase_components.logger

    if bool(getattr(cfg, "phase_safe_2058", False)):
        if "Min" in cfg.filter_type_str:
            final_phase = min_p
        elif is_mixed:
            low_phase = -theo_xo
            phase_components.low_phase = low_phase
            final_phase = low_phase
        else:
            final_phase = -theo_xo
        phase_components.extra_phase = None
        phase_components.phase_mask = None
        return final_phase

    try:
        conf_arr = (
            np.asarray(phase_components.conf_mask, dtype=float)
            if phase_components.conf_mask is not None
            else np.ones_like(f, dtype=float)
        )
        conf_s = scipy.ndimage.gaussian_filter1d(conf_arr, sigma=2)
        conf_s = np.clip(conf_s, 0.0, 1.0)
    except (TypeError, ValueError):
        conf_s = np.ones_like(f, dtype=float)

    phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
    phase_mask = (f > 0) & (f <= phase_lim_hz)
    phase_components.phase_mask = phase_mask
    try:
        if isinstance(st, dict):
            st["phase_limit_hz"] = float(phase_lim_hz)
    except (TypeError, ValueError):
        pass

    try:
        f1 = float(phase_lim_hz)
        f0_fade = f1 / 2.0
        if f0_fade < (f1 - 1.0):
            x = (f - f0_fade) / (f1 - f0_fade + 1e-12)
            x = np.clip(x, 0.0, 1.0)
            w_hi = 0.5 * (1.0 + np.cos(np.pi * x))
            w_hi = np.where(f <= f0_fade, 1.0, w_hi)
            w_hi = np.where(f >= f1, 0.0, w_hi)
        else:
            w_hi = np.ones_like(f, dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        w_hi = np.ones_like(f, dtype=float)

    if is_mixed:
        extra_phase = -_apply_mixed_excess_mask(f, excess_phase, cfg, st)
    else:
        phase_weight = _linear_excess_weight(f, phase_lim_hz)
        phase_weight = phase_weight * phase_mask.astype(float)
        extra_phase = -excess_phase * phase_weight

    try:
        extra_phase *= w_hi
    except (TypeError, ValueError):
        pass

    try:
        conf_floor = 0.10
        conf_power = 1.25
        conf_gain = np.clip(conf_s, 0.0, 1.0) ** conf_power
        conf_gain = conf_floor + (1.0 - conf_floor) * conf_gain
        extra_phase *= conf_gain
    except (TypeError, ValueError):
        pass

    try:
        extra_phase_before = np.asarray(extra_phase, dtype=float).copy()
        if is_mixed:
            clamp_max_deg = float(getattr(cfg, "mixed_phase_budget_lf_deg", 45.0) or 45.0)
            clamp_min_deg = float(getattr(cfg, "mixed_phase_budget_hf_deg", 22.5) or 22.5)
        else:
            clamp_max_deg = 45.0
            clamp_min_deg = 15.0
        if clamp_max_deg < clamp_min_deg:
            clamp_max_deg, clamp_min_deg = clamp_min_deg, clamp_max_deg

        conf_part = np.clip(conf_s, 0.0, 1.0) ** 0.85
        if phase_lim_hz > 0.0:
            freq_rel = np.clip((phase_lim_hz - f) / max(phase_lim_hz, 1e-9), 0.0, 1.0)
        else:
            freq_rel = np.ones_like(f, dtype=float)
        freq_part = np.sqrt(freq_rel)

        blend = 0.70 * conf_part + 0.30 * freq_part
        limit_deg_arr = clamp_min_deg + (clamp_max_deg - clamp_min_deg) * blend
        limit_deg_arr = np.clip(limit_deg_arr, clamp_min_deg, clamp_max_deg)
        limit_rad_arr = np.deg2rad(limit_deg_arr)

        before_rad = float(np.max(np.abs(extra_phase)))
        extra_phase = np.clip(extra_phase, -limit_rad_arr, limit_rad_arr)
        after_rad = float(np.max(np.abs(extra_phase)))

        before_deg = float(np.rad2deg(before_rad))
        after_deg = float(np.rad2deg(after_rad))
        clipped = bool(np.any(np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)))
        try:
            clipped_bins = int(np.sum((np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)) & phase_mask))
        except (TypeError, ValueError, FloatingPointError):
            clipped_bins = int(clipped)
        if clipped:
            msg = (
                "Phase Correction Clamp (adaptive): "
                f"max={before_deg:.1f} deg -> {after_deg:.1f} deg "
                f"(limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg, clipped_bins={clipped_bins})"
            )
        else:
            msg = (
                "Phase Correction Clamp (adaptive): "
                f"max={before_deg:.1f} deg (limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg)"
            )
        logger.info(msg)
        try:
            if isinstance(st, dict):
                st["phase_corr_clamp_deg"] = float(clamp_max_deg)
                st["phase_corr_clamp_min_deg"] = float(clamp_min_deg)
                st["phase_corr_clamp_max_deg"] = float(clamp_max_deg)
                st["phase_corr_clamp_mean_deg"] = (
                    float(np.mean(limit_deg_arr[phase_mask]))
                    if np.any(phase_mask)
                    else float(np.mean(limit_deg_arr))
                )
                st["phase_corr_max_before_deg"] = float(before_deg)
                st["phase_corr_max_after_deg"] = float(after_deg)
                st["phase_corr_clipped"] = bool(clipped)
                st["phase_corr_clipped_bins"] = int(clipped_bins)
                st["phase_corr_clamp_msg"] = str(msg)
        except (TypeError, ValueError):
            pass
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    if is_mixed:
        try:
            max_excess_delay_ms = float(getattr(cfg, "max_excess_delay_ms", 2.5) or 0.0)
        except (AttributeError, TypeError, ValueError):
            max_excess_delay_ms = 0.0
        if np.isfinite(max_excess_delay_ms) and max_excess_delay_ms > 0.0:
            try:
                max_gd_ms = _max_abs_group_delay_ms(f, extra_phase, phase_mask)
                if np.isfinite(max_gd_ms) and max_gd_ms > max_excess_delay_ms:
                    gd_scale = float(np.clip(max_excess_delay_ms / max(max_gd_ms, 1e-9), 0.05, 1.0))
                    extra_phase *= gd_scale
                    logger.info(
                        "Mixed phase excess-delay guard: "
                        f"max|GD|={max_gd_ms:.2f} ms -> target<={max_excess_delay_ms:.2f} ms "
                        f"(scale={gd_scale:.3f})"
                    )
                    try:
                        if isinstance(st, dict):
                            st["mixed_max_excess_delay_ms"] = float(max_excess_delay_ms)
                            st["mixed_excess_delay_before_ms"] = float(max_gd_ms)
                            st["mixed_excess_delay_scale"] = float(gd_scale)
                    except (TypeError, ValueError):
                        pass
            except (TypeError, ValueError, FloatingPointError, IndexError):
                pass

    if is_mixed:
        try:
            max_pre_db = float(getattr(cfg, "max_pre_ringing_db", -35.0) or -35.0)
        except (AttributeError, TypeError, ValueError):
            max_pre_db = -35.0
        if np.isfinite(max_pre_db):
            max_pre_db = float(min(max_pre_db, 0.0))
            extra_guard = np.asarray(extra_phase, dtype=float).copy()
            pre_before_db = None
            pre_after_db = None
            guard_scale_total = 1.0
            h_min = _build_complex_spectrum(phase_components.total_mag, min_p)
            ir_min = _ifft_to_ir(h_min, n=phase_components.n_fft)
            for i in range(3):
                h_lin_guard = _build_complex_spectrum(phase_components.total_mag, _merge_minphase_and_excess(-theo_xo, extra_guard))
                ir_lin_guard = _ifft_to_ir(h_lin_guard, n=phase_components.n_fft)
                ir_mixed_guard = combine_mixed_phase(
                    ir_lin_guard,
                    ir_min,
                    fs=float(cfg.fs),
                    split_freq=phase_components.mixed_split_hz,
                    transition_hz=phase_components.mixed_transition_hz,
                )
                pre_now_db = _pre_ringing_db(ir_mixed_guard)
                if i == 0:
                    pre_before_db = float(pre_now_db)
                pre_after_db = float(pre_now_db)
                if (not np.isfinite(pre_now_db)) or (pre_now_db <= max_pre_db):
                    break
                ratio_now = 10.0 ** (pre_now_db / 10.0)
                ratio_target = 10.0 ** (max_pre_db / 10.0)
                step_scale = float(np.clip(np.sqrt(ratio_target / max(ratio_now, 1e-30)), 0.20, 0.95))
                extra_guard *= step_scale
                guard_scale_total *= step_scale

            if guard_scale_total < 0.999:
                extra_phase = extra_guard
                logger.info(
                    "Mixed phase pre-ringing guard: "
                    f"{pre_before_db:.1f} dB -> {pre_after_db:.1f} dB "
                    f"(limit={max_pre_db:.1f} dB, scale={guard_scale_total:.3f})"
                )
            try:
                if isinstance(st, dict):
                    st["mixed_max_pre_ringing_db"] = float(max_pre_db)
                    st["mixed_pre_ringing_before_db"] = None if pre_before_db is None else float(pre_before_db)
                    st["mixed_pre_ringing_after_db"] = None if pre_after_db is None else float(pre_after_db)
                    st["mixed_pre_ringing_scale"] = float(guard_scale_total)
            except (TypeError, ValueError):
                pass

        try:
            corr_band = phase_mask & (np.abs(excess_phase) > 1e-12)
            if np.any(corr_band):
                eff = np.abs(extra_phase[corr_band]) / np.maximum(np.abs(excess_phase[corr_band]), 1e-12)
                if isinstance(st, dict):
                    st["mixed_phase_eff_strength_mean"] = float(np.mean(eff))
                    st["mixed_phase_eff_strength_max"] = float(np.max(eff))
        except (TypeError, ValueError, FloatingPointError):
            pass

    extra_phase, gd_lim_info = _gd_grad_limiter(
        extra_phase,
        cfg,
        st,
        freq_axis=f,
        phase_mask=phase_mask,
        use_bassfirst=phase_components.use_bassfirst,
        afdw_on=phase_components.afdw_on,
        limiter_fn=phase_components.limit_gd_gradient_ms_per_oct_fn,
    )
    if not is_mixed:
        extra_phase = _smooth_linear_boundary(f, extra_phase, phase_lim_hz, cfg, st)
        extra_phase = _enforce_linear_tail_decay(f, extra_phase, phase_lim_hz, cfg, st)
    try:
        gd_lim_enabled = bool(gd_lim_info.get("enabled", False))
        gd_reason = str(gd_lim_info.get("reason", "unknown"))
        gd_limit = gd_lim_info.get("limit_ms_per_oct", None)
        gd_before = gd_lim_info.get("max_grad_before_ms_per_oct", None)
        gd_after = gd_lim_info.get("max_grad_after_ms_per_oct", None)
        if gd_lim_enabled:
            if gd_limit is None:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
            else:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, limit={float(gd_limit):.2f} ms/oct, "
                    f"max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
        else:
            logger.info(
                "GD gradient limiter: OFF "
                f"(reason={gd_reason}, max|dGD/dOct|={float(gd_before or 0.0):.2f} ms/oct)"
            )
    except (AttributeError, TypeError, ValueError):
        pass

    low_phase = _merge_minphase_and_excess(-theo_xo, extra_phase)
    phase_components.low_phase = low_phase
    phase_components.extra_phase = extra_phase

    if "Min" in cfg.filter_type_str:
        final_phase = min_p
    elif is_mixed:
        final_phase = low_phase
    else:
        sm_mask = _linear_to_minphase_blend_mask(f, phase_lim_hz, cfg, st)
        final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p
    return final_phase
