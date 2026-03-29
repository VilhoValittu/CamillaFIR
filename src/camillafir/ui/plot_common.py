import logging
import os
import sys

import numpy as np
import scipy.ndimage

from ..dsp.smoothing import apply_smoothing_std, psychoacoustic_smoothing

PHASE_SMOOTH_OCT = 0.1
GD_SMOOTH_OCT = 0.1
GD_SMOOTH_SIGMA = 0.1
logger = logging.getLogger("CamillaFIR")


def _maybe_shift_to_abs(mags_db, avg_t_db):
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
    try:
        f = np.asarray(freqs_hz, dtype=float)
        m = np.asarray(meas_db, dtype=float)
        t = np.asarray(targ_db, dtype=float)
        if f.size < 16 or m.size != f.size or t.size != f.size:
            return m
        f_min = float(f_min_hz)
        f_max = float(f_max_hz)
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
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)


def _plotly_js_path() -> str | None:
    p = _resource_path(os.path.join("assets", "plotly.min.js"))
    return p if os.path.isfile(p) else None


def _confidence_bad_segments(
    freqs,
    conf_mask,
    *,
    thr: float = 0.35,
    min_width_hz: float = 30.0,
    min_gap_hz: float = 35.0,
    max_segments: int = 24,
):
    try:
        f = np.asarray(freqs, dtype=float)
        c = np.asarray(conf_mask, dtype=float)
    except Exception:
        return []
    if f.size != c.size or f.size < 8:
        return []

    valid = np.isfinite(f) & np.isfinite(c) & (f > 0.0)
    if np.count_nonzero(valid) < 8:
        return []
    f = f[valid]
    c = c[valid]
    bad = np.asarray(c < float(thr), dtype=bool)
    if not np.any(bad):
        return []

    raw = []
    in_seg = False
    seg_start = None
    for fx, is_bad in zip(f, bad):
        if is_bad and not in_seg:
            in_seg = True
            seg_start = float(fx)
        elif (not is_bad) and in_seg:
            in_seg = False
            seg_end = float(fx)
            if seg_start is not None and seg_end > seg_start:
                raw.append((float(seg_start), float(seg_end)))
    if in_seg and seg_start is not None and float(f[-1]) > float(seg_start):
        raw.append((float(seg_start), float(f[-1])))

    if not raw:
        return []

    merged = []
    for start, end in raw:
        if not merged:
            merged.append([float(start), float(end)])
            continue
        prev = merged[-1]
        if float(start - prev[1]) <= float(min_gap_hz):
            prev[1] = max(float(prev[1]), float(end))
        else:
            merged.append([float(start), float(end)])

    kept = [
        (float(start), float(end))
        for start, end in merged
        if float(end - start) >= float(min_width_hz)
    ]
    if not kept:
        kept = [(float(start), float(end)) for start, end in merged]

    if len(kept) > int(max_segments):
        kept = sorted(
            kept,
            key=lambda seg: (-(float(seg[1]) - float(seg[0])), float(seg[0])),
        )[: int(max_segments)]
        kept = sorted(kept, key=lambda seg: float(seg[0]))

    return kept


def smooth_complex(freqs, spec, oct_frac=1.0):
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    return real_s + 1j * imag_s


def calculate_clean_gd(freqs, complex_resp, *, sigma: float = GD_SMOOTH_SIGMA):
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sigma_v = float(sigma)
    except Exception:
        sigma_v = float(GD_SMOOTH_SIGMA)
    if sigma_v > 0.0:
        gd_ms = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_v)
    return gd_ms


def remove_ir_peak_delay(freqs, complex_resp, ir, fs):
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        h = np.asarray(complex_resp, dtype=complex).reshape(-1)
        x = np.asarray(ir, dtype=float).reshape(-1)
        fs_v = float(fs)
        if f.size == 0 or h.size != f.size or x.size == 0 or fs_v <= 0.0:
            return h, 0.0
        peak_idx = int(np.argmax(np.abs(x)))
        if peak_idx <= 0:
            return h, 0.0
        delay_s = float(peak_idx) / fs_v
        rot = np.exp(1j * 2.0 * np.pi * f * delay_s)
        return h * rot, delay_s * 1000.0
    except Exception:
        return np.asarray(complex_resp, dtype=complex).reshape(-1), 0.0


def _filter_focus_band(freqs, filt_db, *, delta_db: float = 0.75) -> tuple[float, float] | None:
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        g = np.asarray(filt_db, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(g) & (f > 0.0)
        if np.count_nonzero(valid) < 16:
            return None
        fv = f[valid]
        gv = g[valid]
        order = np.argsort(fv, kind="mergesort")
        fv = fv[order]
        gv = gv[order]
        hi_start = int(max(0, round(0.8 * (fv.size - 1))))
        baseline = float(np.median(gv[hi_start:])) if hi_start < gv.size else float(np.median(gv))
        active = np.abs(gv - baseline) >= float(max(0.1, delta_db))
        if np.count_nonzero(active) < 8:
            return None
        lo = float(fv[np.where(active)[0][0]])
        hi = float(fv[np.where(active)[0][-1]])
        lo = max(float(fv[0]), lo / 1.35)
        hi = min(float(fv[-1]), hi * 1.6)
        if hi <= lo:
            return None
        return lo, hi
    except Exception:
        return None


def _robust_axis_range(
    freqs,
    values,
    *,
    focus_band: tuple[float, float] | None = None,
    q_lo: float = 0.03,
    q_hi: float = 0.97,
    pad_ratio: float = 0.18,
    min_span: float = 10.0,
    max_span: float | None = None,
    include_zero: bool = False,
):
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(y)
        if focus_band is not None:
            lo_f, hi_f = focus_band
            valid &= (f >= float(lo_f)) & (f <= float(hi_f))
        if np.count_nonzero(valid) < 12:
            valid = np.isfinite(y)
        if np.count_nonzero(valid) < 4:
            return None
        yy = y[valid]
        lo = float(np.quantile(yy, float(q_lo)))
        hi = float(np.quantile(yy, float(q_hi)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo = float(np.min(yy))
            hi = float(np.max(yy))
        if hi <= lo:
            mid = float(lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)
        span = float(hi - lo)
        lo -= span * float(pad_ratio)
        hi += span * float(pad_ratio)
        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        span = float(hi - lo)
        if span < float(min_span):
            mid = 0.5 * (hi + lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)
        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        if max_span is not None and (hi - lo) > float(max_span):
            center = 0.0 if include_zero else 0.5 * (hi + lo)
            lo = center - 0.5 * float(max_span)
            hi = center + 0.5 * float(max_span)
        return [float(lo), float(hi)]
    except Exception:
        return None


def _view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
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
