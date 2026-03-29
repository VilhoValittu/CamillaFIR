import numpy as np
import logging



logger = logging.getLogger("CamillaFIR.dsp")

AFDW_BW_MIN_OCT = 1.0 / 96.0
AFDW_BW_MAX_OCT = 1.0 / 2.0

def psychoacoustic_smoothing(
    freqs,
    mags,
    *,
    low_bw=1/48.0,
    high_bw=1/1.0,
    f_lo=200.0,
    f_hi=2000.0,
):
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    if f.size < 8 or m.size != f.size:
        return np.copy(m)

    dummy = np.zeros_like(m)
    m_low, _  = apply_smoothing_std(f, m, dummy, float(low_bw))
    m_high, _ = apply_smoothing_std(f, m, dummy, float(high_bw))

    ff = np.maximum(f, 1.0)
    lo = float(max(f_lo, 1.0))
    hi = float(max(f_hi, lo * 1.01))
    w = (np.log10(ff) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    w = np.clip(w, 0.0, 1.0)

    return (1.0 - w) * m_low + w * m_high

def psycho_smooth_safe_gain(freqs, mags):
    """Funktio: psycho smooth safe gain."""
    return psychoacoustic_smoothing(
        freqs, mags,
        low_bw=1/48.0,
        high_bw=1/1.0,
        f_lo=200.0,
        f_hi=2000.0,
    )
    """DSP safety preset for a 'safe' correction reference curve.
    Kept as a named helper for readability, but matches psychoacoustic_smoothing defaults.
    """

def smooth_gain_fractional_octave(freqs, gain_db, filter_smooth, *, mult=1.0):
    """Kasittelee signaalia tai dataa: smooth gain fractional octave."""
    f = np.asarray(freqs, dtype=float)
    g = np.asarray(gain_db, dtype=float)
    if f.size < 8 or g.size != f.size:
        return np.copy(g)

    try:
        fs = float(filter_smooth)
    except (TypeError, ValueError, OverflowError):
        fs = 12.0
    if not np.isfinite(fs) or fs <= 0.0:
        fs = 12.0

    try:
        m = float(mult)
    except (TypeError, ValueError, OverflowError):
        m = 1.0
    if not np.isfinite(m) or m <= 0.0:
        m = 1.0

    octave_fraction = float(np.clip(m / fs, 1.0 / 192.0, 1.0))
    dummy_phase = np.zeros_like(g)
    sm, _ = apply_smoothing_std(f, g, dummy_phase, octave_fraction=octave_fraction)
    return sm


def apply_adaptive_fdw(freqs, mags, confidence_mask, base_cycles=15.0, min_cycles=5.0):
    """Soveltaa tai paivittaa: apply adaptive fdw."""
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    c = np.asarray(confidence_mask, dtype=float) if confidence_mask is not None else None

    if f.size < 8 or m.size != f.size:
        return np.copy(mags)

    if c is None or c.size != f.size:
        c = np.ones_like(f)
    c = np.clip(c, 0.0, 1.0)

    base_cycles = float(base_cycles)
    min_cycles = float(min_cycles)
    if base_cycles < 1.0: base_cycles = 1.0
    if min_cycles < 1.0: min_cycles = 1.0
    if min_cycles > base_cycles:
        min_cycles, base_cycles = base_cycles, min_cycles

    adaptive_cycles = min_cycles + (c * (base_cycles - min_cycles))
    oct_widths = 2.0 / np.maximum(adaptive_cycles, 1.0)

    bw_list = np.array([
        1.0/96.0, 1.0/72.0, 1.0/48.0, 1.0/36.0, 1.0/24.0, 1.0/18.0,
        1.0/12.0, 1.0/9.0, 1.0/6.0, 1.0/4.5, 1.0/3.0, 1.0/2.0,
    ], dtype=float)

    sm_stack = []
    dummy = np.zeros_like(m)
    for bw in bw_list:
        sm, _ = apply_smoothing_std(f, m, dummy, float(bw))
        sm_stack.append(sm)
    sm_stack = np.vstack(sm_stack)

    t = np.clip(oct_widths, AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)

    hi = np.searchsorted(bw_list, t, side='right')
    hi = np.clip(hi, 1, len(bw_list) - 1)
    lo = hi - 1

    bw_lo = bw_list[lo]
    bw_hi = bw_list[hi]
    denom = (bw_hi - bw_lo)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    alpha = (t - bw_lo) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    idx = np.arange(f.size)
    sm_lo = sm_stack[lo, idx]
    sm_hi = sm_stack[hi, idx]

    out = (1.0 - alpha) * sm_lo + alpha * sm_hi


    try:
        if not hasattr(apply_adaptive_fdw, "_dbg_printed"):
            apply_adaptive_fdw._dbg_printed = True

            bw_min = float(np.min(t))
            bw_max = float(np.max(t))
            bw_mean = float(np.mean(t))

            f_min_bw = float(f[np.argmin(t)])
            f_max_bw = float(f[np.argmax(t)])

            logger.info(
                "A-FDW effective BW: "
                f"min={bw_min:.4f} oct @ {f_min_bw:.0f} Hz, "
                f"mean={bw_mean:.4f} oct, "
                f"max={bw_max:.4f} oct @ {f_max_bw:.0f} Hz"
            )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    return out


def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    """Soveltaa tai paivittaa: apply smoothing std."""
    if octave_fraction <= 0: return mags, phases
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    
    
    points_per_octave = 384 
    
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    num_points = max(num_points, 10)
    
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    
    window_size = int(points_per_octave * octave_fraction)
    window_size = max(window_size, 1)
    window = np.hanning(window_size)
    w_sum = window.sum()
    if w_sum > 0:
        window = window / w_sum
    else:
        window = np.ones(window_size) / window_size
    
    pad_len = window_size // 2
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    
    if pad_len > 0:
        sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
        sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
    else:
        sm_mags = np.convolve(m_padded, window, mode='same')
        sm_phases = np.convolve(p_padded, window, mode='same')
        
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))
