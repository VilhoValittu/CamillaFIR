import numpy as np
from .smoothing import apply_smoothing_std

_SYNTH_FREQS = np.array([
    0., 20., 25., 31.5, 40., 50., 63., 80., 100., 125.,
    160., 200., 250., 400., 1000., 2000., 4000., 8000., 16000., 20000.
])
_SYNTH_BASE_MAGS = np.array([
    6., 6., 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5,
    1.4, 0.4, 0., -0.5, -1., -1.8, -2.8, -4., -5.5, -6.
])


def _slope_db_oct(f_hz, mag_db, f_lo=None, f_hi=None) -> float:
    """Least-squares dB/octave slope via polyfit on log10(f)."""
    try:
        ff = np.asarray(f_hz, dtype=float).reshape(-1)
        mm = np.asarray(mag_db, dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    m = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.)
    if f_lo is not None:
        m &= ff >= float(f_lo)
    if f_hi is not None:
        m &= ff <= float(f_hi)
    if int(np.count_nonzero(m)) < 6:
        return float("nan")
    x = np.log10(ff[m])
    y = mm[m]
    if float(np.max(x) - np.min(x)) <= 1e-6:
        return float("nan")
    try:
        p = np.polyfit(x, y, 1)
        return float(p[0] * np.log10(2.))
    except Exception:
        return float("nan")


# Precompute Harman6 slope the same way we will estimate measurements
_SYNTH_HARMAN6_SLOPE_DB_OCT = _slope_db_oct(_SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])


def synthesize_target_from_measurements(
    f_l, m_l, f_r, m_r,
    *,
    bass_comp_frac: float = 0.50,
    bass_comp_ref_db: float = 8.0,
    tilt_comp_frac: float = 0.30,
    hf_comp_frac: float = 0.25,
    smooth_oct: float = 1.0 / 3.0,
    bass_ref_lo_hz: float = 50.0,
    bass_ref_hi_hz: float = 200.0,
    mid_ref_lo_hz: float = 500.0,
    mid_ref_hi_hz: float = 2000.0,
    hf_break_hz: float = 5000.0,
):
    """Synthesize a custom target curve from L/R room measurements.

    Derives a Harman6-based target adjusted for the room's bass buildup,
    broadband tilt, and HF roll-off.

    Returns (freq_array, mag_array) on the standard 20-point grid,
    or None if inputs are insufficient.
    """
    def _to_arr(x):
        try:
            return np.asarray(x, dtype=float).reshape(-1)
        except Exception:
            return np.array([], dtype=float)

    fl, ml = _to_arr(f_l), _to_arr(m_l)
    fr, mr = _to_arr(f_r), _to_arr(m_r)

    l_ok = bool(fl.size >= 32 and ml.size == fl.size)
    r_ok = bool(fr.size >= 32 and mr.size == fr.size)
    if not l_ok and not r_ok:
        return None
    if not l_ok:
        fl, ml = fr.copy(), mr.copy()
    if not r_ok:
        fr, mr = fl.copy(), ml.copy()

    def _sort_finite(f, m):
        idx = np.argsort(f)
        ff, mm = f[idx], m[idx]
        mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.)
        return ff[mask], mm[mask]

    fl, ml = _sort_finite(fl, ml)
    fr, mr = _sort_finite(fr, mr)
    if fl.size < 32 or fr.size < 32:
        return None

    # Common frequency range
    f_lo = max(20., float(np.min(fl)), float(np.min(fr)))
    f_hi = min(20000., float(np.max(fl)), float(np.max(fr)))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo * 1.5:
        return None

    # 320-pt log grid, average L+R in dB
    fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
    m_avg = 0.5 * (np.interp(fg, fl, ml) + np.interp(fg, fr, mr))

    # 1/3-octave smooth → macro spectral envelope
    try:
        m_sm, _ = apply_smoothing_std(fg, m_avg, np.zeros_like(m_avg), float(smooth_oct))
        m_sm = np.asarray(m_sm, dtype=float)
    except Exception:
        m_sm = m_avg.copy()

    # Band averages
    def _band_mean(arr, flo, fhi):
        mask = np.isfinite(arr) & (fg >= float(flo)) & (fg <= float(fhi))
        return float(np.mean(arr[mask])) if int(np.count_nonzero(mask)) >= 4 else float("nan")

    bass_avg = _band_mean(m_sm, bass_ref_lo_hz, bass_ref_hi_hz)
    mid_avg = _band_mean(m_sm, mid_ref_lo_hz, mid_ref_hi_hz)
    bass_excess_db = float(bass_avg - mid_avg) if (np.isfinite(bass_avg) and np.isfinite(mid_avg)) else 0.0

    # Slope estimates
    meas_slope = _slope_db_oct(fg, m_sm)
    hf_slope = _slope_db_oct(fg, m_sm, f_lo=float(hf_break_hz))

    # Working grid: standard 20-pt anchors excluding the 0 Hz sentinel
    f_work = _SYNTH_FREQS[1:].copy()   # 19 points: 20..20000 Hz
    m_work = _SYNTH_BASE_MAGS[1:].copy()

    # --- Bass shelf adjustment ---
    # tanh-limited: if room adds X dB bass, target gets ~X/2 less bass boost
    bass_adj = (
        -float(bass_comp_frac)
        * np.tanh(bass_excess_db / float(bass_comp_ref_db))
        * float(bass_comp_ref_db)
    )
    if np.isfinite(bass_adj) and abs(bass_adj) > 1e-4:
        log_lo = np.log10(max(float(bass_ref_lo_hz), 1.))
        log_hi = np.log10(400.)
        log_f = np.log10(np.maximum(f_work, 1.))
        shelf_w = np.where(
            f_work <= float(bass_ref_lo_hz), 1.0,
            np.where(
                f_work >= 400., 0.0,
                1.0 - (log_f - log_lo) / (log_hi - log_lo),
            ),
        )
        shelf_w = np.clip(shelf_w, 0., 1.)
        m_work += float(bass_adj) * shelf_w

    # --- Broadband tilt adjustment (pivot at 1 kHz) ---
    if np.isfinite(meas_slope) and np.isfinite(_SYNTH_HARMAN6_SLOPE_DB_OCT):
        tilt_adj_per_oct = float(tilt_comp_frac) * float(meas_slope - _SYNTH_HARMAN6_SLOPE_DB_OCT)
        if abs(tilt_adj_per_oct) > 1e-4:
            log2_f_over_1k = np.log2(np.maximum(f_work, 1.) / 1000.)
            m_work += tilt_adj_per_oct * log2_f_over_1k

    # --- HF roll-off accommodation (above hf_break_hz) ---
    harman6_hf_slope = _slope_db_oct(_SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:], f_lo=float(hf_break_hz))
    if np.isfinite(hf_slope) and np.isfinite(harman6_hf_slope) and hf_slope < harman6_hf_slope - 0.5:
        hf_excess_slope = float(hf_slope - harman6_hf_slope)  # negative
        hf_adj_per_oct = float(hf_comp_frac) * hf_excess_slope  # negative: lowers target HF
        if abs(hf_adj_per_oct) > 1e-4:
            mask_hf = f_work > float(hf_break_hz)
            log2_f_over_break = np.log2(np.maximum(f_work[mask_hf], 1.) / float(hf_break_hz))
            m_work[mask_hf] += hf_adj_per_oct * log2_f_over_break

    # Clip and assemble output
    m_work = np.clip(m_work, -12., 14.)
    out_freqs = _SYNTH_FREQS.copy()
    out_mags = np.empty_like(out_freqs)
    out_mags[0] = float(m_work[0])   # 0 Hz sentinel = same as 20 Hz value
    out_mags[1:] = m_work

    return out_freqs, out_mags
