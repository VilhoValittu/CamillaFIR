import numpy as np
import scipy.signal
import scipy.ndimage

def _clamp01(x):
    return np.clip(x, 0.0, 1.0)

def _log_grad(x, f):
    df = np.gradient(f) + 1e-12
    dx_df = np.gradient(x) / df
    return dx_df * np.maximum(f, 1.0)

def _baseline_heavy(freqs, mags_db, sigma_hz=6.0):
    df = np.median(np.diff(freqs[np.isfinite(freqs)])) if len(freqs) > 2 else 1.0
    sigma_bins = max(1.0, float(sigma_hz / max(df, 1e-9)))
    return scipy.ndimage.gaussian_filter1d(mags_db, sigma=sigma_bins)

def _freq_prior(freqs, f1=120.0, f2=200.0):
    p = np.ones_like(freqs, dtype=float)
    p[freqs >= f2] = 0.0
    mid = (freqs >= f1) & (freqs < f2)
    p[mid] = 1.0 - (freqs[mid] - f1) / (f2 - f1)
    return p

def build_bassfirst_masks(freq_axis, m_raw_db, phase_rad_unwrapped, gd_ms, gd_diff,
                          is_wav_source=False, mode_f1=120.0, mode_f2=200.0,
                          rew_asym: bool = False, left_ms: float = 0.0,
                          gd_t0=1.0, gd_t1=6.0,
                          mag_a0=1.5, mag_a1=8.0,
                          q0=2.0, q1=10.0,
                          rough_r0=0.6, rough_r1=2.5,
                          pj_p0=0.0008, pj_p1=0.0040):
    """Rakentaa tai generoi: build bassfirst masks."""

    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(m_raw_db, dtype=float)
    ph = np.asarray(phase_rad_unwrapped, dtype=float)

    if bool(is_wav_source):
        try:
            rough_r0 = float(rough_r0) * 1.5
            rough_r1 = float(rough_r1) * 1.5
            pj_p0 = float(pj_p0) * 2.0
            pj_p1 = float(pj_p1) * 2.0
            gd_t0 = float(gd_t0) * 1.2
            gd_t1 = float(gd_t1) * 1.2
        except (TypeError, ValueError):
            pass

    prior = _freq_prior(f, f1=mode_f1, f2=mode_f2)
    gd_norm = _clamp01((gd_diff - gd_t0) / (gd_t1 - gd_t0 + 1e-12))

    base = _baseline_heavy(f, m, sigma_hz=8.0)
    mag_peak = np.maximum(0.0, m - base)
    mag_norm = _clamp01((mag_peak - mag_a0) / (mag_a1 - mag_a0 + 1e-12))

    q_norm = np.zeros_like(f)
    try:
        peaks, props = scipy.signal.find_peaks(mag_peak, prominence=1.0, distance=max(3, int(0.02*len(f))))
        if len(peaks) > 0:
            results_half = scipy.signal.peak_widths(mag_peak, peaks, rel_height=0.5)
            widths_bins = results_half[0]
            for pi, wb in zip(peaks, widths_bins):
                lo = max(0, int(pi - wb/2))
                hi = min(len(f)-1, int(pi + wb/2))
                bw_hz = max(1e-6, f[hi] - f[lo])
                Q = f[pi] / bw_hz if bw_hz > 0 else 0.0
                q_norm[pi] = _clamp01((Q - q0) / (q1 - q0 + 1e-12))
            q_norm = scipy.ndimage.gaussian_filter1d(q_norm, sigma=6)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    mode_score = prior * (0.45*gd_norm + 0.35*mag_norm + 0.20*q_norm)
    room_mode_mask = scipy.ndimage.gaussian_filter1d(_clamp01(mode_score), sigma=4)

    if bool(is_wav_source):
        mag_only_mode = scipy.ndimage.gaussian_filter1d(_clamp01(mag_norm * prior), sigma=4)
        room_mode_mask = np.maximum(room_mode_mask, mag_only_mode)


    try:
        _rew_asym = bool(rew_asym)
        try:
            _left_ms = float(left_ms)
        except (TypeError, ValueError):
            _left_ms = 0.0
        if not np.isfinite(_left_ms):
            _left_ms = 0.0

        if _rew_asym and (_left_ms < 15.0):
            f1, f2 = 60.0, 80.0
        else:
            f1, f2 = 20.0, 30.0
        taper = np.ones_like(f, dtype=float)
        taper[f <= f1] = 0.0
        mid = (f > f1) & (f < f2)
        taper[mid] = 0.5 - 0.5*np.cos(np.pi * (f[mid] - f1) / (f2 - f1))
        room_mode_mask = room_mode_mask * taper
    except (TypeError, ValueError, FloatingPointError):
        pass

    g = np.abs(_log_grad(m, f))
    rough = scipy.ndimage.gaussian_filter1d(g, sigma=6)
    rough_norm = _clamp01((rough - rough_r0) / (rough_r1 - rough_r0 + 1e-12))

    df = np.gradient(f) + 1e-12
    d1 = np.gradient(ph) / df
    d2 = np.gradient(d1) / df
    pj = scipy.ndimage.gaussian_filter1d(np.abs(d2), sigma=6)
    pj_norm = _clamp01((pj - pj_p0) / (pj_p1 - pj_p0 + 1e-12))

    bad = _clamp01(0.75*rough_norm + 0.25*pj_norm)
    reliability_mask = 1.0 - bad

    dbg = {
        "mag_baseline": base,
        "mag_peak": mag_peak,
        "gd_norm": gd_norm,
        "mag_norm": mag_norm,
        "q_norm": q_norm,
        "rough": rough,
        "pj": pj,
    }
    return reliability_mask, room_mode_mask, dbg

def fuse_conf_for_smoothing(freq_axis, reliability_mask,
                            bass_floor_lo=0.75, bass_floor_hi=0.35,
                            f_lo=80.0, f_hi=200.0):
    """Funktio: fuse conf for smoothing."""
    f = np.asarray(freq_axis, dtype=float)
    rel = np.asarray(reliability_mask, dtype=float)

    floor = np.zeros_like(f) + bass_floor_hi
    floor[f <= f_lo] = bass_floor_lo
    mid = (f > f_lo) & (f < f_hi)
    floor[mid] = bass_floor_lo - (bass_floor_lo - bass_floor_hi) * ((f[mid] - f_lo) / (f_hi - f_lo))

    return np.maximum(rel, floor)

def modulate_gain_bassfirst(gain_db, room_mode_mask,
                            k_mode_cut=0.6, k_mode_boost=0.9):
    """Funktio: modulate gain bassfirst."""
    g = np.asarray(gain_db, dtype=float)
    mm = np.asarray(room_mode_mask, dtype=float)

    out = g.copy()
    cut = out < 0.0
    boost = out > 0.0
    out[cut] = out[cut] * (1.0 + k_mode_cut * mm[cut])
    out[boost] = out[boost] * (1.0 - k_mode_boost * mm[boost])
    return out
