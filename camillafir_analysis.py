# camillafir_analysis.py
import numpy as np
import scipy.signal
import scipy.ndimage


def _sigma_bins_from_hz(freq_axis, sigma_hz: float, fallback_bins: float = 3.0) -> float:
    try:
        f = np.asarray(freq_axis, dtype=float)
        if f.size < 4:
            return float(fallback_bins)
        df = np.median(np.diff(f))
        if not np.isfinite(df) or df <= 0:
            return float(fallback_bins)
        s = float(sigma_hz) / float(df)
        if not np.isfinite(s) or s <= 0:
            return float(fallback_bins)
        return float(max(1.0, s))
    except Exception:
        return float(fallback_bins)


def analyze_acoustic_confidence(freq_axis, complex_meas, fs):
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0

    sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=6.7, fallback_bins=10.0)
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)
    gd_diff = np.abs(gd_ms - gd_smooth)

    threshold_ms = 2.5
    confidence_mask = 1.0 / (1.0 + np.exp(1.5 * (gd_diff - threshold_ms)))
    peaks = np.array([], dtype=int)

    reflection_nodes = []
    valid_idx = np.where(freq_axis > 20)[0]

    if len(valid_idx) > 0:
        peaks, _props = scipy.signal.find_peaks(gd_diff[valid_idx], height=2.0, distance=100)

    raw_nodes = []
    for p in peaks:
        idx = valid_idx[p]
        raw_nodes.append({
            "freq": round(float(freq_axis[idx]), 1),
            "gd_error": round(float(gd_diff[idx]), 2),
            "dist": round(float((gd_diff[idx] / 1000.0 * 343.0) / 2.0), 2),
            "type": "Resonance" if float(freq_axis[idx]) < 200.0 else "Reflection",
        })

    reflection_nodes = sorted(raw_nodes, key=lambda x: x["gd_error"], reverse=True)[:15]
    return confidence_mask, reflection_nodes, gd_ms


def calculate_rt60(impulse, fs):
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return 0.0

        peak_idx = int(np.argmax(np.abs(imp)))
        x = imp[peak_idx:]
        if x.size < int(0.05 * fs):
            return 0.0

        e = x * x

        tail_n = max(int(0.15 * e.size), int(0.05 * fs))
        tail_n = min(tail_n, e.size)
        noise_power = float(np.mean(e[-tail_n:]))

        E = np.cumsum(e[::-1])[::-1]
        E0 = float(E[0]) + 1e-18

        noise_mult = 20.0
        stop_candidates = np.where(E <= noise_mult * noise_power)[0]
        stop_idx = int(stop_candidates[0]) if stop_candidates.size > 0 else (E.size - 1)
        stop_idx = max(stop_idx, 10)

        t = np.arange(E.size) / fs
        edc_db = 10.0 * np.log10((E / E0) + 1e-30)

        smooth_ms = 10.0
        win = max(1, int((smooth_ms / 1000.0) * fs))
        if win > 1:
            kernel = np.ones(win) / win
            edc_db = np.convolve(edc_db, kernel, mode="same")

        t_u = t[:stop_idx + 1]
        d_u = edc_db[:stop_idx + 1]

        def fit_rt(lo_db, hi_db):
            mask = (d_u <= lo_db) & (d_u >= hi_db)
            if np.count_nonzero(mask) < 12:
                return None

            tt = t_u[mask]
            yy = d_u[mask]
            A = np.vstack([tt, np.ones_like(tt)]).T
            a, b = np.linalg.lstsq(A, yy, rcond=None)[0]

            yhat = a * tt + b
            ss_res = float(np.sum((yy - yhat) ** 2))
            ss_tot = float(np.sum((yy - np.mean(yy)) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot

            if a >= -1e-9:
                return None

            rt60 = -60.0 / a
            return rt60, r2

        candidates = []
        r = fit_rt(0.0, -10.0)
        if r:
            candidates.append(("EDT",) + r)
        r = fit_rt(-5.0, -25.0)
        if r:
            candidates.append(("T20",) + r)
        r = fit_rt(-5.0, -35.0)
        if r:
            candidates.append(("T30",) + r)

        if not candidates:
            return 0.0

        pref = {"T30": 0, "T20": 1, "EDT": 2}
        candidates.sort(key=lambda x: (pref[x[0]], -x[2]))

        chosen = None
        for name, rt60, r2 in candidates:
            if r2 >= 0.90:
                chosen = (rt60, r2, name)
                break
        if chosen is None:
            name, rt60, r2 = candidates[0]
            chosen = (rt60, r2, name)

        rt60 = float(chosen[0])
        if 0.05 < rt60 < 5.0:
            return round(rt60, 2)
        return 0.0

    except Exception:
        return 0.0


def _third_oct_centers(f_min=31.5, f_max=8000.0):
    centers = []
    f = float(f_min)
    step = 2 ** (1 / 3)
    while f <= f_max * 1.0001:
        centers.append(float(f))
        f *= step
    return centers


def calculate_rt60_bands(impulse, fs, f_min=31.5, f_max=8000.0, order=4):
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return {}

        nyq = 0.5 * fs
        centers = _third_oct_centers(f_min, min(f_max, nyq * 0.90))
        out = {}

        for fc in centers:
            fl = fc / (2 ** (1 / 6))
            fh = fc * (2 ** (1 / 6))
            fl = max(1.0, fl)
            fh = min(nyq * 0.98, fh)
            if fh <= fl * 1.05:
                continue

            sos = scipy.signal.butter(order, [fl / nyq, fh / nyq], btype="bandpass", output="sos")
            x = scipy.signal.sosfiltfilt(sos, imp)
            rt = calculate_rt60(x, fs)
            if 0.05 < rt < 5.0:
                out[float(round(fc, 2))] = float(rt)
        return out
    except Exception:
        return {}
