"""
ir_alignment_check.py — vertaa kahden mittaus-IR:n ajoitusta, napaisuutta,
tasoa sekä vaihetta/ryhmäviivettä XO-pisteessä.

Algoritmit poimittu tests/check_ir_alignment.py:stä.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Apufunktiot
# ---------------------------------------------------------------------------

def _peak_index(x: np.ndarray) -> int:
    return int(np.argmax(np.abs(x)))


def _window_around_peak(
    x: np.ndarray, fs: int, pre_ms: float = 5.0, post_ms: float = 500.0
) -> np.ndarray:
    peak = _peak_index(x)
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    start = max(0, peak - pre)
    end = min(len(x), peak + post + 1)
    return x[start:end]


def _freq_response(
    x: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Palauttaa (freqs_hz, magnitude_db, phase_deg) ikkunoidusta IR:stä."""
    n = max(len(x), 65536)
    n = 1 << (max(1, n) - 1).bit_length()
    X = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(X) + 1e-12)
    phase_deg = np.angle(X, deg=True)
    return freqs, mag_db, phase_deg


def _interp_at(freqs: np.ndarray, vals: np.ndarray, hz: float) -> float:
    return float(np.interp(hz, freqs, vals))


def _group_delay_ms(freqs: np.ndarray, phase_deg: np.ndarray, hz: float) -> float:
    """Ryhmäviive millisekunteina frekvenssipisteessä hz."""
    phase_rad = np.unwrap(np.deg2rad(phase_deg))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    gd_s = -np.diff(phase_rad) / (2.0 * np.pi * df)
    freqs_mid = (freqs[:-1] + freqs[1:]) / 2.0
    return float(np.interp(hz, freqs_mid, gd_s)) * 1000.0


# ---------------------------------------------------------------------------
# Ristikorrelaatiopohjainen ajoitusanalyysi
# ---------------------------------------------------------------------------

def _xcorr_timing(
    fs_a: int, x_a: np.ndarray,
    fs_b: int, x_b: np.ndarray,
    max_lag_ms: float = 150.0,
) -> dict:
    """
    Laskee ristikorrelaation impulssiikkunoiden välillä ja palauttaa
    suhteellisen viiveen (positiivinen = B myöhässä A:sta).
    """
    peak_a = _peak_index(x_a)
    peak_b = _peak_index(x_b)

    pre = int(0.050 * fs_a)
    post = int(0.600 * fs_a)
    seg_a = x_a[max(0, peak_a - pre): min(len(x_a), peak_a + post)].copy()
    seg_b = x_b[max(0, peak_b - pre): min(len(x_b), peak_b + post)].copy()

    seg_a /= (np.max(np.abs(seg_a)) + 1e-12)
    seg_b /= (np.max(np.abs(seg_b)) + 1e-12)

    n = min(len(seg_a), len(seg_b))
    seg_a = seg_a[:n]
    seg_b = seg_b[:n]

    xcorr_at_zero = float(np.dot(seg_a, seg_b))
    xcorr_polarity_flip = xcorr_at_zero < 0
    if xcorr_polarity_flip:
        seg_b = -seg_b

    n_fft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(seg_a, n=n_fft)
    B = np.fft.rfft(seg_b, n=n_fft)
    xcorr = np.fft.irfft(A * np.conj(B), n=n_fft)

    max_lag_samp = min(int(round(max_lag_ms * fs_a / 1000.0)), n_fft // 2 - 1)
    search = np.concatenate([xcorr[:max_lag_samp + 1], xcorr[n_fft - max_lag_samp:]])
    best_idx = int(np.argmax(np.abs(search)))

    if best_idx <= max_lag_samp:
        lag_samples = best_idx
    else:
        lag_samples = -(2 * max_lag_samp - (best_idx - 1))

    offset_ms = lag_samples / float(fs_a) * 1000.0
    confidence = float(np.max(np.abs(search)))

    return {
        "offset_ms": offset_ms,
        "xcorr_confidence": confidence,
        "xcorr_polarity_flip": xcorr_polarity_flip,
        "peaks_same_position": (peak_a == peak_b),
    }


# ---------------------------------------------------------------------------
# Napaisuus, taso, vaihe
# ---------------------------------------------------------------------------

def _polarity_check(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    sign_a = float(np.sign(x_a[_peak_index(x_a)]))
    sign_b = float(np.sign(x_b[_peak_index(x_b)]))
    return {"inverted": sign_a != sign_b}


def _level_check(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    peak_a = float(np.max(np.abs(x_a)))
    peak_b = float(np.max(np.abs(x_b)))
    rms_a = float(np.sqrt(np.mean(x_a ** 2)))
    rms_b = float(np.sqrt(np.mean(x_b ** 2)))
    return {
        "peak_a_dbfs": 20.0 * np.log10(peak_a + 1e-12),
        "peak_b_dbfs": 20.0 * np.log10(peak_b + 1e-12),
        "rms_diff_db": 20.0 * np.log10(rms_b / (rms_a + 1e-12)),
    }


def _phase_check(
    freqs_a: np.ndarray, phase_a: np.ndarray, gd_a_ms: float,
    freqs_b: np.ndarray, phase_b: np.ndarray, gd_b_ms: float,
    xo_hz: float,
) -> dict:
    ph_a = _interp_at(freqs_a, phase_a, xo_hz)
    ph_b = _interp_at(freqs_b, phase_b, xo_hz)
    diff = ((ph_b - ph_a) + 180.0) % 360.0 - 180.0
    gd_diff_ms = gd_b_ms - gd_a_ms
    return {
        "phase_a_deg": ph_a,
        "phase_b_deg": ph_b,
        "phase_diff_deg": diff,
        "gd_a_ms": gd_a_ms,
        "gd_b_ms": gd_b_ms,
        "gd_diff_ms": gd_diff_ms,
        "in_phase": abs(diff) <= 90.0,
        "phase_polarity_inverted": abs(diff) > 150.0,
    }


# ---------------------------------------------------------------------------
# Julkinen entry point
# ---------------------------------------------------------------------------

def run_ir_alignment_check(
    ir_a: np.ndarray,
    fs_a: int,
    ir_b: np.ndarray,
    fs_b: int,
    xo_hz: float = 80.0,
    max_lag_ms: float = 150.0,
) -> dict:
    """
    Vertaa kahta mittaus-IR:ää (A = referenssi, B = vertailtava).

    Palauttaa dictionaryn ir_align_* -avaimilla, tai {} virhetilanteessa.
    """
    try:
        x_a = np.asarray(ir_a, dtype=float).reshape(-1)
        x_b = np.asarray(ir_b, dtype=float).reshape(-1)
        fs_a = int(fs_a)
        fs_b = int(fs_b)
        xo_hz = float(xo_hz)

        if x_a.size < 64 or x_b.size < 64:
            return {}
        if fs_a <= 0 or fs_b <= 0:
            return {}

        timing = _xcorr_timing(fs_a, x_a, fs_b, x_b, max_lag_ms=max_lag_ms)
        polarity = _polarity_check(x_a, x_b)
        level = _level_check(x_a, x_b)

        # Vaihevasteen ikkuna: lyhyt pre riittää vaihe-erolle
        win_a = _window_around_peak(x_a, fs_a)
        win_b = _window_around_peak(x_b, fs_b)
        freqs_a, _, phase_a = _freq_response(win_a, fs_a)
        freqs_b, _, phase_b = _freq_response(win_b, fs_b)

        # Ryhmäviive-ikkuna: pitkä pre (>= 3 jaksoa XO-taajuudella)
        gd_pre_ms = max(50.0, 4.0 / xo_hz * 1000.0)
        gd_win_a = _window_around_peak(x_a, fs_a, pre_ms=gd_pre_ms, post_ms=500.0)
        gd_win_b = _window_around_peak(x_b, fs_b, pre_ms=gd_pre_ms, post_ms=500.0)
        gd_freqs_a, _, gd_phase_a = _freq_response(gd_win_a, fs_a)
        gd_freqs_b, _, gd_phase_b = _freq_response(gd_win_b, fs_b)

        gd_a_ms = _group_delay_ms(gd_freqs_a, gd_phase_a, xo_hz)
        gd_b_ms = _group_delay_ms(gd_freqs_b, gd_phase_b, xo_hz)
        phase = _phase_check(freqs_a, phase_a, gd_a_ms, freqs_b, phase_b, gd_b_ms, xo_hz)

        return {
            "ir_align_xcorr_offset_ms": timing["offset_ms"],
            "ir_align_xcorr_confidence": timing["xcorr_confidence"],
            "ir_align_xcorr_polarity_flip": timing["xcorr_polarity_flip"],
            "ir_align_polarity_inverted": polarity["inverted"],
            "ir_align_level_peak_a_dbfs": level["peak_a_dbfs"],
            "ir_align_level_peak_b_dbfs": level["peak_b_dbfs"],
            "ir_align_level_rms_diff_db": level["rms_diff_db"],
            "ir_align_xo_hz": xo_hz,
            "ir_align_phase_a_deg": phase["phase_a_deg"],
            "ir_align_phase_b_deg": phase["phase_b_deg"],
            "ir_align_phase_diff_deg": phase["phase_diff_deg"],
            "ir_align_phase_in_phase": phase["in_phase"],
            "ir_align_phase_polarity_inverted": phase["phase_polarity_inverted"],
            "ir_align_gd_a_ms": phase["gd_a_ms"],
            "ir_align_gd_b_ms": phase["gd_b_ms"],
            "ir_align_gd_diff_ms": phase["gd_diff_ms"],
        }
    except Exception:
        return {}
