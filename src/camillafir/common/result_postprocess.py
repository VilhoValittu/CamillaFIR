from __future__ import annotations

from typing import Any

import numpy as np


def _irwin_tag(mode: Any) -> str:
    try:
        m = str(mode or "auto").strip().lower()
    except Exception:
        m = "auto"
    if m == "rew_sym":
        return "sym"
    if m == "rew_asym":
        return "asym"
    if m in ("auto", "off"):
        return m
    return "auto"


def _shift_zeropad_1d(x: np.ndarray, shift: int) -> np.ndarray:
    arr = np.asarray(x)
    n = int(arr.size)
    s = int(shift)
    if n == 0 or s == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if s > 0:
        if s < n:
            out[s:] = arr[:-s]
    else:
        s = -s
        if s < n:
            out[:-s] = arr[s:]
    return out


def _postpolish_wav_filter_ir(
    ir: np.ndarray,
    fs: int,
    *,
    mag_c_min: float,
    mag_c_max: float,
    trans_width: float,
) -> np.ndarray:
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    fs_i = int(fs) if fs else 0
    if n < 64 or fs_i <= 0:
        return x

    cmin = float(mag_c_min if np.isfinite(mag_c_min) else 0.0)
    cmax = float(mag_c_max if np.isfinite(mag_c_max) else 0.0)
    tw = float(trans_width if np.isfinite(trans_width) else 0.0)
    if cmax <= max(1.0, cmin):
        return x
    if tw <= 0.0:
        tw = max(50.0, 0.4 * cmax)

    f_lo = max(cmin, cmax - 0.95 * tw)
    f_hi = min(float(fs_i) * 0.5, cmax + 1.45 * tw)
    if f_hi <= f_lo:
        return x

    h = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_i))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-14))
    ph = np.angle(h)

    df = float(np.median(np.diff(freqs))) if freqs.size > 2 else 1.0
    sigma_bins = max(2.0, 8.0 / max(df, 1e-9))
    half = int(max(3, round(4.0 * sigma_bins)))
    kx = np.arange(-half, half + 1, dtype=float)
    kk = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kk /= np.sum(kk)
    mag_sm = np.convolve(mag_db, kk, mode="same")

    zone = (freqs >= f_lo) & (freqs <= f_hi)
    if int(np.count_nonzero(zone)) < 8:
        return x

    w = np.zeros_like(freqs, dtype=float)
    span = max(1e-9, float(f_hi - f_lo))
    zz = (freqs[zone] - f_lo) / span
    w[zone] = 0.5 - 0.5 * np.cos(np.pi * np.clip(zz, 0.0, 1.0))

    mix = 0.95
    mag_out = mag_db + (mag_sm - mag_db) * (mix * w)
    h2 = np.power(10.0, mag_out / 20.0) * np.exp(1j * ph)
    y = np.fft.irfft(h2, n=n)

    p0 = float(np.max(np.abs(x)))
    p1 = float(np.max(np.abs(y)))
    if p0 > 0.0 and p1 > 0.0:
        y = y * (p0 / p1)
    return y.astype(float, copy=False)


def _ensure_scoring_keys(st: dict | None, f_in, m_in, hc_f, hc_m):
    try:
        if st is None:
            return st

        f = np.asarray(f_in if f_in is not None else [], dtype=float)
        m = np.asarray(m_in if m_in is not None else [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f if hc_f is not None else [], dtype=float)
                hm = np.asarray(hc_m if hc_m is not None else [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

        if st.get("confidence_mask") is None and f.size > 1:
            st["confidence_mask"] = np.ones_like(f, dtype=float)
        return st
    except Exception:
        return st


def _inject_filter_mags_for_ui(st: dict | None, filt_ir, fs: int):
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        f_src = st.get(key_f, None)
        f_axis = np.asarray(f_src if f_src is not None else [], dtype=float)
        if f_axis.size < 4:
            return

        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 8:
            return

        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return

        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

        f_q = np.clip(f_axis, float(np.min(f_fft)), float(np.max(f_fft)))
        st[key_g] = np.interp(f_q, f_fft, g_db).tolist()
        st[f"{key_g}_source"] = "ir_fft_final"
    except Exception:
        return


def _gd_abs_spread_ms(gd_ms, *, mask=None) -> float | None:
    try:
        gd = np.asarray(gd_ms, dtype=float).reshape(-1)
    except Exception:
        return None
    if gd.size == 0:
        return None

    valid = np.isfinite(gd)
    if mask is not None:
        try:
            band_mask = np.asarray(mask, dtype=bool).reshape(-1)
        except Exception:
            return None
        if band_mask.size != gd.size:
            return None
        valid &= band_mask
    if np.count_nonzero(valid) < 16:
        return None

    gd_abs = np.abs(gd[valid])
    lo = float(np.percentile(gd_abs, 5.0))
    hi = float(np.percentile(gd_abs, 95.0))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return float(max(0.0, hi - lo))


def _inject_filter_gd_stats(st: dict | None, filt_ir, fs: int, lo_hz: float = 20.0, hi_hz: float = 500.0) -> None:
    """Compute delay-compensated absolute-GD spread from the final filter IR and store in st."""
    try:
        if st is None or filt_ir is None:
            return
        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 16:
            return
        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return
        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        phase_raw = np.unwrap(np.angle(h))
        # GD in ms — no delay compensation needed since we measure range (offset cancels)
        gd_ms = np.nan_to_num(-np.gradient(phase_raw, 2.0 * np.pi * np.maximum(f_fft, 1e-9)) * 1000.0,
                               nan=0.0, posinf=0.0, neginf=0.0)
        # 5-95th percentile range in band — robust to spikes, no delay compensation needed
        mask = (f_fft >= float(lo_hz)) & (f_fft <= float(hi_hz)) & np.isfinite(gd_ms)
        gd_range = _gd_abs_spread_ms(gd_ms, mask=mask)
        if gd_range is None:
            st["gd_abs_max_20_500_ms"] = None
        else:
            st["gd_abs_max_20_500_ms"] = float(gd_range)
    except Exception:
        if isinstance(st, dict):
            st["gd_abs_max_20_500_ms"] = None


def _avg_confidence_pct(st: dict) -> float:
    if not st:
        return 0.0
    mode = str(st.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        v = st.get("cmp_avg_confidence", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        cm_src = st.get("cmp_confidence_mask", None)
        cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm_src = st.get("confidence_mask", None)
    cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


__all__ = [
    "_avg_confidence_pct",
    "_ensure_scoring_keys",
    "_inject_filter_gd_stats",
    "_inject_filter_mags_for_ui",
    "_irwin_tag",
    "_postpolish_wav_filter_ir",
    "_shift_zeropad_1d",
]
