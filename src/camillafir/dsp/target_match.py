from __future__ import annotations

import numpy as np

# Legacy house-curve anchor frequencies used by older stats payloads.
_HOUSE_FREQS = np.array(
    [
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        400.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        20000.0,
    ],
    dtype=float,
)


def match_pct_from_rms(
    rms_db: float,
    *,
    m50: float = 3.2,
    slope: float = 0.9,
    cap_rms: float = 0.4,
    cap_match_pct: float = 99.0,
) -> float:
    """Map RMS error (dB) to target-match percentage."""
    try:
        rms = float(rms_db)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not np.isfinite(rms):
        return 0.0
    out = 100.0 / (1.0 + np.exp((rms - float(m50)) / float(slope)))
    out = float(np.clip(out, 0.0, 100.0))
    if rms <= float(cap_rms):
        out = float(cap_match_pct)
    return float(np.clip(out, 0.0, 100.0))


def target_match_from_stats(
    stats: dict | None,
    *,
    include_filter: bool = True,
    use_confidence: bool = True,
    confidence_floor: float = 0.05,
    use_smart_scan_range: bool = True,
    default_range_hz: tuple[float, float] = (200.0, 5000.0),
) -> tuple[float | None, float | None]:
    """
    Single source of truth for target-match from stats payload.

    Returns `(rms_db, match_pct)`. If input is insufficient, returns `(None, None)`.
    """
    st = stats or {}

    def _pick(base_key: str):
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        if mode == "comparison":
            ck = "cmp_" + base_key
            if ck in st and st.get(ck) is not None:
                return st.get(ck)
        return st.get(base_key)

    def _as_arr(v):
        if v is None:
            return None
        try:
            a = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return None
        if a.ndim != 1 or a.size == 0:
            return None
        return a

    def _align_to_freq_axis(values, freq_axis):
        arr = _as_arr(values)
        if arr is None:
            return None
        f = np.asarray(freq_axis, dtype=float)
        if f.ndim != 1 or f.size == 0:
            return arr
        if arr.size == f.size:
            return arr
        if arr.size == _HOUSE_FREQS.size:
            fs = np.maximum(f, 1.0)
            lf_dst = np.log10(fs)
            lf_src = np.log10(_HOUSE_FREQS)
            return np.interp(lf_dst, lf_src, arr, left=arr[0], right=arr[-1])
        x_src = np.linspace(0.0, 1.0, num=arr.size, dtype=float)
        x_dst = np.linspace(0.0, 1.0, num=f.size, dtype=float)
        return np.interp(x_dst, x_src, arr)

    f = _as_arr(_pick("freq_axis"))
    m = _as_arr(_pick("measured_mags"))
    t = _as_arr(_pick("target_mags"))
    if m is None or t is None:
        return None, None

    c = _as_arr(_pick("confidence_mask"))
    g = _as_arr(_pick("filter_mags"))

    if f is not None:
        if m.size != f.size:
            m = _align_to_freq_axis(m, f)
        if t.size != f.size:
            t = _align_to_freq_axis(t, f)
        if include_filter:
            if g is None:
                g = np.zeros_like(m, dtype=float)
            elif g.size != f.size:
                g = _align_to_freq_axis(g, f)
        if c is not None and c.size != f.size:
            c = _align_to_freq_axis(c, f)
    else:
        if include_filter:
            if g is None:
                g = np.zeros_like(m, dtype=float)
            n = min(m.size, t.size, g.size)
            m, t, g = m[:n], t[:n], g[:n]
        else:
            n = min(m.size, t.size)
            m, t = m[:n], t[:n]
        if c is not None:
            c = c[: min(c.size, m.size)]

    if m is None or t is None or m.size == 0 or t.size == 0:
        return None, None

    if include_filter:
        if g is None:
            g = np.zeros_like(m, dtype=float)
        pred = m + g
    else:
        pred = m

    n = min(pred.size, t.size)
    pred = pred[:n]
    t = t[:n]
    if f is not None:
        f = f[:n]
    if c is not None:
        c = c[:n]

    mask = np.isfinite(pred) & np.isfinite(t)
    if use_smart_scan_range and f is not None and f.size == n:
        rng = _pick("smart_scan_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                fmin = float(rng[0])
                fmax = float(rng[1])
            except (TypeError, ValueError, OverflowError):
                fmin, fmax = float(default_range_hz[0]), float(default_range_hz[1])
        else:
            fmin, fmax = float(default_range_hz[0]), float(default_range_hz[1])
        if np.isfinite(fmin) and np.isfinite(fmax) and fmin > 0.0 and fmax > fmin:
            mask &= (f >= fmin) & (f <= fmax)
    if int(np.count_nonzero(mask)) < 10:
        return None, None

    d = (pred - t)[mask]
    if d.size < 2:
        return None, None

    if (
        use_confidence
        and c is not None
        and c.size == n
    ):
        w = np.clip(c[mask], 0.0, 1.0)
        w = np.maximum(w, float(confidence_floor))
        w_sum = float(np.sum(w))
        if np.isfinite(w_sum) and w_sum > 1e-12:
            rms = float(np.sqrt(np.sum(w * d * d) / w_sum))
        else:
            rms = float(np.sqrt(np.mean(d * d)))
    else:
        rms = float(np.sqrt(np.mean(d * d)))

    return rms, match_pct_from_rms(rms)
