import numpy as np


def soft_clip_boost(gain_db, max_boost):
    """Softly clips boosts using tanh function to prevent exceeding max_boost abruptly."""
    if gain_db <= 0:
        return gain_db
    return max_boost * np.tanh(gain_db / max_boost)


def soft_clip_gain(gain_db, max_boost_db, max_cut_db):
    """
    Soft limiter for both boost and cut.
    - boost: +max_boost_db * tanh(g/max_boost_db)
    - cut:   -max_cut_db  * tanh(|g|/max_cut_db)
    """
    g = np.asarray(gain_db, dtype=float)
    out = np.empty_like(g)
    pos = g > 0
    neg = ~pos
    # Boost
    if np.any(pos):
        mb = float(max_boost_db) if max_boost_db > 0 else 0.0
        out[pos] = mb * np.tanh(g[pos] / (mb + 1e-12)) if mb > 0 else 0.0
    # Cut
    if np.any(neg):
        mc = float(max_cut_db) if max_cut_db > 0 else 0.0
        out[neg] = -mc * np.tanh((-g[neg]) / (mc + 1e-12)) if mc > 0 else g[neg]
    return out


def limit_slope_per_octave(freq_axis, gain_db, max_db_per_oct=12.0):
    """
    Limit gain curve change (dB) per octave (log2(f)).
    Performs forward+backward pass to enforce limit in both directions.
    """
    f = np.asarray(freq_axis, dtype=float)
    g = np.asarray(gain_db, dtype=float).copy()
    max_db_per_oct = float(max_db_per_oct)
    if max_db_per_oct <= 0:
        return g

    # Only use f>0 region (log2 doesn't work at f=0)
    idx = np.where(f > 0)[0]
    if idx.size < 3:
        return g

    ii = idx
    x = np.log2(f[ii])
    # Forward: limit rise/fall relative to previous
    for k in range(1, ii.size):
        i = ii[k]
        j = ii[k - 1]
        dx = float(x[k] - x[k - 1])
        if dx <= 0:
            continue
        dg = g[i] - g[j]
        lim = max_db_per_oct * dx
        if dg > lim:
            g[i] = g[j] + lim
        elif dg < -lim:
            g[i] = g[j] - lim

    # Backward: enforce in reverse direction too
    for k in range(ii.size - 2, -1, -1):
        i = ii[k]
        j = ii[k + 1]
        dx = float(x[k + 1] - x[k])
        if dx <= 0:
            continue
        dg = g[i] - g[j]
        lim = max_db_per_oct * dx
        if dg > lim:
            g[i] = g[j] + lim
        elif dg < -lim:
            g[i] = g[j] - lim
    return g


def limit_slope_per_octave_asym(freq_axis, gain_db, max_db_per_oct_boost, max_db_per_oct_cut):
    """
    Asymmetric slope limiter in dB/oct:
      - 'boost' slope limit applies when the curve rises (dg > 0)
      - 'cut' slope limit applies when the curve falls (dg < 0)

    This protects small boost candidates from being flattened to 0 by a too-tight symmetric limiter.
    Backward compatible usage: set both limits equal to old max_slope_db_per_oct.
    """
    f = np.asarray(freq_axis, dtype=float)
    g = np.asarray(gain_db, dtype=float).copy()

    b = float(max_db_per_oct_boost or 0.0)
    c = float(max_db_per_oct_cut or 0.0)
    if b <= 0 and c <= 0:
        return g

    # Work only on valid positive freqs (avoid log2(0))
    idx = np.where(f > 0.0)[0]
    if idx.size < 2:
        return g

    lf = np.log2(f[idx])

    def _limit_step(prev_val, cur_val, dx_oct):
        if dx_oct <= 0:
            return cur_val
        dg = cur_val - prev_val
        lim = (b if dg > 0 else c) * dx_oct
        # if one side is disabled, treat it as "infinite"
        if (dg > 0 and b <= 0) or (dg < 0 and c <= 0):
            return cur_val
        if dg > lim:
            return prev_val + lim
        if dg < -lim:
            return prev_val - lim
        return cur_val

    # Forward pass
    for k in range(1, idx.size):
        i = idx[k]
        j = idx[k - 1]
        dx = float(lf[k] - lf[k - 1])
        g[i] = _limit_step(g[j], g[i], dx)

    # Backward pass (enforce constraint both directions)
    for k in range(idx.size - 2, -1, -1):
        i = idx[k]
        j = idx[k + 1]
        dx = float(lf[k + 1] - lf[k])
        g[i] = _limit_step(g[j], g[i], dx)
    return g


def build_slope_limit_envelope(
    freq_axis,
    target_db,
    *,
    mag_c_min: float,
    mag_c_max: float,
    max_slope_boost_db_per_oct: float,
    max_slope_cut_db_per_oct: float,
):
    """Build a *visual* envelope for slope limiting (dB/oct).

    This does NOT affect filter math. It's meant for the UI plot only:
    show how far the target could move (up/down) while still respecting
    the configured slope limits.

    The envelope is anchored at a pivot frequency inside the correction band.
    """
    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target_db, dtype=float)

    if f.size < 8 or t.size != f.size:
        return None, None, None

    # Disabled / invalid => no envelope
    b = float(max_slope_boost_db_per_oct or 0.0)
    c = float(max_slope_cut_db_per_oct or 0.0)
    if b <= 0.0 and c <= 0.0:
        return None, None, None

    # Correction band sanity
    try:
        cmin = float(mag_c_min or 0.0)
        cmax = float(mag_c_max or 0.0)
    except Exception:
        cmin, cmax = 0.0, 0.0
    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin):
        return None, None, None

    # Pivot: geometric mean of correction band (stable on log axis)
    pivot_hz = float(np.sqrt(cmin * cmax))
    pivot_hz = float(
        np.clip(
            pivot_hz,
            float(np.min(f[f > 0])) if np.any(f > 0) else 1.0,
            float(np.max(f)),
        )
    )
    pivot_idx = int(np.argmin(np.abs(f - pivot_hz)))

    upper_delta = np.zeros_like(t, dtype=float)
    lower_delta = np.zeros_like(t, dtype=float)

    # forward (pivot -> hi)
    for i in range(pivot_idx + 1, f.size):
        f0, f1 = float(f[i - 1]), float(f[i])
        if f0 <= 0 or f1 <= 0:
            upper_delta[i] = upper_delta[i - 1]
            lower_delta[i] = lower_delta[i - 1]
            continue
        dx_oct = float(np.log2(f1 / f0))
        dx_oct = max(dx_oct, 0.0)
        upper_delta[i] = upper_delta[i - 1] + (b * dx_oct if b > 0 else 0.0)
        lower_delta[i] = lower_delta[i - 1] + (c * dx_oct if c > 0 else 0.0)

    # backward (pivot -> lo)
    for i in range(pivot_idx - 1, -1, -1):
        f0, f1 = float(f[i + 1]), float(f[i])
        if f0 <= 0 or f1 <= 0:
            upper_delta[i] = upper_delta[i + 1]
            lower_delta[i] = lower_delta[i + 1]
            continue
        dx_oct = float(np.log2(f0 / f1))  # positive
        dx_oct = max(dx_oct, 0.0)
        upper_delta[i] = upper_delta[i + 1] + (b * dx_oct if b > 0 else 0.0)
        lower_delta[i] = lower_delta[i + 1] + (c * dx_oct if c > 0 else 0.0)

    env_hi = t + upper_delta
    env_lo = t - lower_delta

    # Only show inside correction band; outside return NaN to avoid drawing.
    band = (f >= cmin) & (f <= cmax)
    env_hi = np.where(band, env_hi, np.nan)
    env_lo = np.where(band, env_lo, np.nan)

    return env_lo, env_hi, pivot_hz