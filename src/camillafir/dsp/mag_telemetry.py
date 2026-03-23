from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _summarize_correction_metrics(
    corr_mag_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    mask_c: np.ndarray,
    logger: Any,
    boost_cand_peak: float,
    n_boost_cand: int,
    n_boost_cand_low: int,
    n_boost_cand_exc: int,
) -> dict[str, Any]:
    """Laskee loppumetriikat ja lokittaa ne yhteen paikkaan."""
    if np.any(mask_c):
        boost_peak_db = float(np.max(corr_mag_db[mask_c]))
        cut_peak_db = float(np.min(corr_mag_db[mask_c]))
        n_boost = int(np.sum((corr_mag_db > 1e-6) & mask_c))
    else:
        boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0
    logger.info(
        "Diagnostic: "
        f"boost_peak={boost_peak_db:.2f} dB, cut_peak={cut_peak_db:.2f} dB, "
        f"boost_bins={n_boost}, "
        f"boost_candidate_peak={float(boost_cand_peak):.2f} dB, "
        f"boost_candidate_bins={int(n_boost_cand)}, "
        f"boost_candidate_bins_lowbass={int(n_boost_cand_low)}, "
        f"boost_candidate_bins_excprot={int(n_boost_cand_exc)}"
    )
    return {
        "boost_peak_db": float(boost_peak_db),
        "cut_peak_db": float(cut_peak_db),
        "n_boost": int(n_boost),
    }


def _log_stage_stats(
    name: str,
    x: np.ndarray,
    mask: np.ndarray | None = None,
    ref: np.ndarray | None = None,
    *,
    logger: Any,
    enabled: bool,
) -> None:
    """Kirjaa vaihekohtaiset dB-tilastot debug-kayttoon."""
    if not enabled:
        return
    try:
        a = np.asarray(x, dtype=float)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape == a.shape and np.any(m):
                v = a[m]
            else:
                v = a
        else:
            m = None
            v = a
        v = v[np.isfinite(v)]
        if v.size < 4:
            return
        v_max = float(np.max(v))
        v_min = float(np.min(v))
        v_rms = float(np.sqrt(np.mean(v * v)))
        msg = f"StageStats: {name}: max={v_max:.3f} dB, min={v_min:.3f} dB, rms={v_rms:.3f} dB"
        if ref is not None:
            r = np.asarray(ref, dtype=float)
            if m is not None and m.shape == r.shape and np.any(m):
                dv = (a - r)[m]
            else:
                dv = (a - r)
            dv = dv[np.isfinite(dv)]
            if dv.size >= 4:
                d_abs_max = float(np.max(np.abs(dv)))
                d_rms = float(np.sqrt(np.mean(dv * dv)))
                msg += f" | Delta_max={d_abs_max:.3f} dB, Delta_rms={d_rms:.3f} dB"
        logger.info(msg)
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass


def _record_stage_probe(
    stage_probes: dict[str, Any],
    key: str,
    probe_fn: Callable[..., Any],
    freq_axis: np.ndarray,
    values: np.ndarray,
    mask_c: np.ndarray,
    cfg: Any,
    logger: Any,
) -> None:
    """Kirjaa stage-probe-tuloksen turvallisesti yhteen apufunktioon."""
    try:
        stage_probes[key] = probe_fn(
            key,
            freq_axis,
            values,
            mask_c,
            global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
            auto_headroom_db=0.0,
            logger_obj=logger,
        )
    except (AttributeError, TypeError, ValueError):
        pass


def _band_delta_metrics(
    a_db: np.ndarray,
    b_db: np.ndarray,
    freq_axis: np.ndarray,
    *,
    f_lo: float = 20.0,
    f_hi: float = 200.0,
) -> tuple[float, float, float | None]:
    """Laskee kahden kayran erotuksen RMS-, huippuarvo- ja huipputaajuusmetriikan annetussa taajuuskaistassa."""
    try:
        a = np.asarray(a_db, dtype=float).reshape(-1)
        b = np.asarray(b_db, dtype=float).reshape(-1)
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        n = int(min(a.size, b.size, f.size))
        if n < 8:
            return 0.0, 0.0, None
        a = a[:n]
        b = b[:n]
        f = f[:n]
        m = (
            np.isfinite(a)
            & np.isfinite(b)
            & np.isfinite(f)
            & (f >= float(f_lo))
            & (f <= float(f_hi))
        )
        if int(np.count_nonzero(m)) < 8:
            return 0.0, 0.0, None
        d = a[m] - b[m]
        ad = np.abs(d)
        idx = int(np.argmax(ad))
        f_band = f[m]
        max_hz = float(f_band[idx]) if f_band.size else None
        return float(np.sqrt(np.mean(d * d))), float(np.max(ad)), max_hz
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0.0, 0.0, None
