from __future__ import annotations

from typing import Any

import numpy as np

from .gain_policy import clamp_gain_curve, resolve_gain_policy
from .limits import limit_slope_per_octave, limit_slope_per_octave_asym, soft_clip_gain
from .smoothing import smooth_gain_fractional_octave


def _apply_max_boost_cut(err_db: np.ndarray, cfg: Any, max_cut_db: float) -> np.ndarray:
    """Kaare boost/cut-rajoitukselle; hyodyntaa nykyista soft clip -toteutusta."""
    return soft_clip_gain(err_db, cfg.max_boost_db, max_cut_db)


def _apply_slope_limits(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    mask_c: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Soveltaa slope-rajat nykyisella asym/sym-logiikalla."""
    out = np.asarray(err_db, dtype=float).copy()

    max_slope = float(getattr(cfg, "max_slope_db_per_oct", 24.0) or 0.0)
    max_slope_boost = float(getattr(cfg, "max_slope_boost_db_per_oct", 0.0) or 0.0)
    max_slope_cut = float(getattr(cfg, "max_slope_cut_db_per_oct", 0.0) or 0.0)
    if max_slope_boost <= 0.0:
        max_slope_boost = max_slope
    if max_slope_cut <= 0.0:
        max_slope_cut = max_slope

    if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
        g2 = out.copy()
        try:
            if max_slope_boost == max_slope_cut and max_slope_boost > 0:
                g2 = limit_slope_per_octave(freq_axis, g2, max_db_per_oct=float(max_slope_boost))
            else:
                g2 = limit_slope_per_octave_asym(
                    freq_axis,
                    g2,
                    max_db_per_oct_boost=float(max_slope_boost),
                    max_db_per_oct_cut=float(max_slope_cut),
                )
        except Exception:
            pass
        out[mask_c] = g2[mask_c]

    return out, {
        "max_slope": float(max_slope),
        "max_slope_boost": float(max_slope_boost),
        "max_slope_cut": float(max_slope_cut),
    }


def _apply_hard_boost_cut_clamp(
    corr_mag_db: np.ndarray,
    cfg: Any,
    max_cut_db: float,
    *,
    boost_cap_db: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Tekee lopullisen kovan boost/cut-rajauksen.

    Oletuksena kayttaa globaalia `max_boost_db`-rajaa. Jos `boost_cap_db`
    annetaan taajuuspistekohtaisena vektorina, boostin ylaraja voidaan
    nostaa/laskea paikallisesti valitussa `mask`-alueessa.
    """
    policy = resolve_gain_policy(cfg)
    return clamp_gain_curve(
        corr_mag_db,
        policy=policy,
        boost_cap_db=boost_cap_db,
        mask=mask,
    )


def _blend_masked_fractional_octave(
    corr_mag_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    smooth_value: float,
    mix: float,
) -> np.ndarray:
    """Sekoittaa maskialueella smoothatun kayran nykyiseen kayraan."""
    out = np.asarray(corr_mag_db, dtype=float).copy()
    if (not np.any(mask_c)) or (mix <= 0.0):
        return out

    g0 = out.copy()
    idx = np.where(mask_c)[0]
    if idx.size >= 2:
        i0, i1 = int(idx[0]), int(idx[-1])
        if i0 > 0:
            g0[:i0] = g0[i0]
        if i1 < (g0.size - 1):
            g0[i1 + 1:] = g0[i1]

    g_sm = smooth_gain_fractional_octave(freq_axis, g0, smooth_value)
    out[mask_c] = out[mask_c] + (g_sm[mask_c] - out[mask_c]) * float(mix)
    return out
