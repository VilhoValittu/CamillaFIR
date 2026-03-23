from __future__ import annotations

from typing import Any

import numpy as np


def _select_active_band(freq_axis: np.ndarray, cfg: Any) -> tuple[np.ndarray, tuple[float, float]]:
    """Laskee nykylogiikkaa vastaavan aktiivisen korjauskaistan."""
    fmin = 0.0 if cfg.hpf_settings else float(cfg.mag_c_min)
    fmax = float(cfg.mag_c_max)
    mask = (freq_axis >= fmin) & (freq_axis <= fmax)
    return mask, (fmin, fmax)


def _compute_error_db(meas_mag_db: np.ndarray, target_mag_db: np.ndarray) -> np.ndarray:
    """Laskee tavoitteen ja mittauksen erotuksen (dB)."""
    return np.asarray(target_mag_db, dtype=float) - np.asarray(meas_mag_db, dtype=float)


def _apply_confidence_logic(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Soveltaa nykyisen confidence-painotuksen ilman algoritmimuutosta."""
    if bool(getattr(cfg, "enable_afdw", False)):
        return np.asarray(err_db, dtype=float).copy(), {"mode": "afdw_bypass"}
    eff_conf = np.where(freq_axis < 100, np.maximum(conf_mask, 0.6), conf_mask)
    return (np.asarray(err_db, dtype=float) * eff_conf).copy(), {"mode": "weighted_conf_floor"}


def _error_to_correction_mag(err_db_final: np.ndarray) -> np.ndarray:
    """Muuntaa virhekayran korjausmagnitudiksi (nykytilassa suora lapivienti)."""
    return np.asarray(err_db_final, dtype=float)


def _resolve_filter_smooth(cfg: Any) -> float:
    """Lukee filter smooth -asetuksen nykyisella fallback-logiikalla."""
    try:
        value = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 12)) or 12)
    except Exception:
        value = 12.0
    if not np.isfinite(value) or value <= 0:
        value = 12.0
    return value


def _apply_regularization(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    smoothed_err_db: np.ndarray,
) -> np.ndarray:
    """Sekoittaa raakavirheen ja smoothatun kayran nykyisella reg_strength-kaavalla."""
    return err_db - (err_db - smoothed_err_db) * (cfg.reg_strength / 100.0)
