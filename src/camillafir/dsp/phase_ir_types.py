from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class PhaseIRInputs:
    """Phase/IR-vaiheen syotepaketti siistimpaa funktiorajaa varten."""

    cfg: Any
    freq_axis: np.ndarray
    n_fft: int
    gain_db: np.ndarray
    p_rad_interp: np.ndarray
    conf_mask: np.ndarray | None
    m_anal: np.ndarray
    calc_offset_db: float
    target_mags: np.ndarray
    st: Any
    mask_c: np.ndarray
    base_sigma: Any
    filter_smooth: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    use_bassfirst: bool
    afdw_on: bool
    logger: Any
    apply_hpf_to_mags_fn: Callable[..., np.ndarray]
    limit_gd_gradient_ms_per_oct_fn: Callable[..., np.ndarray]
    cfg_float_allow_zero_fn: Callable[[Any, str, float], float]


@dataclass
class PhaseIROutputs:
    """Phase/IR-vaiheen tuotospaketti; yhteensopiva vanhan dict-API:n kanssa."""

    impulse: np.ndarray
    gain_db: np.ndarray
    auto_global_gain_db: float
    gain_margin_db: float
    auto_headroom_db: float
    current_peak_gain: float
    final_gain_total: np.ndarray

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "impulse": self.impulse,
            "gain_db": self.gain_db,
            "auto_global_gain_db": float(self.auto_global_gain_db),
            "gain_margin_db": float(self.gain_margin_db),
            "auto_headroom_db": float(self.auto_headroom_db),
            "current_peak_gain": float(self.current_peak_gain),
            "final_gain_total": np.asarray(self.final_gain_total, dtype=float),
        }
