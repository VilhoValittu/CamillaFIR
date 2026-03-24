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
    """Phase/IR-vaiheen typed tuotospaketti."""

    impulse: np.ndarray
    gain_db: np.ndarray
    auto_global_gain_db: float
    gain_margin_db: float
    auto_headroom_db: float
    current_peak_gain: float
    final_gain_total: np.ndarray
    residual_telemetry: "ResidualTelemetry | None" = None


@dataclass
class ResidualTelemetry:
    """Typed residual-pass telemetria stats-sovitusta varten."""

    residual_pass_enabled: bool
    residual_strength: float
    residual_smoothing_mult: float
    residual_conf_power: float
    residual_max_delta_db: float
    residual_band_min_hz: float
    residual_band_max_hz: float
    residual_band_edge_hz: float
    residual_band_bins: int
    residual_lf_guard_bins: int
    residual_lf_boost_blocked_bins: int
    residual_right_transition_fade_skipped: bool
