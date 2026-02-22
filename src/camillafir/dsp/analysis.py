import numpy as np
import scipy.ndimage

from .camillafir_analysis import (
    analyze_acoustic_confidence,
    calculate_rt60,
    calculate_rt60_bands,
    _sigma_bins_from_hz,
    _third_oct_centers,
)

__all__ = [
    "analyze_acoustic_confidence",
    "calculate_rt60",
    "calculate_rt60_bands",
    "_sigma_bins_from_hz",
    "_third_oct_centers",
    "calculate_group_delay",
]

def calculate_group_delay(freqs, phases_deg):
    """Laskee: calculate group delay."""
    phase_rad = np.unwrap(np.deg2rad(phases_deg))
    d_phi_d_f = np.gradient(phase_rad, freqs)
    gd_ms = -d_phi_d_f / (2 * np.pi) * 1000.0
    sigma_bins = _sigma_bins_from_hz(freqs, sigma_hz=2.0, fallback_bins=3.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)
