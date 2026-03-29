import numpy as np

from camillafir.common.result_postprocess import (
    _avg_confidence_pct,
    _ensure_scoring_keys,
    _inject_filter_mags_for_ui,
)
from camillafir.workflow.process_support import resolve_ui_stats_fs as _resolve_ui_stats_fs
from camillafir.dsp.camillafir_analysis import (
    _distance_bins_from_hz,
    _sigma_bins_from_hz,
    calculate_group_delay,
)


def test_sigma_bins_falls_back_for_short_axis():
    freq_axis = np.array([0.0, 1.0, 2.0], dtype=float)
    sigma = _sigma_bins_from_hz(freq_axis, sigma_hz=6.0, fallback_bins=7.5)
    assert sigma == 7.5


def test_sigma_bins_clamps_to_minimum_one_bin():
    freq_axis = np.linspace(0.0, 100.0, 101, dtype=float)
    sigma = _sigma_bins_from_hz(freq_axis, sigma_hz=0.2, fallback_bins=3.0)
    assert sigma == 1.0


def test_calculate_group_delay_matches_linear_phase_delay():
    freqs = np.linspace(20.0, 20000.0, 4096, dtype=float)
    delay_ms = 3.0
    phase_deg = -360.0 * freqs * (delay_ms / 1000.0)

    gd_ms = calculate_group_delay(freqs, phase_deg)

    assert np.all(np.isfinite(gd_ms))
    center = gd_ms[200:-200]
    assert abs(float(np.median(center)) - delay_ms) < 0.05


def test_distance_bins_from_hz_scales_with_fft_resolution():
    dense = np.linspace(0.0, 20000.0, 20001, dtype=float)  # 1 Hz/bin
    coarse = np.linspace(0.0, 20000.0, 2001, dtype=float)  # 10 Hz/bin

    d_dense = _distance_bins_from_hz(dense, distance_hz=120.0, fallback_bins=100)
    d_coarse = _distance_bins_from_hz(coarse, distance_hz=120.0, fallback_bins=100)

    assert d_dense == 120
    assert d_coarse == 12


def test_ensure_scoring_keys_accepts_numpy_arrays():
    st = {}
    f = np.array([20.0, 30.0, 40.0, 50.0], dtype=float)
    m = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    out = _ensure_scoring_keys(st, f, m, f, m)

    assert "freq_axis" in out
    assert "measured_mags" in out
    assert "target_mags" in out
    assert "confidence_mask" in out
    assert np.asarray(out["freq_axis"]).size == f.size


def test_inject_filter_mags_accepts_numpy_freq_axis():
    st = {"freq_axis": np.array([20.0, 30.0, 40.0, 50.0], dtype=float)}
    ir = np.hanning(128)

    _inject_filter_mags_for_ui(st, ir, 44100)

    assert "filter_mags" in st
    assert len(st["filter_mags"]) == 4


def test_avg_confidence_pct_accepts_numpy_mask():
    st = {"confidence_mask": np.array([0.2, 0.8], dtype=float)}
    assert _avg_confidence_pct(st) == 50.0


def test_resolve_ui_stats_fs_prefers_dashboard_rate():
    assert _resolve_ui_stats_fs(48000, 44100) == 48000
    assert _resolve_ui_stats_fs(None, 44100) == 44100
    assert _resolve_ui_stats_fs(None, None) == 44100
