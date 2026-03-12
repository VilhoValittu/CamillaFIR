from types import SimpleNamespace

import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter
from camillafir.dsp.camillafir_leveling import compute_leveling, find_stable_level_window


def test_stereo_link_produces_identical_offset(lr_measurements):
    """
    Regression test:
    stereo_link=True must produce identical offsets for identical L/R input.
    """
    (fL, mL, pL), (fR, mR, pR) = lr_measurements

    cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Linear Phase",
        stereo_link=True,
    )

    _, stL = generate_filter(fL, mL, pL, cfg)
    _, stR = generate_filter(fR, mR, pR, cfg)

    offL = float(stL.get("offset_db"))
    offR = float(stR.get("offset_db"))

    assert abs(offL - offR) < 1e-6


def test_without_stereo_link_offsets_present(lr_measurements):
    """
    Sanity check: without stereo_link, offsets still exist.
    """
    (fL, mL, pL), (fR, mR, pR) = lr_measurements

    cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )

    _, stL = generate_filter(fL, mL, pL, cfg)
    _, stR = generate_filter(fR, mR, pR, cfg)

    assert "offset_db" in stL
    assert "offset_db" in stR


def test_compute_leveling_records_forced_window_errors():
    cfg = SimpleNamespace(
        lvl_manual_db=0.0,
        lvl_min=500.0,
        lvl_max=2000.0,
        lvl_mode="Auto",
        lvl_tilt_comp=True,
        lvl_tilt_max_db_per_oct=2.0,
        stereo_link=False,
        lvl_force_window=(1000.0,),  # invalid tuple length => triggers forced-window fallback
        lvl_force_offset_db=None,
        hpf_settings=None,
    )
    freq = np.linspace(20.0, 20000.0, 2048, dtype=float)
    m = np.zeros_like(freq)
    t = np.zeros_like(freq)

    out = compute_leveling(cfg, freq, m, t)

    assert len(out) == 7
    err = getattr(cfg, "_lvl_last_error", None)
    assert isinstance(err, str)
    assert err.startswith("forced_window:")


def test_compute_leveling_uses_log_balanced_median_for_manual_window():
    cfg = SimpleNamespace(
        lvl_manual_db=0.0,
        lvl_min=500.0,
        lvl_max=2000.0,
        lvl_mode="Manual",
        lvl_tilt_comp=False,
        lvl_tilt_max_db_per_oct=2.0,
        stereo_link=False,
        lvl_force_window=None,
        lvl_force_offset_db=None,
        hpf_settings=None,
    )
    freq = np.linspace(500.0, 2000.0, 3001, dtype=float)
    m = np.where(freq < 1100.0, 6.0, 0.0)
    t = np.zeros_like(freq)

    _, calc_offset_db, meas_level_db_window, _, offset_method, _, _ = compute_leveling(cfg, freq, m, t)

    assert offset_method == "ManualMedian"
    assert 5.5 < calc_offset_db < 6.1
    assert 5.5 < meas_level_db_window < 6.1


def test_find_stable_level_window_ignores_narrow_deep_null():
    freq = np.linspace(500.0, 2000.0, 751, dtype=float)
    mags = np.zeros_like(freq)
    narrow_null = (freq >= 740.0) & (freq <= 760.0)
    mags[narrow_null] = -18.0

    upper = freq >= 1000.0
    mags[upper] += 3.5 * np.sin((freq[upper] - 1000.0) / 1000.0 * 6.0 * np.pi)
    target = np.zeros_like(freq)

    w_start, w_end = find_stable_level_window(
        freq,
        mags,
        target,
        500.0,
        2000.0,
        window_size_octaves=1.0,
        hpf_freq=0.0,
    )

    assert w_start < 650.0
    assert w_end < 1200.0


def test_find_stable_level_window_prefers_consistent_offset_window():
    freq = np.linspace(500.0, 2000.0, 2001, dtype=float)
    target = np.zeros_like(freq)
    mags = np.zeros_like(freq)

    # Lower band looks fairly smooth, but offset moves around inside the window.
    mask_lo = (freq >= 520.0) & (freq <= 1050.0)
    f_lo = freq[mask_lo]
    x_lo = (f_lo - f_lo.min()) / max(f_lo.max() - f_lo.min(), 1e-9)
    mags[mask_lo] = (
        0.10 * np.sin(2.0 * np.pi * x_lo)
        + 1.25 * np.exp(-((x_lo - 0.26) / 0.12) ** 2)
        - 1.25 * np.exp(-((x_lo - 0.74) / 0.12) ** 2)
    )

    # Upper band has a bit more ripple, but the level anchor stays consistent.
    mask_hi = (freq >= 980.0) & (freq <= 1980.0)
    f_hi = freq[mask_hi]
    x_hi = (f_hi - f_hi.min()) / max(f_hi.max() - f_hi.min(), 1e-9)
    mags[mask_hi] += 0.30 * np.sin(2.0 * np.pi * x_hi * 3.0)

    w_start, w_end = find_stable_level_window(
        freq,
        mags,
        target,
        500.0,
        2000.0,
        window_size_octaves=1.0,
        hpf_freq=0.0,
    )

    assert w_start > 900.0
    assert w_end > 1800.0
