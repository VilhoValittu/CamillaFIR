from types import SimpleNamespace

import numpy as np

import camillafir.dsp.camillafir_leveling as leveling_module
from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter, generate_filter_pair
from camillafir.dsp.camillafir_leveling import (
    StereoLinkContext,
    compute_leveling,
    find_shared_stereo_level_window,
    find_stable_level_window,
)


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


def test_find_shared_stereo_level_window_prefers_common_compromise():
    freq = np.linspace(500.0, 2000.0, 2001, dtype=float)
    target = np.zeros_like(freq)

    left = np.zeros_like(freq)
    left[freq >= 1000.0] = 2.0 * np.sin((freq[freq >= 1000.0] - 1000.0) / 1000.0 * 12.0 * np.pi)

    right = np.zeros_like(freq)
    right[freq <= 1000.0] = 2.0 * np.sin((freq[freq <= 1000.0] - 500.0) / 500.0 * 12.0 * np.pi)

    left_window = find_stable_level_window(
        freq,
        left,
        target,
        500.0,
        2000.0,
        window_size_octaves=1.0,
        hpf_freq=0.0,
    )
    right_window = find_stable_level_window(
        freq,
        right,
        target,
        500.0,
        2000.0,
        window_size_octaves=1.0,
        hpf_freq=0.0,
    )
    shared_window = find_shared_stereo_level_window(
        freq,
        left,
        target,
        freq,
        right,
        target,
        500.0,
        2000.0,
        window_size_octaves=1.0,
        hpf_freq=0.0,
    )

    assert left_window[0] < 600.0
    assert right_window[0] > 900.0
    assert 650.0 < shared_window[0] < 900.0
    assert 1200.0 < shared_window[1] < 1700.0


def test_find_stable_level_window_reuses_exact_cache(monkeypatch):
    leveling_module._clear_level_window_cache()
    freq = np.linspace(500.0, 2000.0, 2001, dtype=float)
    mags = np.zeros_like(freq)
    target = np.zeros_like(freq)
    calls = {"count": 0}

    def fake_impl(*_args, **_kwargs):
        calls["count"] += 1
        return 640.0, 1280.0

    monkeypatch.setattr(leveling_module, "find_stable_level_window_impl", fake_impl)

    first = find_stable_level_window(freq, mags, target, 500.0, 2000.0)
    second = find_stable_level_window(freq, mags, target, 500.0, 2000.0)

    assert calls["count"] == 1
    assert first == (640.0, 1280.0)
    assert second == first

    leveling_module._clear_level_window_cache()


def test_find_stable_level_window_cache_ignores_irrelevant_params(monkeypatch):
    leveling_module._clear_level_window_cache()
    freq = np.linspace(500.0, 2000.0, 2001, dtype=float)
    mags = np.zeros_like(freq)
    target = np.zeros_like(freq)
    calls = {"count": 0}

    def fake_impl(*_args, **_kwargs):
        calls["count"] += 1
        return 720.0, 1440.0

    monkeypatch.setattr(leveling_module, "find_stable_level_window_impl", fake_impl)

    first = find_stable_level_window(
        freq,
        mags,
        target,
        500.0,
        2000.0,
        tilt_comp=False,
        tilt_max_db_per_oct=2.0,
        perceptual_weighting=False,
        perceptual_strength=0.18,
        perceptual_min_hz=250.0,
        perceptual_max_hz=4000.0,
        perceptual_tie_only=True,
    )
    second = find_stable_level_window(
        freq,
        mags,
        target,
        500.0,
        2000.0,
        tilt_comp=False,
        tilt_max_db_per_oct=24.0,
        perceptual_weighting=False,
        perceptual_strength=0.85,
        perceptual_min_hz=900.0,
        perceptual_max_hz=1800.0,
        perceptual_tie_only=False,
    )

    assert calls["count"] == 1
    assert first == (720.0, 1440.0)
    assert second == first

    leveling_module._clear_level_window_cache()


def test_stereo_link_shared_anchors_to_quieter_channel_without_boost():
    freq = np.linspace(20.0, 20000.0, 2048, dtype=float)
    phase = np.zeros_like(freq)
    left = np.zeros_like(freq)
    right = np.full_like(freq, 6.0)

    cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Linear Phase",
        stereo_link=True,
        stereo_link_strategy="shared",
        mag_c_min=20.0,
        mag_c_max=20000.0,
    )

    _, st_left, _, st_right = generate_filter_pair(freq, left, phase, freq, right, phase, cfg)

    assert st_left.get("stereo_link_level_anchor_channel") == "left"
    assert st_right.get("stereo_link_level_anchor_channel") == "left"
    assert abs(float(st_left.get("target_shift_db", 0.0) or 0.0)) < 1e-6
    assert abs(float(st_right.get("target_shift_db", 0.0) or 0.0)) < 1e-6

    freq_axis = np.asarray(st_left.get("freq_axis", []), dtype=float)
    filt_left = np.asarray(st_left.get("predicted_filter_mags", st_left.get("filter_mags", [])), dtype=float)
    filt_right = np.asarray(st_right.get("predicted_filter_mags", st_right.get("filter_mags", [])), dtype=float)
    mask = (freq_axis >= 200.0) & (freq_axis <= 400.0)

    assert np.any(mask)
    assert abs(float(np.median(filt_left[mask]))) < 0.5
    assert float(np.median(filt_right[mask])) < -4.5


def test_stereo_link_shared_auto_gain_matches_independent_channel_peaks():
    freq = np.linspace(20.0, 20000.0, 2048, dtype=float)
    phase = np.zeros_like(freq)
    left = np.zeros_like(freq)
    right = np.full_like(freq, 6.0)

    single_cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Linear Phase",
        stereo_link=False,
        mag_c_min=20.0,
        mag_c_max=20000.0,
    )
    stereo_cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Linear Phase",
        stereo_link=True,
        stereo_link_strategy="shared",
        mag_c_min=20.0,
        mag_c_max=20000.0,
    )

    _, left_single = generate_filter(freq, left, phase, single_cfg)
    _, right_single = generate_filter(freq, right, phase, single_cfg)
    _, left_linked, _, right_linked = generate_filter_pair(freq, left, phase, freq, right, phase, stereo_cfg)

    expected_shared = min(
        float(left_single.get("auto_global_gain_db", 0.0) or 0.0),
        float(right_single.get("auto_global_gain_db", 0.0) or 0.0),
    )

    assert float(left_linked.get("stereo_link_shared_auto_gain_db", 0.0) or 0.0) == expected_shared
    assert float(right_linked.get("stereo_link_shared_auto_gain_db", 0.0) or 0.0) == expected_shared


def test_compute_leveling_reuses_exact_auto_leveling_cache(monkeypatch):
    leveling_module._clear_leveling_cache()
    cfg = SimpleNamespace(
        lvl_manual_db=0.0,
        lvl_min=500.0,
        lvl_max=2000.0,
        lvl_mode="Auto",
        lvl_tilt_comp=True,
        lvl_tilt_max_db_per_oct=2.0,
        lvl_perceptual_weighting=False,
        lvl_perceptual_strength=0.12,
        lvl_perceptual_min_hz=250.0,
        lvl_perceptual_max_hz=4000.0,
        lvl_perceptual_tie_only=True,
        stereo_link=False,
        lvl_force_window=None,
        lvl_force_offset_db=None,
        hpf_settings=None,
    )
    freq = np.linspace(20.0, 20000.0, 2048, dtype=float)
    target = np.zeros_like(freq)
    measured = 1.5 * np.log2(np.clip(freq, 20.0, None) / 1000.0)
    calls = {"count": 0}

    def fake_find_stable_level_window(*_args, **_kwargs):
        calls["count"] += 1
        return 700.0, 1400.0

    monkeypatch.setattr(leveling_module, "find_stable_level_window", fake_find_stable_level_window)

    first = compute_leveling(cfg, freq, measured, target)
    first_slope = getattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
    first_debug = dict(getattr(cfg, "_lvl_window_debug", {}))

    cfg._lvl_last_error = "stale"
    cfg._lvl_tilt_slope_db_per_oct = 999.0
    cfg._lvl_window_debug = {"ss_min": -1.0, "ss_max": -1.0}

    second = compute_leveling(cfg, freq, measured, target)

    assert calls["count"] == 1
    assert second == first
    assert getattr(cfg, "_lvl_last_error", None) is None
    assert getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) == first_slope
    assert getattr(cfg, "_lvl_window_debug", None) == first_debug

    leveling_module._clear_leveling_cache()


def test_compute_leveling_cache_key_includes_shared_target_level(monkeypatch):
    leveling_module._clear_leveling_cache()
    cfg = SimpleNamespace(
        lvl_manual_db=0.0,
        lvl_min=500.0,
        lvl_max=2000.0,
        lvl_mode="Auto",
        lvl_tilt_comp=False,
        lvl_tilt_max_db_per_oct=2.0,
        lvl_perceptual_weighting=False,
        lvl_perceptual_strength=0.12,
        lvl_perceptual_min_hz=250.0,
        lvl_perceptual_max_hz=4000.0,
        lvl_perceptual_tie_only=True,
        stereo_link=True,
        lvl_force_window=None,
        lvl_force_offset_db=None,
        hpf_settings=None,
    )
    freq = np.linspace(20.0, 20000.0, 2048, dtype=float)
    measured = np.zeros_like(freq)
    target = np.zeros_like(freq)
    calls = {"count": 0}

    def fake_find_stable_level_window(*_args, **_kwargs):
        calls["count"] += 1
        return 800.0, 1600.0

    monkeypatch.setattr(leveling_module, "find_stable_level_window", fake_find_stable_level_window)

    out_low = compute_leveling(
        cfg,
        freq,
        measured,
        target,
        stereo_link_ctx=StereoLinkContext(shared_target_level_db=0.0),
    )
    out_high = compute_leveling(
        cfg,
        freq,
        measured,
        target,
        stereo_link_ctx=StereoLinkContext(shared_target_level_db=6.0),
    )

    assert calls["count"] == 2
    assert out_low[0] != out_high[0]

    leveling_module._clear_leveling_cache()


def test_tilt_fit_piecewise_lf_improves_subwoofer_offset():
    freq = np.geomspace(20.0, 200.0, 240, dtype=float)
    log_f = np.log2(freq)
    log_f0 = float(np.median(log_f))
    true_offset_db = 1.8

    diff = true_offset_db + np.where(
        log_f <= log_f0,
        -0.2 * (log_f - log_f0),
        1.4 * (log_f - log_f0),
    )
    diff -= 5.0 * np.exp(-0.5 * (np.log2(freq / 58.0) / 0.05) ** 2)

    linear_off, linear_slope = leveling_module._tilt_fit_offset_and_slope_db_per_oct(
        freq,
        diff,
        max_db_per_oct=2.0,
        prefer_lf_piecewise_tilt=False,
    )
    piecewise_off, piecewise_slope = leveling_module._tilt_fit_offset_and_slope_db_per_oct(
        freq,
        diff,
        max_db_per_oct=2.0,
        prefer_lf_piecewise_tilt=True,
    )

    assert abs(piecewise_off - true_offset_db) < abs(linear_off - true_offset_db)
    assert abs(piecewise_off - true_offset_db) < 0.2
    assert abs(linear_slope) <= 2.0
    assert abs(piecewise_slope) <= 2.0


def test_leveling_cache_key_includes_auto_goal():
    freq = np.geomspace(20.0, 200.0, 240, dtype=float)
    m = np.zeros_like(freq)
    t = np.zeros_like(freq)
    cfg_a = SimpleNamespace(
        auto_goal="balanced",
        lvl_manual_db=0.0,
        lvl_min=20.0,
        lvl_max=200.0,
        lvl_mode="Auto",
        lvl_tilt_comp=True,
        lvl_tilt_max_db_per_oct=2.0,
        lvl_perceptual_weighting=False,
        lvl_perceptual_strength=0.12,
        lvl_perceptual_min_hz=250.0,
        lvl_perceptual_max_hz=4000.0,
        lvl_perceptual_tie_only=True,
        lvl_force_window=None,
        lvl_force_offset_db=None,
        hpf_settings=None,
    )
    cfg_b = SimpleNamespace(**vars(cfg_a))
    cfg_b.auto_goal = "subwoofers"

    key_a = leveling_module._leveling_cache_key(cfg_a, freq, m, t)
    key_b = leveling_module._leveling_cache_key(cfg_b, freq, m, t)

    assert key_a != key_b
