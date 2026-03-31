from types import SimpleNamespace

import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter
from camillafir.dsp.camillafir_leveling import StereoLinkContext
from camillafir.dsp import correction_baseline
from camillafir.dsp.correction_baseline import apply_null_guard_target
from camillafir.dsp.dsp_preprocess import run_preprocess
from camillafir.dsp.tdc import apply_smart_tdc


def test_generate_filter_smoke(lr_measurements):
    """
    DSP smoke test: real FR input -> impulse + stats.
    """
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )

    imp, st = generate_filter(f, m, p, cfg)

    assert imp is not None
    assert len(imp) > 0
    assert len(imp) == cfg.num_taps
    assert isinstance(st, dict)
    assert "offset_db" in st


def test_run_preprocess_preserves_requested_even_fft_size(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    prep = run_preprocess(f, m, p, cfg)

    assert int(prep.ctx.n_fft) == int(cfg.num_taps)
    assert np.array_equal(
        prep.ctx.freq_axis,
        np.fft.rfftfreq(int(cfg.num_taps), d=1.0 / float(cfg.fs)),
    )


def test_run_preprocess_presolve_skips_comparison_and_plot_smoothing(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=True,
    )
    cfg.plot_smoothing_level = "psy"

    prep = run_preprocess(f, m, p, cfg, presolve_mode=True)

    assert prep.cmp is None
    assert prep.analysis_mode == "native"
    assert prep.m_plot_db is None


def test_apply_null_guard_target_matches_reference_convolution():
    freq_axis = np.linspace(20.0, 20000.0, 4096, dtype=float)
    target = np.zeros_like(freq_axis)
    measured = np.zeros_like(freq_axis)
    null_band = (freq_axis >= 1800.0) & (freq_axis <= 2200.0)
    measured[null_band] = -12.0

    out = apply_null_guard_target(
        freq_axis,
        target,
        measured,
        mag_c_min=20.0,
        mag_c_max=20000.0,
        enable=True,
        depth_db=12.0,
        max_blend=0.85,
        max_total_relax_db=12.0,
        smooth_oct=0.18,
    )

    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target, dtype=float)
    m = np.asarray(measured, dtype=float)
    band = (f >= 20.0) & (f <= 20000.0)
    lf = np.log2(np.clip(f, 1e-3, None))
    dlf = np.diff(lf[band])
    dlf = dlf[np.isfinite(dlf) & (dlf > 0)]
    step = float(np.median(dlf))
    sigma_bins = max(1.0, float(0.18) / max(step, 1e-6))
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= np.sum(k)
    env = m.copy()
    env[band] = np.convolve(env[band], k, mode="same")
    dip = np.where(band, env - m, 0.0)
    w = np.clip((dip - 12.0) / 12.0, 0.0, 1.0) ** 1.4
    w2 = np.zeros_like(w)
    w2[band] = np.clip(np.convolve(w[band], k, mode="same"), 0.0, 1.0) * 0.85
    ref = np.maximum(((1.0 - w2) * t) + (w2 * m), t - 12.0)

    assert np.allclose(out, ref, atol=1e-9, rtol=1e-9)


def test_prepare_correction_baseline_skips_second_leveling_pass_after_uniform_target_shift(monkeypatch):
    freq_axis = np.linspace(20.0, 20000.0, 2048, dtype=float)
    measured = np.zeros_like(freq_axis)
    gain_db = np.zeros_like(freq_axis)
    cfg = SimpleNamespace(
        fs=44100,
        num_taps=65536,
        house_freqs=None,
        house_mags=None,
        hpf_settings=None,
        enable_tdc=False,
        enable_null_guard=False,
        lvl_min=500.0,
        lvl_max=2000.0,
        comparison_mode=False,
        lvl_force_offset_db=None,
    )

    correction_baseline._clear_rt60_cache()
    rt60_key = correction_baseline._rt60_cache_key(freq_axis, measured, cfg.fs, cfg.num_taps)
    correction_baseline._RT60_CACHE[rt60_key] = (0.3, {}, 0.3)

    calls = {"count": 0}

    def fake_compute_leveling(_cfg, _freq, _meas, _target, *, stereo_link_ctx=None):
        calls["count"] += 1
        assert stereo_link_ctx is None
        return (5.0, 1.5, 5.0, 0.0, "SmartScanMedian", 500.0, 2000.0)

    monkeypatch.setattr(correction_baseline, "compute_leveling", fake_compute_leveling)

    baseline = correction_baseline._prepare_correction_baseline(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=freq_axis,
        m_in=measured,
        reflections=[],
        st={},
        m_anal=measured,
        m_plot_db=None,
        is_psy=False,
        cmp=None,
        analysis_mode="native",
        gain_db=gain_db,
        logger=None,
        interpolate_response=lambda *_args, **_kwargs: np.zeros_like(freq_axis),
        _cfg_float_allow_zero=lambda *_args, **_kwargs: 0.0,
        stereo_link_ctx=None,
        presolve_mode=False,
    )

    correction_baseline._clear_rt60_cache()

    assert calls["count"] == 1
    assert baseline.target_shift_db == 5.0
    assert baseline.calc_offset_db == -3.5
    assert baseline.target_level_db_window == 5.0
    assert np.allclose(baseline.target_mags, 5.0)


def test_prepare_correction_baseline_keeps_forced_offset_when_shared_target_shift_is_applied(monkeypatch):
    freq_axis = np.linspace(20.0, 20000.0, 2048, dtype=float)
    measured = np.zeros_like(freq_axis)
    gain_db = np.zeros_like(freq_axis)
    cfg = SimpleNamespace(
        fs=44100,
        num_taps=65536,
        house_freqs=None,
        house_mags=None,
        hpf_settings=None,
        enable_tdc=False,
        enable_null_guard=False,
        lvl_min=500.0,
        lvl_max=2000.0,
        comparison_mode=False,
        lvl_force_offset_db=None,
    )
    stereo_ctx = StereoLinkContext(
        forced_window_hz=(500.0, 2000.0),
        forced_offset_db=1.5,
        shared_target_level_db=5.0,
        shared_target_shift_db=3.0,
    )

    correction_baseline._clear_rt60_cache()
    rt60_key = correction_baseline._rt60_cache_key(freq_axis, measured, cfg.fs, cfg.num_taps)
    correction_baseline._RT60_CACHE[rt60_key] = (0.3, {}, 0.3)

    calls = {"count": 0}

    def fake_compute_leveling(_cfg, _freq, _meas, _target, *, stereo_link_ctx=None):
        calls["count"] += 1
        assert stereo_link_ctx is stereo_ctx
        return (5.0, 1.5, 5.0, 0.0, "ForcedOffset", 500.0, 2000.0)

    monkeypatch.setattr(correction_baseline, "compute_leveling", fake_compute_leveling)

    baseline = correction_baseline._prepare_correction_baseline(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=freq_axis,
        m_in=measured,
        reflections=[],
        st={},
        m_anal=measured,
        m_plot_db=None,
        is_psy=False,
        cmp=None,
        analysis_mode="native",
        gain_db=gain_db,
        logger=None,
        interpolate_response=lambda *_args, **_kwargs: np.zeros_like(freq_axis),
        _cfg_float_allow_zero=lambda *_args, **_kwargs: 0.0,
        stereo_link_ctx=stereo_ctx,
        presolve_mode=False,
    )

    correction_baseline._clear_rt60_cache()

    assert calls["count"] == 1
    assert baseline.target_shift_db == 3.0
    assert baseline.calc_offset_db == 1.5
    assert baseline.target_level_db_window == 3.0
    assert np.allclose(baseline.target_mags, 3.0)


def test_generate_filter_smoke_trans_width_none_does_not_crash(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )
    cfg.trans_width = None

    imp, st = generate_filter(f, m, p, cfg)

    assert imp is not None
    assert len(imp) > 0
    assert isinstance(st, dict)
    assert "filter_mags" in st


def _sample_tdc_inputs():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    reflections = [{"freq": 50.0, "gd_error": 500.0}]
    rt60_info = 0.4
    return freq_axis, target_mags, reflections, rt60_info


def test_tdc_strength_zero_disables_reduction():
    freq_axis, target_mags, reflections, rt60_info = _sample_tdc_inputs()

    out_enabled = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=1.0,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )
    out_zero = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.0,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )

    assert float(np.max(target_mags - out_enabled)) > 0.1
    assert np.allclose(out_zero, target_mags, atol=1e-12)


def test_tdc_max_reduction_zero_disables_reduction():
    freq_axis, target_mags, reflections, rt60_info = _sample_tdc_inputs()

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.8,
        max_total_reduction_db=0.0,
        max_slope_db_per_oct=0.0,
    )

    assert np.allclose(out, target_mags, atol=1e-12)
