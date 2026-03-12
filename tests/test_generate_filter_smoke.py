import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter
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
    assert isinstance(st, dict)
    assert "offset_db" in st


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
