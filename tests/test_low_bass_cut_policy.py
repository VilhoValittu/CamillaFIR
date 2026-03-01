import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter


def _synth_flat_measurement(fs: int = 48000, n: int = 1024):
    f = np.geomspace(10.0, fs * 0.49, n).astype(float)
    m = np.zeros_like(f, dtype=float)
    p = np.zeros_like(f, dtype=float)
    return f, m, p


def _synth_lf_hump_measurement(fs: int = 48000, n: int = 2048):
    f = np.geomspace(10.0, fs * 0.49, n).astype(float)
    m = np.zeros_like(f, dtype=float)
    # Deliberate low-frequency hump around ~24 Hz to force LF cuts.
    m += 12.0 * np.exp(-0.5 * ((np.log2(f) - np.log2(24.0)) / 0.35) ** 2)
    p = np.zeros_like(f, dtype=float)
    return f, m, p


def test_low_bass_cut_remains_hard_after_full_post_pipeline():
    f, m, p = _synth_flat_measurement()
    low_hz = 40.0

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        enable_mag_correction=True,
        low_bass_cut_enable=True,
        low_bass_cut_hz=low_hz,
        exc_prot=False,
        mag_c_min=10.0,
        mag_c_max=200.0,
        max_boost_db=8.0,
        max_cut_db=15.0,
        enable_afdw=False,
        filter_smooth=12,
        reg_strength=30.0,
        house_freqs=[10.0, 20.0, 30.0, 40.0, 80.0, 200.0, 1000.0],
        house_mags=[8.0, 8.0, 8.0, 7.0, 4.0, 1.0, 0.0],
    )
    setattr(cfg, "enable_residual_pass", True)

    _, st = generate_filter(f, m, p, cfg)

    freq_axis = np.asarray(st.get("freq_axis", []), dtype=float)
    filt_db = np.asarray(st.get("filter_mags", []), dtype=float)
    lf_mask = (freq_axis > 0.0) & (freq_axis <= low_hz)

    assert np.any(lf_mask)
    assert float(np.max(filt_db[lf_mask])) <= 1e-9
    assert float(st.get("lf_boost_max_db", 0.0) or 0.0) <= 1e-9


def test_low_bass_cut_strength_survives_post_slope_confpull_pipeline():
    f, m, p = _synth_lf_hump_measurement()
    low_hz = 40.0

    base_kwargs = dict(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        enable_mag_correction=True,
        low_bass_cut_enable=True,
        low_bass_cut_hz=low_hz,
        exc_prot=False,
        mag_c_min=10.0,
        mag_c_max=230.0,
        max_boost_db=3.0,
        max_cut_db=30.0,
        max_slope_db_per_oct=24.0,
        enable_afdw=True,
        filter_smooth=12,
        reg_strength=30.0,
        conf_pull_floor=0.15,
        conf_pull_ceil=0.95,
        conf_pull_max_hz=200.0,
        conf_pull_gamma_cut=0.55,
        conf_pull_gamma_boost=1.35,
        conf_pull_conf_smooth_sigma=2.0,
        conf_pull_bass_floor_hz=120.0,
        conf_pull_bass_floor_min=0.25,
        house_freqs=[10.0, 20.0, 40.0, 80.0, 200.0, 1000.0],
        house_mags=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    cfg_weak = FilterConfig(low_bass_cut_strength=0.0, **base_kwargs)
    _, st_weak = generate_filter(f, m, p, cfg_weak)

    cfg_strong = FilterConfig(low_bass_cut_strength=1.0, **base_kwargs)
    _, st_strong = generate_filter(f, m, p, cfg_strong)

    freq_axis = np.asarray(st_strong.get("freq_axis", []), dtype=float)
    g_weak = np.asarray(st_weak.get("predicted_filter_mags", []), dtype=float)
    g_strong = np.asarray(st_strong.get("predicted_filter_mags", []), dtype=float)

    assert freq_axis.size > 64
    assert g_weak.size == freq_axis.size
    assert g_strong.size == freq_axis.size

    cut20_weak = float(np.interp(20.0, freq_axis, g_weak))
    cut20_strong = float(np.interp(20.0, freq_axis, g_strong))
    assert cut20_strong <= (cut20_weak - 0.2)

    floor_bins = int(st_strong.get("low_bass_hard_reapply_floor_bins", 0) or 0)
    assert floor_bins > 0
