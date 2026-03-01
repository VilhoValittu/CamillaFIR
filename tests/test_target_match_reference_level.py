import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter
from camillafir.dsp.target_match import target_match_from_stats


def _synth_abs_level_measurement(fs: int = 48000, n: int = 1024, level_db: float = 68.0):
    f = np.geomspace(10.0, fs * 0.49, n).astype(float)
    m = np.zeros_like(f, dtype=float) + float(level_db)
    p = np.zeros_like(f, dtype=float)
    return f, m, p


def test_target_match_keeps_absolute_reference_level_consistent():
    f, m, p = _synth_abs_level_measurement()

    cfg = FilterConfig(
        fs=48000,
        num_taps=16384,
        filter_type_str="Linear Phase",
        enable_mag_correction=True,
        comparison_mode=False,
        mag_c_min=20.0,
        mag_c_max=200.0,
        max_boost_db=3.0,
        max_cut_db=15.0,
        enable_afdw=True,
        house_freqs=[10.0, 20.0, 40.0, 80.0, 200.0, 1000.0, 20000.0],
        house_mags=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    _, st = generate_filter(f, m, p, cfg)
    rms_db, match_pct = target_match_from_stats(st, include_filter=True, use_confidence=True)

    assert rms_db is not None
    assert match_pct is not None
    assert float(rms_db) < 3.0
    assert float(match_pct) > 80.0

