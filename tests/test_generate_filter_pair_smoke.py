import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter_pair


def test_generate_filter_pair_smoke_minimal_stereo():
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    left_mag = np.zeros_like(freq)
    right_mag = np.zeros_like(freq)

    cfg = FilterConfig(
        fs=44100,
        num_taps=16384,
        filter_type_str="Linear Phase",
        stereo_link=True,
    )

    l_imp, l_st, r_imp, r_st = generate_filter_pair(freq, left_mag, phase, freq, right_mag, phase, cfg)

    assert isinstance(l_imp, np.ndarray)
    assert isinstance(r_imp, np.ndarray)
    assert l_imp.ndim == 1
    assert r_imp.ndim == 1
    assert l_imp.size > 0
    assert r_imp.size > 0
    assert isinstance(l_st, dict)
    assert isinstance(r_st, dict)
    assert "offset_db" in l_st
    assert "offset_db" in r_st
