from types import SimpleNamespace

import numpy as np

from camillafir.config.models import FilterConfig
from camillafir.dsp.camillafir_dsp import generate_filter
from camillafir.dsp.camillafir_leveling import compute_leveling


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
