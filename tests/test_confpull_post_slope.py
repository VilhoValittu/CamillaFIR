from types import SimpleNamespace

import numpy as np

from camillafir.dsp.camillafir_dsp import apply_confidence_weighted_target_pull
from camillafir.dsp.correction_mag import _apply_confpull_post_slope


class _NullLogger:
    def info(self, _msg: str) -> None:
        return None


def test_confpull_post_slope_does_not_create_or_increase_boost():
    n = 128
    freq_axis = np.geomspace(10.0, 500.0, n).astype(float)
    mask_c = np.ones(n, dtype=bool)
    conf_mask = np.zeros(n, dtype=float)  # force strong pull towards reference

    gain_db = np.zeros(n, dtype=float)
    gain_db[50] = 3.0

    measured_ref = np.zeros(n, dtype=float)
    measured_ref[30] = 12.0  # would create new boost if not guarded
    measured_ref[50] = 20.0  # would increase existing boost if not guarded

    cfg = SimpleNamespace(
        conf_pull_floor=0.05,
        conf_pull_ceil=0.95,
        conf_pull_max_hz=300.0,
        conf_pull_gamma_cut=0.55,
        conf_pull_gamma_boost=1.35,
        conf_pull_conf_smooth_sigma=0.0,
        conf_pull_bass_floor_hz=120.0,
        conf_pull_bass_floor_min=0.0,
    )

    out = _apply_confpull_post_slope(
        gain_db_in=gain_db,
        mask_c_in=mask_c,
        measured_ref_db=measured_ref,
        cfg=cfg,
        st={},
        conf_mask=conf_mask,
        freq_axis=freq_axis,
        logger=_NullLogger(),
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
    )
    out = np.asarray(out, dtype=float)

    assert float(out[30]) <= 1e-9
    assert float(out[50]) <= float(gain_db[50]) + 1e-9


def test_confpull_post_slope_disables_bass_boost_restore_in_isolation_mode():
    n = 128
    freq_axis = np.geomspace(10.0, 500.0, n).astype(float)
    mask_c = np.ones(n, dtype=bool)
    conf_mask = np.zeros(n, dtype=float)
    gain_db = np.zeros(n, dtype=float)
    gain_db[(freq_axis >= 30.0) & (freq_axis <= 120.0)] = 2.0
    measured_ref = np.zeros(n, dtype=float)

    cfg = SimpleNamespace(
        conf_pull_floor=0.05,
        conf_pull_ceil=0.95,
        conf_pull_max_hz=300.0,
        conf_pull_gamma_cut=0.55,
        conf_pull_gamma_boost=1.35,
        conf_pull_conf_smooth_sigma=0.0,
        conf_pull_bass_floor_hz=120.0,
        conf_pull_bass_floor_min=0.0,
        conf_pull_bass_boost_floor_hz=200.0,
        conf_pull_bass_boost_floor_min=0.45,
        conf_pull_bass_boost_restore=0.55,
        bass_adaptive_isolation_mode=True,
    )
    st = {}
    _ = _apply_confpull_post_slope(
        gain_db_in=gain_db,
        mask_c_in=mask_c,
        measured_ref_db=measured_ref,
        cfg=cfg,
        st=st,
        conf_mask=conf_mask,
        freq_axis=freq_axis,
        logger=_NullLogger(),
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
    )

    assert bool(st.get("bass_adaptive_isolation_mode", False))
    assert float(st.get("conf_pull_post_bass_boost_floor_hz", -1.0)) == 0.0
    assert float(st.get("conf_pull_post_bass_boost_floor_min", -1.0)) == 0.0
    assert float(st.get("conf_pull_post_bass_boost_restore", -1.0)) == 0.0
