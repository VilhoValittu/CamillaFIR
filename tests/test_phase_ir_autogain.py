from types import SimpleNamespace

import numpy as np
import pytest

from camillafir.dsp.phase_ir_autogain import compute_auto_gain_and_headroom


class _NullLogger:
    def info(self, _msg: str) -> None:
        return None


def _run(cfg: SimpleNamespace, gain_db, mask_c):
    return compute_auto_gain_and_headroom(
        cfg=cfg,
        gain_db=np.asarray(gain_db, dtype=float),
        mask_c=np.asarray(mask_c, dtype=bool),
        logger=_NullLogger(),
    )


def test_spike_peak_uses_percentile_for_auto_gain():
    gain_db = np.zeros(200, dtype=float)
    gain_db[100] = 8.0
    cfg = SimpleNamespace(
        auto_gain_margin_db=0.0,
        global_gain_db=0.0,
        do_normalize=False,
        auto_gain_peak_percentile=99.5,
        auto_gain_spike_threshold_db=1.0,
        auto_gain_spike_guard_db=0.0,
    )

    out = _run(cfg, gain_db=gain_db, mask_c=np.ones_like(gain_db, dtype=bool))

    expected_eff_peak = float(np.percentile(gain_db, 99.5))
    assert float(out["current_peak_gain"]) == pytest.approx(8.0, abs=1e-9)
    assert float(out["auto_global_gain_db"]) == pytest.approx(-expected_eff_peak, abs=1e-9)


def test_non_spike_peak_keeps_max_based_auto_gain():
    gain_db = np.linspace(0.0, 2.0, 400, dtype=float)
    cfg = SimpleNamespace(
        auto_gain_margin_db=0.0,
        global_gain_db=0.0,
        do_normalize=False,
        auto_gain_peak_percentile=99.5,
        auto_gain_spike_threshold_db=1.0,
        auto_gain_spike_guard_db=0.0,
    )

    out = _run(cfg, gain_db=gain_db, mask_c=np.ones_like(gain_db, dtype=bool))

    assert float(out["current_peak_gain"]) == pytest.approx(2.0, abs=1e-9)
    assert float(out["auto_global_gain_db"]) == pytest.approx(-2.0, abs=1e-9)


def test_normalize_headroom_uses_effective_peak_not_spike_max():
    gain_db = np.zeros(200, dtype=float)
    gain_db[40] = 8.0
    cfg = SimpleNamespace(
        auto_gain_margin_db=0.0,
        global_gain_db=0.0,
        do_normalize=True,
        auto_gain_peak_percentile=99.5,
        auto_gain_spike_threshold_db=1.0,
        auto_gain_spike_guard_db=0.0,
    )

    out = _run(cfg, gain_db=gain_db, mask_c=np.ones_like(gain_db, dtype=bool))

    expected_eff_peak = float(np.percentile(gain_db, 99.5))
    assert float(out["auto_global_gain_db"]) == pytest.approx(-expected_eff_peak, abs=1e-9)
    assert float(out["auto_headroom_db"]) == pytest.approx(-0.1, abs=1e-9)


def test_peak_detection_uses_correction_band_mask():
    gain_db = np.zeros(64, dtype=float)
    gain_db[0] = 6.0  # Outside correction mask
    mask_c = np.ones_like(gain_db, dtype=bool)
    mask_c[0] = False
    cfg = SimpleNamespace(
        auto_gain_margin_db=0.0,
        global_gain_db=0.0,
        do_normalize=False,
        auto_gain_peak_percentile=99.5,
        auto_gain_spike_threshold_db=1.0,
        auto_gain_spike_guard_db=0.0,
    )

    out = _run(cfg, gain_db=gain_db, mask_c=mask_c)

    assert float(out["current_peak_gain"]) == pytest.approx(0.0, abs=1e-9)
    assert float(out["auto_global_gain_db"]) == pytest.approx(0.0, abs=1e-9)
