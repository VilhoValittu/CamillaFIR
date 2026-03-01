from types import SimpleNamespace

import pytest
import numpy as np

from camillafir.dsp.phase_ir_metrics import _summarize_ir_metrics
from camillafir.dsp.phase_ir_utils import _pre_post_energy_ratio, _pre_ringing_db


def test_pre_post_energy_ratio_returns_nan_for_too_short_ir():
    ir = np.ones(16, dtype=float)
    ratio = _pre_post_energy_ratio(ir, split=8)
    db = _pre_ringing_db(ir, split=8)
    assert np.isnan(ratio)
    assert np.isnan(db)


def test_pre_post_energy_ratio_clamps_split_safely():
    ir = np.ones(64, dtype=float)
    ratio = _pre_post_energy_ratio(ir, split=0)
    db = _pre_ringing_db(ir, split=0)
    assert ratio == pytest.approx(1.0, abs=1e-12)
    assert db == pytest.approx(0.0, abs=1e-12)


def test_summarize_ir_metrics_sets_suspect_flag_for_near_zero_pre_energy():
    ir = np.zeros(64, dtype=float)
    ir[40] = 1.0
    ir[41:] = 0.2
    st = {"ir_energy_split_samples": 40}

    out = _summarize_ir_metrics(ir, SimpleNamespace(), st)

    assert np.isnan(float(out["ir_pre_post_ratio"]))
    assert float(out["ir_pre_post_ratio_raw"]) < 1e-10
    assert bool(out["pre_energy_metric_suspect"]) is True
    assert "pre/post < 1e-10" in str(out["pre_energy_metric_note"])
