from types import SimpleNamespace

import numpy as np
import pytest

from camillafir.dsp.phase_ir_guards import _pre_energy_guard


def test_pre_energy_guard_sets_trigger_flag_and_stats_when_triggered():
    cfg = SimpleNamespace(enable_ir_pre_energy_guard=True, max_pre_ringing_db=-20.0)
    st = {"ir_energy_split_samples": 24}
    ir = np.zeros(64, dtype=float)
    ir[:24] = 0.8
    ir[24] = 1.0
    ir[25:] = 0.05

    out, info = _pre_energy_guard(ir, cfg, st)

    assert bool(info["triggered"]) is True
    assert float(info["scale"]) < 1.0
    assert bool(st["ir_pre_energy_guard_enabled"]) is True
    assert bool(st["ir_pre_energy_guard_triggered"]) is True
    assert float(st["ir_pre_energy_guard_scale"]) == pytest.approx(float(info["scale"]), abs=1e-12)
    assert np.max(np.abs(out[:24])) < np.max(np.abs(ir[:24]))
    assert not np.allclose(out[:24], 0.0, atol=0.0, rtol=0.0)


def test_pre_energy_guard_writes_non_triggered_stats_when_disabled():
    cfg = SimpleNamespace(enable_ir_pre_energy_guard=False, max_pre_ringing_db=-20.0)
    st = {}
    ir = np.zeros(64, dtype=float)
    ir[20] = 1.0
    ir[21:] = 0.1

    out, info = _pre_energy_guard(ir, cfg, st)

    assert np.allclose(out, ir, atol=0.0, rtol=0.0)
    assert bool(info["triggered"]) is False
    assert bool(st["ir_pre_energy_guard_enabled"]) is False
    assert bool(st["ir_pre_energy_guard_triggered"]) is False
    assert float(st["ir_pre_energy_guard_scale"]) == pytest.approx(1.0, abs=1e-12)


def test_pre_energy_guard_uses_alignment_split_and_caps_ratio():
    cfg = SimpleNamespace(
        enable_ir_pre_energy_guard=True,
        pre_energy_ratio_max=0.25,
        pre_energy_guard_strength=1.0,
        max_pre_ringing_db=-20.0,
    )
    st = {"ir_energy_split_samples": 20}
    ir = np.zeros(64, dtype=float)
    ir[:20] = 1.0
    ir[20] = 1.0
    ir[21:] = 0.2

    out, info = _pre_energy_guard(ir, cfg, st)

    pre_e = float(np.mean(out[:20] ** 2))
    post_e = float(np.mean(out[20:] ** 2))
    ratio = pre_e / max(post_e, 1e-20)
    assert bool(info["triggered"]) is True
    assert ratio <= 0.25 + 5e-2
    assert int(st["ir_pre_energy_guard_split_samples"]) == 20
