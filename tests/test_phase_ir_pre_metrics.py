from types import SimpleNamespace

import numpy as np

from camillafir.dsp.phase_ir_metrics import (
    compute_pre_post_energy_metrics,
    format_pre_energy_metric_note,
    _summarize_ir_metrics,
)
from camillafir.ui import camillafir_plot as plots


def test_summarize_ir_metrics_linear_phase_ir_is_valid():
    ir = np.zeros(512, dtype=float)
    center = 256
    for idx, amp in enumerate((0.2, 0.45, 0.8, 1.0, 0.8, 0.45, 0.2)):
        ir[center - 3 + idx] = amp
    st = {"ir_energy_split_samples": center}

    out = _summarize_ir_metrics(
        ir,
        SimpleNamespace(fs=44100, filter_type_str="Linear", ir_anchor_mode="peak"),
        st,
    )

    assert bool(out["pre_energy_metric_valid"]) is True
    assert bool(out["pre_energy_metric_suspect"]) is False
    assert np.isfinite(float(out["ir_pre_post_ratio"]))
    assert np.isfinite(float(out["ir_pre_ringing_db"]))
    assert str(out["pre_energy_metric_note"]) == "ok"


def test_compute_pre_post_energy_metrics_minimum_phase_returns_clean_na():
    ir = np.zeros(256, dtype=float)
    ir[8] = 1.0
    ir[9:20] = np.linspace(0.8, 0.1, 11)

    out = compute_pre_post_energy_metrics(
        ir,
        44100,
        peak_idx=8,
        filter_type="Minimum",
        phase_mode="min_causal",
    )

    assert bool(out["valid"]) is False
    assert str(out["reason_code"]) == "not_applicable_phase_mode"
    assert format_pre_energy_metric_note(valid=False, reason_code=out["reason_code"]).startswith("n/a (")


def test_summarize_ir_metrics_anchor_near_boundary_avoids_bug_text():
    ir = np.zeros(256, dtype=float)
    ir[6] = 1.0
    ir[7:24] = np.linspace(0.7, 0.02, 17)
    st = {"ir_energy_split_samples": 6}

    out = _summarize_ir_metrics(
        ir,
        SimpleNamespace(fs=44100, filter_type_str="Asymmetric", ir_anchor_mode="peak"),
        st,
    )

    assert bool(out["pre_energy_metric_valid"]) is False
    assert str(out["pre_energy_metric_reason_code"]) == "anchor_near_boundary"
    assert "likely zeroed or split issue" not in str(out["pre_energy_metric_note"])
    assert str(out["pre_energy_metric_note"]).startswith("n/a (")


def test_compute_pre_post_energy_metrics_zero_energy_is_guarded():
    ir = np.zeros(128, dtype=float)

    out = compute_pre_post_energy_metrics(ir, 44100, peak_idx=64, filter_type="Linear", phase_mode="peak")

    assert bool(out["valid"]) is False
    assert str(out["reason_code"]) == "near_zero_total_energy"
    assert np.isnan(float(out["ratio"]))
    assert np.isnan(float(out["pre_ringing_db"]))


def test_dsp_quality_report_uses_clean_pre_energy_reason_text():
    st = {
        "pre_energy_metric_valid": False,
        "pre_energy_metric_suspect": True,
        "pre_energy_metric_reason_code": "anchor_near_boundary",
        "pre_energy_metric_note": "",
    }

    block = "\n".join(plots.format_dsp_quality_report_block({}, st, st))

    assert "Pre-energy metric sanity:" in block
    assert "anchor too close to boundary after alignment" in block
    assert "likely zeroed or split issue" not in block
