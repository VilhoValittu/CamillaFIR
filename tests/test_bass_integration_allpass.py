from __future__ import annotations

import math

import numpy as np

from camillafir.dsp.bass_integration import (
    _apply_allpass_to_transfer,
    _apply_branch_filters,
    _build_direct_dac_trial_bundle,
    recommend_direct_dac_allpass,
)
from camillafir.io.measurement_bundle import BassIntegrationBundle, TransferData


def _transfer(
    freqs_hz: np.ndarray,
    *,
    delay_s: float = 0.0,
    gain: float = 1.0,
    label: str,
) -> TransferData:
    freqs = np.asarray(freqs_hz, dtype=float)
    spec = float(gain) * np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    return TransferData(
        freqs_hz=freqs,
        complex_spec=np.asarray(spec, dtype=np.complex128),
        mag_db=np.asarray(20.0 * np.log10(np.maximum(np.abs(spec), 1e-12)), dtype=float),
        phase_deg=np.asarray(np.rad2deg(np.unwrap(np.angle(spec))), dtype=float),
        sample_rate=48_000,
        label=label,
    )


def _bundle(
    *,
    main_delay_s: float = 0.0,
    sub_delay_s: float = 0.0,
) -> BassIntegrationBundle:
    freqs = np.geomspace(10.0, 320.0, 1024)
    l_main = _transfer(freqs, delay_s=main_delay_s, gain=1.0, label="l_main")
    r_main = _transfer(freqs, delay_s=main_delay_s, gain=1.0, label="r_main")
    l_sub = _transfer(freqs, delay_s=sub_delay_s, gain=0.5, label="l_sub")
    r_sub = _transfer(freqs, delay_s=sub_delay_s, gain=0.5, label="r_sub")
    l_total = _transfer(freqs, delay_s=0.0, gain=1.0, label="l_total")
    r_total = _transfer(freqs, delay_s=0.0, gain=1.0, label="r_total")
    return BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )


def test_direct_dac_trial_bundle_identity_when_allpass_not_provided() -> None:
    bundle = _bundle()

    baseline = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )
    explicit_none = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_allpass_freq_hz=None,
        sub_allpass_q=None,
    )

    np.testing.assert_allclose(baseline.l_sub.complex_spec, explicit_none.l_sub.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.r_sub.complex_spec, explicit_none.r_sub.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.l_total.complex_spec, explicit_none.l_total.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.r_total.complex_spec, explicit_none.r_total.complex_spec, atol=1e-12)


def test_direct_dac_trial_bundle_applies_shared_allpass_to_both_sub_branches() -> None:
    bundle = _bundle()
    trial = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_allpass_freq_hz=76.0,
        sub_allpass_q=0.9,
    )
    expected_l = _apply_allpass_to_transfer(
        _apply_branch_filters(
            bundle.l_sub,
            hpf_hz=20.0,
            hpf_order=2,
            lpf_hz=80.0,
            lpf_order=4,
            label="expected_l_sub",
        ),
        freq_hz=76.0,
        q=0.9,
        label="expected_l_sub_ap",
    )
    expected_r = _apply_allpass_to_transfer(
        _apply_branch_filters(
            bundle.r_sub,
            hpf_hz=20.0,
            hpf_order=2,
            lpf_hz=80.0,
            lpf_order=4,
            label="expected_r_sub",
        ),
        freq_hz=76.0,
        q=0.9,
        label="expected_r_sub_ap",
    )

    np.testing.assert_allclose(trial.l_sub.complex_spec, expected_l.complex_spec, atol=1e-12)
    np.testing.assert_allclose(trial.r_sub.complex_spec, expected_r.complex_spec, atol=1e-12)


def test_recommend_direct_dac_allpass_rejects_no_gain_case() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=0.0)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert result["enabled"] is False
    assert result["freq_hz"] == 0.0
    assert "baseline" in result and isinstance(result["baseline"], dict)
    assert "optimized" in result and isinstance(result["optimized"], dict)


def test_recommend_direct_dac_allpass_improves_synthetic_misaligned_case() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=-0.0030)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert result["enabled"] is True
    baseline = dict(result["baseline"])
    optimized = dict(result["optimized"])
    assert (
        float(optimized.get("cancellation_risk", float("inf"))) < float(baseline.get("cancellation_risk", float("inf")))
        or float(optimized.get("overlap_ripple_db", float("inf"))) < float(baseline.get("overlap_ripple_db", float("inf")))
        or float(optimized.get("xo_gd_mismatch_ms", float("inf"))) < float(baseline.get("xo_gd_mismatch_ms", float("inf")))
    )


def test_recommend_direct_dac_allpass_returns_structured_metrics() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=-0.0030)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert isinstance(result["enabled"], bool)
    assert math.isfinite(float(result["improvement_score"]))
    assert isinstance(result["baseline"], dict)
    assert isinstance(result["optimized"], dict)
    if result["enabled"]:
        assert math.isfinite(float(result["freq_hz"]))
        assert math.isfinite(float(result["q"]))
