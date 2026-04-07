from __future__ import annotations

import io

import numpy as np
import scipy.io.wavfile

from camillafir.application.health_service import compute_health
from camillafir.auto_mode.cache_signature import _auto_measurement_signature
from camillafir.config.camillafir_pipeline import build_filter_config, build_xos_hpf, collect_ui_data
from camillafir.config.models import FilterConfig
from camillafir.engine_run import run_pipeline
from camillafir.io.measurement_bundle import BassIntegrationBundle, TransferData
from camillafir.io.measurements_loader import load_bass_integration_measurements
from camillafir.io.measurements_wav import parse_coherent_transfer_from_wav_bytes
from camillafir.ui.export_summary_text import (
    _append_bass_integration_summary,
    _append_main_speaker_xo_hpf_summary,
)
from camillafir.workflow.auto_flow import (
    _build_auto_selected_text,
    _resolve_auto_hpf_seed_source,
)


def _impulse_wav_bytes(*, fs_hz: int = 48000, length: int = 4096, delay_samples: int = 0) -> bytes:
    sig = np.zeros(int(length), dtype=np.float32)
    sig[int(delay_samples)] = 1.0
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, int(fs_hz), sig)
    return buf.getvalue()


def _upload_dict(name: str, content: bytes) -> dict:
    return {"filename": name, "content": bytes(content)}


def _flat_transfer(freqs_hz: np.ndarray, complex_gain: complex, *, label: str) -> TransferData:
    spec = np.full(np.asarray(freqs_hz, dtype=float).shape, complex_gain, dtype=np.complex128)
    return TransferData(
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        complex_spec=spec,
        mag_db=20.0 * np.log10(np.maximum(np.abs(spec), 1e-12)),
        phase_deg=np.rad2deg(np.unwrap(np.angle(spec))),
        sample_rate=48000,
        label=label,
    )


def _delayed_transfer(freqs_hz: np.ndarray, delay_s: float, *, label: str) -> TransferData:
    freqs = np.asarray(freqs_hz, dtype=float)
    spec = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s)).astype(np.complex128)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=np.zeros_like(freqs, dtype=float),
        phase_deg=np.rad2deg(np.unwrap(np.angle(spec))),
        sample_rate=48000,
        label=label,
    )


def test_parse_coherent_transfer_from_wav_bytes_preserves_delay_phase() -> None:
    fs_hz = 48000
    delayed_samples = 24
    ref = parse_coherent_transfer_from_wav_bytes(
        _impulse_wav_bytes(fs_hz=fs_hz, delay_samples=0),
        label="ref",
    )
    delayed = parse_coherent_transfer_from_wav_bytes(
        _impulse_wav_bytes(fs_hz=fs_hz, delay_samples=delayed_samples),
        label="delayed",
    )

    assert ref is not None
    assert delayed is not None

    idx = int(np.argmin(np.abs(ref.freqs_hz - 1000.0)))
    ratio = delayed.complex_spec[idx] / ref.complex_spec[idx]
    expected_phase = -2.0 * np.pi * ref.freqs_hz[idx] * float(delayed_samples) / float(fs_hz)
    phase_err = np.angle(np.exp(1j * (np.angle(ratio) - expected_phase)))

    assert abs(float(phase_err)) < 0.05


def test_load_bass_integration_measurements_builds_complex_totals_and_bundle() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "avr_lfe_main_decomposed",
        "bass_integration_profile": "normal",
        "avr_crossover_hz": 80.0,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=0)),
    }

    bundle, f_l, m_l, p_l, f_r, m_r, p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    assert f_l is not None and m_l is not None and p_l is not None
    assert f_r is not None and m_r is not None and p_r is not None
    assert int(bundle.l_main.sample_rate) == 48000
    assert bundle.profile == "normal"
    assert "cancellation_risk" in bundle.diagnostics

    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    l_main_db = float(bundle.l_main.mag_db[idx])
    l_total_db = float(np.asarray(m_l, dtype=float)[idx])

    assert l_total_db > (l_main_db + 9.0)


def test_load_bass_integration_measurements_uses_shared_anchor_for_late_rew_ir() -> None:
    base_delay = 48_000
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "avr_lfe_main_decomposed",
        "bass_integration_profile": "safe",
        "avr_crossover_hz": 120.0,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
    }

    bundle, f_l, m_l, _p_l, _f_r, _m_r, _p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    assert float(bundle.l_main.mag_db[idx]) > -1.0
    assert float(bundle.l_sub.mag_db[idx]) > -1.0
    assert float(np.asarray(m_l, dtype=float)[idx]) > 9.0


def test_load_bass_integration_measurements_accepts_single_direct_dac_sub() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "safe",
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
    }

    bundle, f_l, m_l, _p_l, f_r, m_r, _p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    assert f_l is not None and m_l is not None
    assert f_r is not None and m_r is not None
    assert float(np.max(np.abs(bundle.r_sub.complex_spec))) == 0.0

    # In direct_dac mode the returned m_l/m_r come from the main-only measurement,
    # not the combined total, so the level is ~0 dB for a unit impulse (not +6 dB).
    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    assert float(np.asarray(m_l, dtype=float)[idx]) > -1.0
    assert float(np.asarray(m_r, dtype=float)[idx]) > -1.0


def test_collect_ui_data_preserves_direct_dac_and_forces_main_hpf_from_sub_xo() -> None:
    data = collect_ui_data(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 92.0,
            "sub_crossover_slope": 24,
            "sub_crossover_manual_override": True,
            "sub_hpf_freq": 19.0,
            "sub_hpf_slope": 12,
        }
    )
    xos, hpf = build_xos_hpf(data)
    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=48000,
        taps_v=65536,
        data=data,
        xos=xos,
        hpf=hpf,
        hc_f=None,
        hc_m=None,
    )

    assert data["bass_integration_mode"] == "direct_dac"
    assert data["sub_crossover_manual_override"] is True
    assert cfg.sub_integration_enable is True
    assert cfg.sub_generate_ir is True
    assert cfg.hpf_settings == {"enabled": True, "freq": 92.0, "order": 4}
    assert float(cfg.sub_crossover_hz) == 92.0
    assert int(cfg.sub_crossover_order) == 4
    assert float(cfg.sub_hpf_freq) == 19.0
    assert int(cfg.sub_hpf_order) == 2


def test_build_xos_hpf_ignores_main_speaker_xo_for_unsupported_filter_types() -> None:
    xos, hpf = build_xos_hpf(
        {
            "filter_type": "mixed phase",
            "xo1_f": 500.0,
            "xo1_s": 12,
            "hpf_enable": True,
            "hpf_freq": 20.0,
            "hpf_slope": 24,
        }
    )

    assert xos == []
    assert hpf == {"enabled": True, "freq": 20.0, "order": 4}


def test_main_speaker_xo_summary_reports_off_for_unsupported_filter_types() -> None:
    summary = _append_main_speaker_xo_hpf_summary(
        "",
        {
            "filter_type": "minimum phase",
            "xo1_f": 500.0,
            "xo1_s": 12,
            "hpf_enable": True,
            "hpf_freq": 20.0,
            "hpf_slope": 24,
        },
    )

    assert "Crossovers: OFF" in summary
    assert "500.0 Hz / 12 dB/oct" not in summary
    assert "HPF: ON (20.0 Hz / 24 dB/oct)" in summary


def test_direct_dac_summary_reports_main_hpf_from_sub_crossover() -> None:
    summary = _append_main_speaker_xo_hpf_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 40.0,
            "sub_crossover_slope": 24,
            "hpf_enable": True,
            "hpf_freq": 17.9,
            "hpf_slope": 12,
        },
    )

    assert "HPF: ON (40.0 Hz / 24 dB/oct)" in summary
    assert "17.9 Hz / 12 dB/oct" not in summary


def test_direct_dac_summary_labels_xo_as_main_sub_crossover() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 70.0,
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 70.0,
                "recommended_crossover_hz": 80.0,
                "profile": "safe",
                "inputs": {},
            },
        },
    )

    assert "Main/Sub XO: 70.0 Hz" in summary
    assert "Recommended Main/Sub XO: 80 Hz" in summary
    assert "Direct-DAC XO is the crossover between the main-speaker HPF and the subwoofer LPF." in summary
    assert "AVR crossover:" not in summary


def test_compute_health_blocks_bass_integration_without_all_four_wav_files() -> None:
    hr = compute_health(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "avr_crossover_hz": 80.0,
            "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        },
        mode="AUTO",
    )

    assert hr.blocked
    assert any(issue.level == "crit" for issue in hr.issues)
    assert any("Bass Integration" in issue.title for issue in hr.issues)


def test_compute_health_accepts_single_direct_dac_sub() -> None:
    hr = compute_health(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "avr_crossover_hz": 80.0,
            "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
            "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
            "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
        },
        mode="AUTO",
    )

    assert not any(issue.level == "crit" for issue in hr.issues)


def test_auto_measurement_signature_changes_when_component_split_changes() -> None:
    data_a = {
        "bass_integration_enable": True,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=24)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=24)),
    }
    data_b = {
        "bass_integration_enable": True,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=12)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=12)),
    }

    bundle_a, f_l_a, m_l_a, p_l_a, f_r_a, m_r_a, p_r_a = load_bass_integration_measurements(data_a)
    bundle_b, f_l_b, m_l_b, p_l_b, f_r_b, m_r_b, p_r_b = load_bass_integration_measurements(data_b)

    assert bundle_a is not None
    assert bundle_b is not None

    sig_a = _auto_measurement_signature(
        {
            "f_l": f_l_a,
            "m_l": m_l_a,
            "p_l": p_l_a,
            "f_r": f_r_a,
            "m_r": m_r_a,
            "p_r": p_r_a,
            "bass_integration_enabled": True,
            "bass_integration_bundle": bundle_a,
            "avr_crossover_hz": 80.0,
            "bass_integration_profile": "safe",
            "bass_integration_mode": "avr_lfe_main_decomposed",
        }
    )
    sig_b = _auto_measurement_signature(
        {
            "f_l": f_l_b,
            "m_l": m_l_b,
            "p_l": p_l_b,
            "f_r": f_r_b,
            "m_r": m_r_b,
            "p_r": p_r_b,
            "bass_integration_enabled": True,
            "bass_integration_bundle": bundle_b,
            "avr_crossover_hz": 80.0,
            "bass_integration_profile": "safe",
            "bass_integration_mode": "avr_lfe_main_decomposed",
        }
    )

    assert sig_a != sig_b


def test_run_pipeline_generates_direct_dac_sub_ir_from_sub_bundle(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, 1.0 + 0.0j, label="r_main")
    l_sub = _flat_transfer(freqs, 2.0 + 0.0j, label="sub_1")
    r_sub = _flat_transfer(freqs, 3.0 + 0.0j, label="sub_2")
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_flat_transfer(freqs, 6.0 + 0.0j, label="l_total"),
        r_total=_flat_transfer(freqs, 6.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    seen_mags: list[np.ndarray] = []

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        seen_mags.append(np.asarray(mags_in, dtype=float))
        imp = np.zeros(32, dtype=float)
        imp[0] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": 0.0,
        }

    monkeypatch.setattr("camillafir.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=80.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
    )
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "bass_integration_bundle": bundle,
            "bass_integration_enabled": True,
            "bass_integration_mode": "direct_dac",
            "ui_data": {"comparison_mode": False},
        },
    )

    assert result.sub_ir is not None
    assert len(seen_mags) == 3
    np.testing.assert_allclose(
        seen_mags[-1],
        20.0 * np.log10(np.full(freqs.shape, 5.0, dtype=float)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.l_st["direct_dac_sum_measured_mags"], dtype=float),
        20.0 * np.log10(np.full(freqs.shape, 6.0, dtype=float)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.l_st["direct_dac_sum_predicted_mags"], dtype=float),
        20.0 * np.log10(np.full(freqs.shape, 6.0, dtype=float)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.r_st["direct_dac_sum_predicted_mags"], dtype=float),
        20.0 * np.log10(np.full(freqs.shape, 6.0, dtype=float)),
        atol=1e-6,
    )


def test_run_pipeline_aligns_direct_dac_sub_ir_to_main_ir_delay(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    calls = {"n": 0}

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        calls["n"] += 1
        delay_samples = 10.0 if calls["n"] <= 2 else 2.0
        peak_idx = int(round(delay_samples))
        imp = np.zeros(32, dtype=float)
        imp[peak_idx] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": delay_samples,
        }

    monkeypatch.setattr("camillafir.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=80.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
    )
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "ui_data": {"comparison_mode": False},
        },
        include_response_arrays=False,
    )

    assert result.sub_ir is not None
    assert int(np.argmax(np.abs(result.l_ir))) == 10
    assert int(np.argmax(np.abs(result.r_ir))) == 10
    assert int(np.argmax(np.abs(result.sub_ir))) == 10


def _rolloff_transfer(
    freqs_hz: np.ndarray,
    *,
    f3db: float,
    order: int = 4,
    label: str,
) -> TransferData:
    """TransferData with a Butterworth-like low-pass rolloff for the main speaker."""
    freqs = np.asarray(freqs_hz, dtype=float)
    # Simple Butterworth magnitude: 1 / sqrt(1 + (f/f3db)^(2*order))
    # The speaker is flat in the passband and rolls off below f3db.
    # We model the speaker as a HIGH-pass (passes high freqs, rolls off low).
    hp_mag = (freqs / np.maximum(f3db, 1e-9)) ** order / np.sqrt(
        1.0 + (freqs / np.maximum(f3db, 1e-9)) ** (2 * order)
    )
    hp_mag = np.clip(hp_mag, 1e-12, None)
    spec = hp_mag.astype(np.complex128)  # zero phase, real-positive
    mag_db = 20.0 * np.log10(hp_mag)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=mag_db,
        phase_deg=np.zeros_like(freqs),
        sample_rate=48000,
        label=label,
    )


def test_recommend_avr_crossover_avoids_40hz_when_main_rolls_off_at_80hz() -> None:
    """When the main speaker rolls off around 80 Hz the recommendation must not
    pick 40 Hz (where the main has no output).  Before the main_activity fix
    the scoring always selected 40 Hz because cancellation_risk ≈ 0 there."""
    from camillafir.dsp.bass_integration import recommend_avr_crossover

    freqs = np.logspace(np.log10(10.0), np.log10(20000.0), 800)

    # Main speaker: rolls off below ~80 Hz (4th-order high-pass at 80 Hz)
    main = _rolloff_transfer(freqs, f3db=80.0, order=4, label="main")
    # Subwoofer: flat across full range
    sub = _flat_transfer(freqs, complex_gain=1.0 + 0j, label="sub")
    # Zero right-side sub (single sub scenario)
    sub_zero = _flat_transfer(freqs, complex_gain=0j, label="sub_r_zero")

    # Build a synthetic total (main + sub)
    total_spec = main.complex_spec + sub.complex_spec
    total_mag_db = 20.0 * np.log10(np.maximum(np.abs(total_spec), 1e-12))
    total = TransferData(
        freqs_hz=freqs,
        complex_spec=total_spec,
        mag_db=total_mag_db,
        phase_deg=np.zeros_like(freqs),
        sample_rate=48000,
        label="total",
    )

    bundle = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=sub_zero,
        l_total=total,
        r_total=total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    result = recommend_avr_crossover(bundle, profile="safe")
    rec_hz = float(result["recommended_hz"])
    scores = result["scores"]

    # The recommended XO must not be 40 Hz — main speaker is silent there
    assert rec_hz >= 60.0, f"Expected XO >= 60 Hz, got {rec_hz} Hz"

    # At 40 Hz, main speaker activity drop must be large
    drop_40 = float(scores[40.0]["main_activity_drop_db"])
    assert drop_40 >= 10.0, f"Expected main_activity_drop_db >= 10 dB at 40 Hz, got {drop_40:.1f}"


def test_recommend_direct_dac_crossover_scores_filtered_acoustic_sum() -> None:
    from camillafir.dsp.bass_integration import recommend_direct_dac_crossover

    freqs = np.logspace(np.log10(10.0), np.log10(20000.0), 1200)
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    # 6.25 ms delay makes the 80 Hz trial sum acoustically poor after HPF/LPF
    # branch filtering, while 40 Hz remains smooth.
    sub = _delayed_transfer(freqs, 0.00625, label="sub")
    sub_zero = _flat_transfer(freqs, 0.0 + 0.0j, label="sub_r_zero")

    total_spec = main.complex_spec + sub.complex_spec
    total = TransferData(
        freqs_hz=freqs,
        complex_spec=total_spec,
        mag_db=20.0 * np.log10(np.maximum(np.abs(total_spec), 1e-12)),
        phase_deg=np.rad2deg(np.unwrap(np.angle(total_spec))),
        sample_rate=48000,
        label="total",
    )
    bundle = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=sub_zero,
        l_total=total,
        r_total=total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    result = recommend_direct_dac_crossover(
        bundle,
        candidates=(40.0, 80.0, 120.0),
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert float(result["recommended_hz"]) == 40.0
    assert float(result["scores"][80.0]["overlap_ripple_db"]) > (
        float(result["scores"][40.0]["overlap_ripple_db"]) + 8.0
    )


def test_recommend_direct_dac_crossover_uses_decimal_grid_by_default(monkeypatch) -> None:
    import camillafir.dsp.bass_integration as bass_integration

    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    sub = _flat_transfer(freqs, 1.0 + 0.0j, label="sub")
    silent = _flat_transfer(freqs, 0.0 + 0.0j, label="silent")
    bundle = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=silent,
        l_total=main,
        r_total=main,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    def _fake_diag(_bundle, fc_hz, _profile, **_kwargs):
        delta = abs(float(fc_hz) - 82.5)
        return {
            "cancellation_risk": delta / 100.0,
            "overlap_ripple_db": delta,
            "sub_dominance_db": 0.0,
        }

    monkeypatch.setattr(
        bass_integration,
        "compute_direct_dac_bass_integration_diagnostics",
        _fake_diag,
    )
    monkeypatch.setattr(
        bass_integration,
        "_main_guard_band_drop_db",
        lambda _main, _fc_hz: 0.0,
    )

    result = bass_integration.recommend_direct_dac_crossover(bundle, profile="safe")

    assert float(result["recommended_hz"]) == 82.5
    assert 82.5 in result["scores"]
    assert float(result["scores"][82.5]["score"]) > float(result["scores"][80.0]["score"])


def test_direct_dac_auto_hpf_seed_source_uses_sub_bundle_data() -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, 1.0 + 0.0j, label="r_main")
    l_sub = _flat_transfer(freqs, 2.0 + 0.0j, label="l_sub")
    r_sub = _flat_transfer(freqs, 3.0 + 0.0j, label="r_sub")
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_flat_transfer(freqs, 6.0 + 0.0j, label="l_total"),
        r_total=_flat_transfer(freqs, 6.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    out = _resolve_auto_hpf_seed_source(
        {"bass_integration_bundle": bundle},
        {"bass_integration_enable": True, "bass_integration_mode": "direct_dac"},
        np.asarray([20.0, 40.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([20.0, 40.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
    )

    f_l_src, m_l_src, f_r_src, m_r_src, freq_key, slope_key, label, user_hpf_enabled = out
    np.testing.assert_allclose(f_l_src, l_sub.freqs_hz)
    np.testing.assert_allclose(m_l_src, l_sub.mag_db)
    np.testing.assert_allclose(f_r_src, r_sub.freqs_hz)
    np.testing.assert_allclose(m_r_src, r_sub.mag_db)
    assert freq_key == "sub_hpf_freq"
    assert slope_key == "sub_hpf_slope"
    assert label == "Sub HPF"
    assert user_hpf_enabled is True


def test_direct_dac_summary_formats_decimal_recommendation() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 82.5,
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 82.5,
                "recommended_crossover_hz": 82.5,
                "profile": "safe",
                "inputs": {},
            },
        },
    )

    assert "Main/Sub XO: 82.5 Hz" in summary
    assert "Recommended Main/Sub XO: 82.5 Hz" in summary


def test_build_auto_selected_text_reports_sub_hpf_in_direct_dac() -> None:
    txt = _build_auto_selected_text(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_hpf_freq": 18.5,
            "sub_hpf_slope": 24,
            "hpf_enable": False,
            "hpf_freq": 44.0,
            "hpf_slope": 12,
            "mag_c_min": 28.0,
            "filter_type": "Asymmetric",
            "phase_limit": 500.0,
            "target_curve_name": "Harman8",
            "best_metrics": {},
        }
    )

    assert "Sub HPF 18.5 Hz/24 dB/oct" in txt
    assert "HPF 44.0 Hz/12 dB/oct" not in txt
