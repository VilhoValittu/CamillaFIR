import numpy as np

from camillafir.application.house_curve_service import (
    apply_manual_target_curve_tilt,
    load_house_curve,
)
from camillafir.config.camillafir_config import load_config
from camillafir.config.camillafir_pipeline import build_filter_config, collect_ui_data
from camillafir.config.models import FilterConfig
from camillafir.ui.plot_summary import format_summary_content


def test_apply_manual_target_curve_tilt_uses_1khz_pivot_with_bass_positive_sign():
    freq_axis = np.array([125.0, 1000.0, 8000.0], dtype=float)
    target_curve = np.array([0.0, 0.0, 0.0], dtype=float)

    tilted = apply_manual_target_curve_tilt(freq_axis, target_curve, 0.5)

    np.testing.assert_allclose(tilted, np.array([1.5, 0.0, -1.5], dtype=float))


def test_load_house_curve_applies_manual_target_tilt_only_in_manual_mode(monkeypatch):
    monkeypatch.setattr(
        "camillafir.application.house_curve_service.get_house_curve_by_name",
        lambda name: (
            np.array([0.0, 125.0, 1000.0, 8000.0], dtype=float),
            np.zeros(4, dtype=float),
        ),
    )

    manual_freqs, manual_mags, _ = load_house_curve(
        {
            "hc_mode": "Flat",
            "lvl_mode": "Manual",
            "manual_target_tilt_db_per_oct": 1.0,
        }
    )
    auto_freqs, auto_mags, _ = load_house_curve(
        {
            "hc_mode": "Flat",
            "lvl_mode": "Auto",
            "manual_target_tilt_db_per_oct": 1.0,
        }
    )

    np.testing.assert_allclose(manual_freqs, auto_freqs)
    np.testing.assert_allclose(manual_mags, np.array([3.0, 3.0, 0.0, -3.0], dtype=float))
    np.testing.assert_allclose(auto_mags, np.zeros(4, dtype=float))


def test_collect_ui_data_normalizes_manual_target_tilt_db_per_oct():
    data = collect_ui_data(
        {
            "mode": "ADVANCED",
            "lvl_mode": "Manual",
            "manual_target_tilt_db_per_oct": "1.25",
        }
    )

    assert float(data.get("manual_target_tilt_db_per_oct", 0.0)) == 1.25


def test_build_filter_config_keeps_manual_target_tilt_db_per_oct():
    data = load_config()
    data.update(
        {
            "mode": "ADVANCED",
            "filter_type": "Linear Phase",
            "mixed_freq": 180.0,
            "mag_c_min": 10.0,
            "mag_c_max": 200.0,
            "max_boost": 5.0,
            "phase_limit": 600.0,
            "mag_correct": True,
            "reg_strength": 30.0,
            "normalize_opt": False,
            "exc_prot": False,
            "exc_freq": 20.0,
            "low_bass_cut_hz": 40.0,
            "ir_window_right": 500.0,
            "ir_window_left": 85.0,
            "lvl_mode": "Manual",
            "lvl_manual_db": 0.0,
            "manual_target_tilt_db_per_oct": 0.8,
            "lvl_min": 200.0,
            "lvl_max": 3000.0,
            "lvl_algo": "Median",
            "trans_width": 100.0,
        }
    )

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=None,
    )

    assert float(getattr(cfg, "manual_target_tilt_db_per_oct", 0.0)) == 0.8


def test_format_summary_content_reports_manual_target_tilt():
    stats = {
        "rt60_val": 0.30,
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
    }

    summary = format_summary_content(
        {
            "lvl_mode": "Manual",
            "lvl_manual_db": 1.5,
            "manual_target_tilt_db_per_oct": 0.8,
        },
        stats,
        stats,
    )

    assert "Manual target level: +1.5 dB" in summary
    assert "Manual target tilt: +0.8 dB/oct @ 1 kHz" in summary
