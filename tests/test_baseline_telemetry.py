import numpy as np

from camillafir.dsp.correction_types import (
    BaselineComparisonTelemetry,
    BaselineNativeTelemetry,
    apply_baseline_telemetry_to_stats,
)


def test_apply_baseline_telemetry_to_stats_populates_legacy_keys():
    st = {}
    cmp = {}
    native = BaselineNativeTelemetry(
        analysis_mode="native",
        freq_axis=np.array([20.0, 40.0, 80.0], dtype=float),
        measured_mags=np.array([1.0, 2.0, 3.0], dtype=float),
        target_mags=np.array([0.0, 0.5, 1.0], dtype=float),
        target_env_lo=np.array([-1.0, -0.5, 0.0], dtype=float),
        target_env_hi=np.array([1.0, 1.5, 2.0], dtype=float),
        target_env_pivot_hz=63.0,
        target_shift_db=2.5,
        eff_target_db=74.0,
        target_level_db_window=73.5,
        meas_level_db_window=75.0,
        offset_db=1.0,
        offset_method="Median",
        smart_scan_range=(200.0, 3000.0),
    )
    comparison = BaselineComparisonTelemetry(
        analysis_mode="comparison",
        target_mags=np.array([0.0, 0.5, 1.0], dtype=float),
        measured_mags=np.array([1.0, 2.0, 3.0], dtype=float),
        filter_mags=np.array([0.2, 0.1, 0.0], dtype=float),
        eff_target_db=74.0,
        offset_db=1.25,
        smart_scan_range=(180.0, 2800.0),
        meas_level_db_window=75.0,
        target_level_db_window=73.5,
        offset_method="Median",
        target_shift_db=2.5,
    )

    apply_baseline_telemetry_to_stats(
        st=st,
        cmp=cmp,
        native=native,
        comparison=comparison,
    )

    assert st["analysis_mode"] == "native"
    assert st["freq_axis"] == [20.0, 40.0, 80.0]
    assert st["target_mags"] == [0.0, 0.5, 1.0]
    assert st["target_env_pivot_hz"] == 63.0
    assert st["smart_scan_range"] == [200.0, 3000.0]
    assert cmp["analysis_mode"] == "comparison"
    assert cmp["cmp_target_mags"] == [0.0, 0.5, 1.0]
    assert cmp["cmp_filter_mags"] == [0.2, 0.1, 0.0]
    assert cmp["cmp_smart_scan_range"] == [180.0, 2800.0]


def test_apply_baseline_telemetry_skips_optional_target_envelope_when_missing():
    st = {}
    apply_baseline_telemetry_to_stats(
        st=st,
        cmp=None,
        native=BaselineNativeTelemetry(
            analysis_mode="native",
            freq_axis=np.array([20.0, 40.0], dtype=float),
            measured_mags=np.array([1.0, 2.0], dtype=float),
            target_mags=np.array([0.0, 0.5], dtype=float),
            target_env_lo=None,
            target_env_hi=None,
            target_env_pivot_hz=None,
            target_shift_db=0.0,
            eff_target_db=74.0,
            target_level_db_window=74.0,
            meas_level_db_window=74.0,
            offset_db=0.0,
            offset_method="Median",
            smart_scan_range=(200.0, 2000.0),
        ),
        comparison=None,
    )

    assert "target_env_lo" not in st
    assert "target_env_hi" not in st
    assert "target_env_pivot_hz" not in st
