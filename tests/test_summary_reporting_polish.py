import numpy as np

from camillafir.ui import camillafir_plot as plots


def test_format_band_rt60_summary_deduplicates_duplicate_display_band():
    bands = {
        63.0: 0.40,
        126.0: 0.20,
        252.0: 0.10,
        400.0: 0.07,
        2016.0: 0.13,
    }

    out = plots.format_band_rt60_summary(bands)

    assert out.count("400Hz:0.07s") == 1


def test_format_band_rt60_summary_preserves_first_occurrence_order():
    bands = {
        63.0: 0.40,
        126.0: 0.20,
        252.0: 0.10,
        400.0: 0.07,
        2016.0: 0.13,
    }

    out = plots.format_band_rt60_summary(bands)

    assert out == "63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s"


def test_format_summary_content_uses_same_rt60_dedupe_for_left_and_right():
    stats = {
        "rt60_val": 0.30,
        "rt60_band_avg": 0.22,
        "rt60_bands": {
            63.0: 0.40,
            126.0: 0.20,
            252.0: 0.10,
            400.0: 0.07,
            2016.0: 0.13,
        },
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
    }

    summary = plots.format_summary_content({}, stats, stats)

    assert "Band RT60 L: 63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s" in summary
    assert "Band RT60 R: 63Hz:0.40s | 126Hz:0.20s | 252Hz:0.10s | 400Hz:0.07s | 2016Hz:0.13s" in summary


def test_dsp_quality_report_uses_aligned_measurement_axis_for_mag_error():
    freqs = np.geomspace(20.0, 20000.0, 32)
    measured_aligned = np.full(freqs.shape, -1.0)
    measured_raw = measured_aligned + 92.98
    target = np.zeros(freqs.shape)
    gain = np.ones(freqs.shape)
    stats = {
        "freq_axis": freqs.tolist(),
        "target_mags": target.tolist(),
        "measured_mags": measured_aligned.tolist(),
        "measured_mags_raw": measured_raw.tolist(),
        "predicted_filter_mags": gain.tolist(),
        "realized_filter_mags": gain.tolist(),
        "mag_mask": np.ones(freqs.shape).tolist(),
        "offset_db": 92.98,
        "phase_limit": 125.0,
    }

    report = "\n".join(plots.format_dsp_quality_report_block({}, stats, stats))

    assert "Predicted mag error RMS within mag_c band:      L 0.00 dB | R 0.00 dB" in report
    assert "Realized mag error RMS within mag_c band:       L 0.00 dB | R 0.00 dB" in report
    assert "186." not in report


def test_format_summary_content_warns_on_large_leveling_offset():
    stats = {
        "rt60_val": 0.30,
        "cmp_avg_confidence": 80.0,
        "avg_confidence": 80.0,
        "reflections": [],
        "offset_db": 45.0,
    }

    summary = plots.format_summary_content({}, stats, stats)

    assert "Warning: Very large leveling offset detected" in summary
