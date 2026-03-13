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
