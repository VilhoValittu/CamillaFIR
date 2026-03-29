from types import SimpleNamespace

import pytest

from camillafir.io.auto_mode.runtime_context import (
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
)
from camillafir.io.auto_mode.scoring_metrics import _auto_focus_ripple_from_stats
from camillafir.io.auto_mode.winner_polish import apply_mag_c_min_winner_polish


def test_focus_ripple_tracks_corrected_response_error_not_filter_realization_delta():
    freq_axis = [20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0]
    measured = [3.0] * len(freq_axis)
    target = [0.0] * len(freq_axis)
    realized_filter = [-3.0] * len(freq_axis)
    predicted_filter = [-1.0] * len(freq_axis)

    ripple = _auto_focus_ripple_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": measured,
            "target_mags": target,
            "realized_filter_mags": realized_filter,
            "predicted_filter_mags": predicted_filter,
            "confidence_mask": [1.0] * len(freq_axis),
        },
        focus_lo_hz=20.0,
        focus_hi_hz=160.0,
    )

    assert ripple is not None
    assert float(ripple) == pytest.approx(0.0, abs=1e-9)


def test_mode_detection_includes_sub_bass_resonance_with_moderate_gd():
    result = SimpleNamespace(
        l_st={
            "reflections": [
                {"type": "resonance", "freq": 28.0, "gd_error": 140.0},
            ]
        },
        r_st={},
    )

    top_modes = _auto_get_top_modes_hz(result, top_n=1)

    assert top_modes == pytest.approx([28.0], abs=1e-9)
    assert _auto_get_worst_mode_hz(result) == pytest.approx(28.0, abs=1e-9)


def test_mode_band_is_not_clamped_by_bass_first_mode_limit():
    band = _auto_mode_band(
        240.0,
        base_data={
            "bass_first_ai": True,
            "bass_first_mode_max_hz": 180.0,
        },
    )

    assert band is not None
    assert float(band[0]) < 200.0
    assert float(band[1]) > 220.0


def test_mag_c_min_winner_polish_can_improve_upward():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        mag_c_min = float(dict(preset or {}).get("mag_c_min", 25.0))
        rank_score = 80.0 - abs(mag_c_min - 26.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_mag_c_min_winner_polish(
        best_preset={"mag_c_min": 25.0},
        best_metrics={"rank_score": 79.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 25.0},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=1.0,
        max_down_hz=2.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(26.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert 26.0 in [float(v) for v in list(meta.get("tested_mag_c_min_hz", []) or [])]
