from types import SimpleNamespace

import pytest

from camillafir.io.auto_mode.runtime_context import (
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
)
from camillafir.io.auto_mode.materialize import (
    AutoModeMaterializeContext,
    build_materialize_helpers,
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


def test_mag_c_min_winner_polish_reuses_matching_candidate_metrics():
    calls = {"materialize": 0}

    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = preset, include_response_arrays, summarize, base_data_override
        calls["materialize"] += 1
        raise AssertionError("materialize_preset_result should not run for reused exact candidate")

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
        best_preset={"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5},
        best_metrics={"rank_score": 79.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 25.0},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=1.0,
        max_down_hz=0.0,
        max_up_hz=1.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
        candidate_items=[
            {
                "preset": {"mag_c_min": 26.0, "_auto_exc_freq_hz": 31.5},
                "metrics": {"rank_score": 80.0, "avg_score": 90.0},
                "phase": "phase 2/2 local center#1",
            }
        ],
    )

    assert bool(improved) is True
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(26.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert int(calls["materialize"]) == 0
    assert [float(v) for v in list(meta.get("tested_mag_c_min_hz", []) or [])] == [26.0]


def test_materialize_score_only_cache_reuses_exact_preset():
    calls = {
        "build_config": 0,
        "run_pipeline": 0,
        "auto_score_result": 0,
        "summarize_run": 0,
    }

    def fake_build_config(*args, **kwargs):
        _ = args, kwargs
        calls["build_config"] += 1
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, *, include_response_arrays):
        _ = cfg, measurements, include_response_arrays
        calls["run_pipeline"] += 1
        return SimpleNamespace(metrics={}, l_st={}, r_st={})

    def fake_auto_score_result(result, *, auto_exc_freq_hz, base_data):
        _ = result, auto_exc_freq_hz, base_data
        calls["auto_score_result"] += 1
        return {"rank_score": 80.0, "avg_score": 90.0}

    def fake_summarize_run(result):
        _ = result
        calls["summarize_run"] += 1
        return "summary"

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"phase_limit": 400.0, "mag_c_min": 25.0},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        filter_key="linear",
        max_safe_boost=6.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=fake_auto_score_result,
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=lambda metrics: float(dict(metrics or {}).get("rank_score", 0.0)),
        auto_is_better_refine_fn=lambda *args, **kwargs: False,
        build_config_fn=fake_build_config,
        run_pipeline_fn=fake_run_pipeline,
        summarize_run_fn=fake_summarize_run,
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=3,
        residual_rank_eps=0.35,
    )
    _cache_ready_preset, materialize_preset_result, _preset_signature, _maybe_apply = build_materialize_helpers(ctx)
    _ = _cache_ready_preset, _preset_signature, _maybe_apply

    preset = {"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5}
    result_a, metrics_a, data_a = materialize_preset_result(
        preset,
        include_response_arrays=False,
        summarize=False,
    )
    result_b, metrics_b, data_b = materialize_preset_result(
        dict(preset),
        include_response_arrays=False,
        summarize=False,
    )

    assert result_a is not None
    assert result_b is not None
    assert metrics_b == metrics_a
    assert data_b == data_a
    assert calls == {
        "build_config": 1,
        "run_pipeline": 1,
        "auto_score_result": 1,
        "summarize_run": 0,
    }


def test_materialize_score_only_cache_does_not_replace_full_materialization():
    calls = {
        "build_config": 0,
        "run_pipeline": 0,
        "auto_score_result": 0,
        "summarize_run": 0,
    }

    def fake_build_config(*args, **kwargs):
        _ = args, kwargs
        calls["build_config"] += 1
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, *, include_response_arrays):
        _ = cfg, measurements
        calls["run_pipeline"] += 1
        return SimpleNamespace(metrics={"include_response_arrays": bool(include_response_arrays)}, l_st={}, r_st={})

    def fake_auto_score_result(result, *, auto_exc_freq_hz, base_data):
        _ = result, auto_exc_freq_hz, base_data
        calls["auto_score_result"] += 1
        return {"rank_score": 80.0, "avg_score": 90.0}

    def fake_summarize_run(result):
        _ = result
        calls["summarize_run"] += 1
        return "summary"

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"phase_limit": 400.0, "mag_c_min": 25.0},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        filter_key="linear",
        max_safe_boost=6.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=fake_auto_score_result,
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=lambda metrics: float(dict(metrics or {}).get("rank_score", 0.0)),
        auto_is_better_refine_fn=lambda *args, **kwargs: False,
        build_config_fn=fake_build_config,
        run_pipeline_fn=fake_run_pipeline,
        summarize_run_fn=fake_summarize_run,
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=3,
        residual_rank_eps=0.35,
    )
    _cache_ready_preset, materialize_preset_result, _preset_signature, _maybe_apply = build_materialize_helpers(ctx)
    _ = _cache_ready_preset, _preset_signature, _maybe_apply

    preset = {"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5}
    materialize_preset_result(
        preset,
        include_response_arrays=False,
        summarize=False,
    )
    result_full, metrics_full, data_full = materialize_preset_result(
        dict(preset),
        include_response_arrays=True,
        summarize=True,
    )

    assert result_full.metrics.get("summary") == "summary"
    assert metrics_full == {"rank_score": 80.0, "avg_score": 90.0}
    assert float(data_full.get("mag_c_min", 0.0)) == pytest.approx(25.0, abs=1e-9)
    assert calls == {
        "build_config": 2,
        "run_pipeline": 2,
        "auto_score_result": 2,
        "summarize_run": 1,
    }
