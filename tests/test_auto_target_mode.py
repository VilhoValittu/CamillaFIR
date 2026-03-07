from camillafir.config.camillafir_pipeline import collect_ui_data
from camillafir.io.camillafir_automatic_mode import _auto_select_target_curve_with_trials
from camillafir.io.camillafir_automatic_mode import _run_auto_mode_search


def test_collect_ui_data_normalizes_auto_target_mode_to_auto_by_default():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "invalid-value",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "auto"


def test_collect_ui_data_accepts_selected_aliases_for_auto_target_mode():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "manual",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "selected"


def test_auto_target_curve_selection_uses_exact_signature_cache_without_recomputing(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: {
            "best_target_curve": "Harman6",
            "best_preset": {"preset_id": "measurement"},
        },
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_cache_get_best_target",
        lambda *args, **kwargs: "Harman8",
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_cache_get_best",
        lambda *args, **kwargs: {"preset_id": "signature"},
    )

    def _unexpected_quick_preselect(*args, **kwargs):
        raise AssertionError("quick target preselect should be skipped on exact cache hit")

    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_select_builtin_target_curve",
        _unexpected_quick_preselect,
    )

    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selected_hc_mode")) == "Harman8"
    assert str(result.get("selection_method")) == "cache_signature_hit"
    assert int(result.get("top_n", -1)) == 0
    assert int(result.get("trials_per_curve", -1)) == 0
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "signature"


def test_auto_mode_search_uses_exact_signature_cache_without_trials(monkeypatch):
    from types import SimpleNamespace

    trial_counter = {"n": 0}

    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: {
            "best_preset": {
                "phase_limit": 432.1,
                "tdc_strength": 54.5,
                "preset_id": "exact-cache",
                "_auto_exc_freq_hz": 31.5,
            },
            "best_metrics": {"rank_score": 87.5, "avg_score": 81.7},
        },
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_cache_get_best",
        lambda *args, **kwargs: {
            "phase_limit": 432.1,
            "tdc_strength": 54.5,
            "preset_id": "exact-cache",
            "_auto_exc_freq_hz": 31.5,
        },
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.build_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.run_pipeline",
        lambda *args, **kwargs: (
            trial_counter.__setitem__("n", int(trial_counter["n"]) + 1) or
            SimpleNamespace(l_st={}, r_st={}, metrics={})
        ),
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.summarize_run",
        lambda result: "cached-summary",
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_score_result",
        lambda *args, **kwargs: {"rank_score": 87.5, "avg_score": 81.7},
    )

    def _unexpected_candidate_build(*args, **kwargs):
        raise AssertionError("preset search trials should be skipped on exact cache hit")

    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._build_auto_mode_candidates",
        _unexpected_candidate_build,
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
        },
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert int(result.get("trials_total", -1)) == 20
    assert int(result.get("trials_phase1_total", -1)) == 0
    assert int(result.get("trials_phase2_total", -1)) == 20
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "exact-cache"
    assert dict(result.get("best_metrics", {}) or {}).get("rank_score") == 87.5
    assert dict(result.get("best_preset", {}) or {}).get("_auto_exc_freq_hz") == 31.5
    assert dict(result.get("best_preset", {}) or {}).get("exc_freq") == 31.5
    assert result.get("best_auto_exc_freq_hz") == 31.5
    assert int(trial_counter["n"]) == 21
    assert bool(result.get("phase2_plateau_hit", False)) is True
