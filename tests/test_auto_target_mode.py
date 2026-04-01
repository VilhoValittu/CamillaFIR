from camillafir.config.camillafir_config import load_config
from camillafir.config.camillafir_pipeline import build_filter_config
from camillafir.config.camillafir_pipeline import collect_ui_data
from camillafir.config.models import FilterConfig
from camillafir.io.auto_mode.orchestrator_target import _target_eval_one
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


def test_collect_ui_data_normalizes_layout_from_stable_ui_keys():
    data = collect_ui_data({"layout": "stereo"})
    assert str(data.get("layout")) == "stereo"

    data = collect_ui_data({"layout": "mono"})
    assert str(data.get("layout")) == "mono"


def test_collect_ui_data_normalizes_legacy_layout_labels_to_stable_keys():
    data = collect_ui_data({"layout": "Stereo"})
    assert str(data.get("layout")) == "stereo"

    data = collect_ui_data({"layout": "Mono"})
    assert str(data.get("layout")) == "mono"


def test_collect_ui_data_normalizes_level_mode_and_algo_to_stable_keys():
    data = collect_ui_data({"mode": "ADVANCED", "lvl_mode": "Manual", "lvl_algo": "Average"})
    assert str(data.get("lvl_mode")) == "manual"
    assert str(data.get("lvl_algo")) == "average"


def test_collect_ui_data_auto_mode_preserves_allowed_inputs_but_forces_managed_settings(monkeypatch):
    monkeypatch.setattr(
        "camillafir.config.camillafir_pipeline.get_auto_mode_filter_auto_defaults",
        lambda filter_type: {
            "filter_type_str": str(filter_type),
            "phase_limit": 407.2,
            "enable_tdc": True,
            "enable_afdw": True,
            "filter_smooth": 12,
            "max_boost_db": 4.11,
            "mixed_split_freq": 177.3,
            "comparison_mode": True,
        },
    )

    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "filter_type": "Mixed Phase",
        "auto_goal": "low-ripple",
        "auto_target_mode": "selected",
        "hc_mode": "Cinema",
        "fs": 96000,
        "taps": 131072,
        "multi_rate_opt": [True],
        "gain": 9.5,
        "phase_limit": 999.0,
        "enable_tdc": [],
        "enable_afdw": [],
        "filter_smooth": 96,
        "max_boost": 12.0,
        "comparison_mode": [],
        "hpf_enable": [True],
        "hpf_freq": 27.5,
        "hpf_slope": 18,
        "xo1_f": 80.0,
        "xo1_s": 24,
        "xo2_f": 2200.0,
        "xo2_s": 12,
    }

    data = collect_ui_data(pin)

    assert str(data.get("filter_type")) == "Mixed Phase"
    assert str(data.get("auto_goal")) == "low-ripple"
    assert str(data.get("auto_target_mode")) == "selected"
    assert str(data.get("hc_mode")) == "Cinema"
    assert int(data.get("fs")) == 96000
    assert int(data.get("taps")) == 131072
    assert bool(data.get("multi_rate_opt")) is True
    assert float(data.get("phase_limit")) == 407.2
    assert bool(data.get("enable_tdc")) is True
    assert bool(data.get("enable_afdw")) is True
    assert int(data.get("filter_smooth")) == 12
    assert float(data.get("max_boost")) == 4.11
    assert float(data.get("mixed_freq")) == 177.3
    assert bool(data.get("comparison_mode")) is True
    assert bool(data.get("hpf_enable")) is True
    assert float(data.get("hpf_freq")) == 27.5
    assert int(data.get("hpf_slope")) == 18
    assert float(data.get("gain")) == 0.0
    assert float(data.get("xo1_f")) == 80.0
    assert int(data.get("xo1_s")) == 24
    assert float(data.get("xo2_f")) == 2200.0
    assert int(data.get("xo2_s")) == 12


def test_build_filter_config_auto_mode_does_not_crash_and_uses_locked_data_values():
    class _Pin:
        def __init__(self):
            self._d = {
                "enable_tdc": [],
                "enable_afdw": [],
                "filter_smooth": 96,
                "df_smoothing": [True],
            }

        def get(self, key, default=None):
            return self._d.get(key, default)

        def __getitem__(self, key):
            if key in self._d:
                return self._d[key]
            raise KeyError(key)

    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": "Linear Phase",
            "mixed_freq": 180.0,
            "mag_c_min": 15.0,
            "mag_c_max": 196.1,
            "max_boost": 3.22,
            "phase_limit": 448.0,
            "mag_correct": True,
            "reg_strength": 22.5,
            "normalize_opt": False,
            "exc_prot": True,
            "exc_freq": 24.0,
            "low_bass_cut_hz": 18.0,
            "low_bass_cut_enable": True,
            "ir_window_right": 500.0,
            "ir_window_left": 85.0,
            "lvl_manual_db": 0.0,
            "lvl_min": 200.0,
            "lvl_max": 3000.0,
            "lvl_algo": "Median",
            "trans_width": 139.2,
            "enable_tdc": True,
            "enable_afdw": True,
            "tdc_strength": 63.9,
            "tdc_max_reduction_db": 23.7,
            "tdc_slope_db_per_oct": 12.0,
            "filter_smooth": 12,
            "df_smoothing": False,
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
        pin=_Pin(),
    )

    assert bool(getattr(cfg, "enable_tdc", False)) is True
    assert bool(getattr(cfg, "enable_afdw", False)) is True
    assert int(getattr(cfg, "filter_smooth", 0)) == 12
    assert bool(getattr(cfg, "df_smoothing", True)) is False


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
        lambda ui_data, preset=None, *, fs_v=None, taps_v=None, xos=None, hpf=None, hc_f=None, hc_m=None, filter_config_cls=None, max_safe_boost=8.0: SimpleNamespace(),
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
            "auto_mode_optuna": False,
            "auto_mode_optuna_persistent_study": False,
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
    assert int(dict(result.get("phase_limit_winner_polish", {}) or {}).get("tested_count", -1)) == 4
    assert int(trial_counter["n"]) == 26
    assert bool(result.get("phase2_plateau_hit", False)) is True


def test_auto_mode_search_replays_exact_cache_results_into_optuna_study(monkeypatch):
    from types import SimpleNamespace

    remembered = []

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
        lambda ui_data, preset=None, *, fs_v=None, taps_v=None, xos=None, hpf=None, hc_f=None, hc_m=None, filter_config_cls=None, max_safe_boost=8.0: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.run_pipeline",
        lambda *args, **kwargs: SimpleNamespace(l_st={}, r_st={}, metrics={}),
    )

    def _score_result(result, **kwargs):
        base = dict(kwargs.get("base_data", {}) or {})
        preset_id = str(base.get("preset_id", "exact-cache"))
        if preset_id == "micro-1":
            return {"rank_score": 87.7, "avg_score": 81.6}
        return {"rank_score": 87.5, "avg_score": 81.7}

    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode.summarize_run",
        lambda result: "cached-summary",
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_score_result",
        _score_result,
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._build_auto_mode_candidates_micro",
        lambda *args, **kwargs: [
            {"phase_limit": 432.1, "tdc_strength": 54.5, "preset_id": "exact-cache", "_auto_exc_freq_hz": 31.5},
            {"phase_limit": 430.0, "tdc_strength": 53.0, "preset_id": "micro-1", "_auto_exc_freq_hz": 31.5},
        ],
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_import_optuna",
        lambda: object(),
    )
    monkeypatch.setattr(
        "camillafir.io.camillafir_automatic_mode._auto_optuna_remember_result",
        lambda *args, **kwargs: remembered.append(dict(kwargs or {})) or True,
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_mode_optuna": True,
            "auto_mode_optuna_persistent_study": True,
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
    assert len(remembered) >= 2
    assert any(str(call.get("study_name", "")).find("phase3-micro") >= 0 for call in remembered)
    assert any(str(call.get("study_name", "")).find("phase1") >= 0 for call in remembered)
    assert any(dict(call.get("preset", {}) or {}).get("preset_id") == "micro-1" for call in remembered)
    assert any(dict(call.get("preset", {}) or {}).get("preset_id") == "exact-cache" for call in remembered)


def test_target_eval_one_uses_current_build_config_signature_without_pin():
    from types import SimpleNamespace

    calls = []

    def _build_config(
        ui_data,
        preset=None,
        *,
        fs_v=None,
        taps_v=None,
        xos=None,
        hpf=None,
        hc_f=None,
        hc_m=None,
        filter_config_cls=None,
        max_safe_boost=8.0,
    ):
        calls.append(
            {
                "fs_v": fs_v,
                "taps_v": taps_v,
                "max_safe_boost": max_safe_boost,
                "ui_data": dict(ui_data or {}),
            }
        )
        return SimpleNamespace()

    runtime = SimpleNamespace(
        build_config=_build_config,
        run_pipeline=lambda cfg, measurements, include_response_arrays=False: SimpleNamespace(
            metrics={},
            ui_data=dict(measurements.get("ui_data", {}) or {}),
        ),
        auto_score_result=lambda result, **kwargs: {"rank_score": 84.0, "avg_score": 80.0},
    )

    out = _target_eval_one(
        runtime=runtime,
        preset={"phase_limit": 420.0},
        base_tc={"filter_type": "Linear Phase", "comparison_mode": True},
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "p_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
            "p_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f_arr=[20.0, 100.0],
        hc_m_arr=[0.0, 0.0],
        pin_obj=object(),
        filter_key="linear",
    )

    assert out["ok"] is True
    assert out["metrics"]["rank_score"] == 84.0
    assert len(calls) == 1
    assert int(calls[0]["fs_v"]) == 44100
