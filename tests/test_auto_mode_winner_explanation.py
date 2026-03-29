from types import SimpleNamespace

import pytest

from camillafir.io import camillafir_automatic_mode as auto


def test_auto_build_winner_explanation_fallback_on_none():
    explanation = auto._auto_build_winner_explanation(
        None,
        phase_label="phase 1/2",
        target_name="Harman6",
    )
    assert explanation == {
        "summary": "Winner explanation unavailable.",
        "reasons": [],
        "deltas": {},
        "phase_label": "phase 1/2",
        "target_name": "Harman6",
    }


def test_auto_build_winner_explanation_summary_for_normal_metrics():
    explanation = auto._auto_build_winner_explanation(
        {
            "rank_score": 83.2,
            "avg_score": 80.4,
            "mode_ripple_db": 0.05,
            "max_net_boost_db": 2.6,
            "event_penalty": 0.0,
        },
        phase_label="phase 1/2",
        target_name="Harman6",
    )
    assert explanation["summary"]
    assert explanation["summary"] != "Winner explanation unavailable."
    assert explanation["reasons"]
    assert explanation["phase_label"] == "phase 1/2"
    assert explanation["target_name"] == "Harman6"


def test_auto_build_winner_explanation_includes_deltas():
    explanation = auto._auto_build_winner_explanation(
        {
            "rank_score": 84.0,
            "avg_score": 81.5,
            "mode_ripple_db": 0.04,
            "max_net_boost_db": 2.8,
            "event_penalty": 0.10,
        },
        {
            "rank_score": 82.5,
            "avg_score": 80.0,
            "mode_ripple_db": 0.07,
            "max_net_boost_db": 3.4,
            "event_penalty": 0.20,
        },
    )
    assert explanation["deltas"]["avg_score_delta"] == pytest.approx(1.5)
    assert explanation["deltas"]["rank_score_delta"] == pytest.approx(1.5)
    assert explanation["deltas"]["mode_ripple_delta"] == pytest.approx(-0.03)
    assert explanation["deltas"]["boost_delta"] == pytest.approx(-0.6)
    assert explanation["deltas"]["event_penalty_delta"] == pytest.approx(-0.1)


def test_auto_build_winner_explanation_avoids_wrong_direction_improvement_wording():
    explanation = auto._auto_build_winner_explanation(
        {
            "rank_score": 80.0,
            "avg_score": 78.5,
            "mode_ripple_db": 0.09,
            "max_net_boost_db": 4.6,
            "event_penalty": 0.30,
        },
        {
            "rank_score": 81.0,
            "avg_score": 79.5,
            "mode_ripple_db": 0.05,
            "max_net_boost_db": 4.0,
            "event_penalty": 0.10,
        },
    )
    rendered = " ".join([explanation["summary"]] + list(explanation["reasons"]))
    assert "improved average score" not in rendered.lower()
    assert "reduced mode ripple" not in rendered.lower()
    assert "reduced event penalty" not in rendered.lower()


def test_run_auto_mode_search_impl_returns_winner_explanation(monkeypatch):
    def fake_build_candidates(*args, **kwargs):
        return [
            {"preset_id": "a", "mixed_freq": 180.0},
            {"preset_id": "b", "mixed_freq": 160.0},
        ]

    def fake_build_config(
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
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, include_response_arrays=False):
        return SimpleNamespace(metrics={}, ui_data=dict(measurements.get("ui_data", {}) or {}))

    def fake_score_result(result, **kwargs):
        trial = dict(kwargs.get("base_data", {}) or {})
        preset_id = str(trial.get("preset_id", "a"))
        if preset_id == "b":
            return {
                "rank_score": 84.0,
                "avg_score": 81.0,
                "mode_ripple_db": 0.04,
                "max_net_boost_db": 2.7,
                "event_penalty": 0.10,
            }
        return {
            "rank_score": 81.0,
            "avg_score": 79.0,
            "mode_ripple_db": 0.08,
            "max_net_boost_db": 3.4,
            "event_penalty": 0.25,
        }

    monkeypatch.setattr(auto, "_build_auto_mode_candidates", fake_build_candidates)
    monkeypatch.setattr(auto, "_auto_trial_workers", lambda *args, **kwargs: 1)
    monkeypatch.setattr(auto, "build_config", fake_build_config)
    monkeypatch.setattr(auto, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(auto, "_auto_score_result", fake_score_result)
    monkeypatch.setattr(auto, "summarize_run", lambda result: "summary")

    result = auto._run_auto_mode_search_impl(
        base_data={
            "filter_type": "Mixed",
            "hc_mode": "Harman6",
            "auto_mode_cache_enabled": False,
            "auto_mode_local_refine_enabled": False,
            "auto_mode_phase3_micro_enabled": False,
            "auto_mode_workers": 1,
        },
        measurements={"ui_data": {}},
        fs_v=44100,
        taps_v=2048,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=2,
    )

    assert isinstance(result, dict)
    explanation = dict(result.get("winner_explanation", {}) or {})
    assert explanation["summary"]
    assert explanation["summary"] != "Winner explanation unavailable."
    assert explanation["target_name"] == "Harman6"


def test_run_auto_mode_search_impl_uses_optuna_backend_even_for_small_trial_count(monkeypatch):
    class _FakeTrialState:
        FAIL = "fail"

    class _FakeStudy:
        def __init__(self):
            self.told = []

        def ask(self):
            return object()

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    class _FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(direction=None, sampler=None):
            return _FakeStudy()

    def fake_build_candidates(*args, **kwargs):
        n_trials = int(kwargs.get("n_trials", 0) or 0)
        if n_trials <= 1:
            return [{"preset_id": "seed", "mixed_freq": 180.0}]
        raise AssertionError("builtin candidate sampler should not be used for full phase1 when optuna is enabled")

    def fake_build_config(
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
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, include_response_arrays=False):
        return SimpleNamespace(metrics={}, ui_data=dict(measurements.get("ui_data", {}) or {}))

    def fake_score_result(result, **kwargs):
        trial = dict(kwargs.get("base_data", {}) or {})
        preset_id = str(trial.get("preset_id", "seed"))
        if preset_id == "optuna":
            return {
                "rank_score": 88.0,
                "avg_score": 84.0,
                "mode_ripple_db": 0.03,
                "max_net_boost_db": 2.1,
                "event_penalty": 0.05,
            }
        return {
            "rank_score": 79.0,
            "avg_score": 76.0,
            "mode_ripple_db": 0.08,
            "max_net_boost_db": 3.8,
            "event_penalty": 0.30,
        }

    monkeypatch.setattr(auto, "_auto_import_optuna", lambda: _FakeOptuna)
    monkeypatch.setattr(auto, "_build_auto_mode_candidates", fake_build_candidates)
    monkeypatch.setattr(auto, "_suggest_auto_mode_candidate_optuna", lambda *args, **kwargs: {"preset_id": "optuna", "mixed_freq": 160.0})
    monkeypatch.setattr(auto, "_auto_trial_workers", lambda *args, **kwargs: 1)
    monkeypatch.setattr(auto, "build_config", fake_build_config)
    monkeypatch.setattr(auto, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(auto, "_auto_score_result", fake_score_result)
    monkeypatch.setattr(auto, "summarize_run", lambda result: "summary")

    result = auto._run_auto_mode_search_impl(
        base_data={
            "filter_type": "Mixed",
            "hc_mode": "Harman6",
            "auto_mode_cache_enabled": False,
            "auto_mode_local_refine_enabled": False,
            "auto_mode_phase3_micro_enabled": False,
            "auto_mode_workers": 1,
            "auto_mode_optuna": True,
        },
        measurements={"ui_data": {}},
        fs_v=44100,
        taps_v=2048,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=2,
    )

    assert isinstance(result, dict)
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "optuna"
