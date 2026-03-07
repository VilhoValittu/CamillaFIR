from types import SimpleNamespace

import pytest

from camillafir.io import camillafir_automatic_mode as auto
from camillafir.ui import camillafir_ui as ui


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

    def fake_build_config(*args, **kwargs):
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


def test_render_auto_winner_explanation_handles_missing_explanation(monkeypatch):
    calls = []

    monkeypatch.setattr(ui, "put_markdown", lambda text: calls.append(("markdown", text)))
    monkeypatch.setattr(ui, "put_info", lambda text: calls.append(("info", text)))
    monkeypatch.setattr(ui, "put_html", lambda text: calls.append(("html", text)))

    ui._render_auto_winner_explanation({})

    assert ("markdown", "### Why this preset won") in calls
    assert ("info", "Winner explanation unavailable.") in calls
