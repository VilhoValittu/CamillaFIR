from camillafir.io.camillafir_automatic_mode import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _auto_goal,
    _auto_hybrid_mixed_freq_penalty,
    _auto_rank_key,
    _auto_rank_key_goal,
    _auto_reject,
)


def test_auto_goal_defaults_to_balanced():
    assert _auto_goal({}) == "balanced"
    assert _auto_goal({"auto_goal": "unknown"}) == "balanced"
    assert _auto_goal({"auto_goal": "hybrid"}) == "low-ripple"


def test_auto_rank_key_goal_balanced_matches_legacy():
    metrics = {
        "rank_score": 81.2,
        "avg_score": 86.5,
        "max_net_boost_db": 4.1,
        "events_severity": 0.7,
        "events_total": 2,
        "lr_delta_score": 0.4,
        "dsp_penalty_raw": 3.0,
        "exc_penalty_raw": 1.2,
    }
    assert _auto_rank_key_goal(metrics, goal="balanced") == _auto_rank_key(metrics)
    assert _auto_rank_key_goal(metrics, goal="not-a-goal") == _auto_rank_key(metrics)


def test_auto_rank_key_goal_acoustic_prefers_avg_score_first():
    high_rank_low_avg = {
        "rank_score": 88.0,
        "avg_score": 76.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    lower_rank_higher_avg = {
        "rank_score": 82.0,
        "avg_score": 79.0,
        "lr_delta_score": 0.7,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }

    assert _auto_rank_key_goal(high_rank_low_avg, goal="balanced") < _auto_rank_key_goal(
        lower_rank_higher_avg, goal="balanced"
    )
    assert _auto_rank_key_goal(lower_rank_higher_avg, goal="acoustic") < _auto_rank_key_goal(
        high_rank_low_avg, goal="acoustic"
    )


def test_auto_reject_is_hard_guard_for_acoustic_only():
    metrics = {"max_net_boost_db": 8.5}
    st = {"pre_energy_metric_suspect": False, "ir_pre_post_ratio": 0.02}
    assert _auto_reject(metrics, st, st, goal="acoustic")
    assert not _auto_reject(metrics, st, st, goal="balanced")


def test_auto_rank_key_goal_hybrid_keeps_rank_score_primary():
    better_rank_worse_avg = {
        "rank_score": 85.0,
        "avg_score": 74.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    tied_rank_better_avg = {
        "rank_score": 85.0,
        "avg_score": 76.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    lower_rank_best_avg = {
        "rank_score": 84.0,
        "avg_score": 90.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }

    assert _auto_rank_key_goal(tied_rank_better_avg, goal="hybrid") < _auto_rank_key_goal(
        better_rank_worse_avg, goal="hybrid"
    )
    assert _auto_rank_key_goal(better_rank_worse_avg, goal="hybrid") < _auto_rank_key_goal(
        lower_rank_best_avg, goal="hybrid"
    )

    low_mixed_pen = dict(tied_rank_better_avg)
    high_mixed_pen = dict(tied_rank_better_avg)
    low_mixed_pen["mixed_freq_penalty"] = 0.0
    high_mixed_pen["mixed_freq_penalty"] = 1.0
    assert _auto_rank_key_goal(low_mixed_pen, goal="hybrid") < _auto_rank_key_goal(
        high_mixed_pen, goal="hybrid"
    )


def test_hybrid_mixed_freq_penalty_is_soft_tiebreak():
    base = {"filter_type": "Mixed", "bass_first_ai": True}
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 95.0}, base_data=base, goal="hybrid") == 0.0
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 150.0}, base_data=base, goal="hybrid") > 0.0
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 150.0}, base_data=base, goal="balanced") == 0.0


def test_build_auto_mode_candidates_hybrid_has_tdc_floor():
    cands = _build_auto_mode_candidates(
        {
            "auto_goal": "hybrid",
            "enable_tdc": True,
            "tdc_strength": 40.0,
            "filter_type": "Mixed",
        },
        n_trials=24,
        seed=1234,
    )
    vals = [float(c.get("tdc_strength", 0.0)) for c in cands if isinstance(c, dict)]
    assert vals
    assert min(vals) >= 55.0


def test_build_auto_mode_candidates_local_center_first_and_clamped():
    base = {
        "filter_type": "Mixed",
        "enable_tdc": True,
        "enable_afdw": True,
        "bass_first_ai": True,
        "mag_c_min": 26.0,
        "low_bass_cut_hz": 34.0,
    }
    center = {
        "mixed_freq": 92.0,
        "fdw_cycles": 10.0,
        "tdc_strength": 58.0,
        "tdc_max_reduction_db": 12.0,
        "tdc_slope_db_per_oct": 6.0,
        "reg_strength": 28.0,
        "max_boost": 4.0,
        "mag_c_max": 230.0,
        "trans_width": 105.0,
        "bass_first_mode_max_hz": 185.0,
    }
    cands = _build_auto_mode_candidates_local(base, center, n_trials=9, seed=12345, shrink=0.35)
    assert len(cands) == 9
    first = dict(cands[0])
    assert float(first.get("mixed_freq", 0.0)) == 92.0
    assert float(first.get("mag_c_min", 0.0)) == 26.0
    assert float(first.get("low_bass_cut_hz", 0.0)) == 34.0

    for c in cands:
        assert 80.0 <= float(c.get("mixed_freq", 0.0)) <= 320.0
        assert 8.0 <= float(c.get("fdw_cycles", 0.0)) <= 16.0
        assert 15.0 <= float(c.get("tdc_strength", 0.0)) <= 75.0
        assert 6.0 <= float(c.get("tdc_max_reduction_db", 0.0)) <= 36.0
        assert 15.0 <= float(c.get("reg_strength", 0.0)) <= 45.0
        assert 3.0 <= float(c.get("max_boost", 0.0)) <= 8.0
        assert 170.0 <= float(c.get("mag_c_max", 0.0)) <= 300.0
        assert 70.0 <= float(c.get("trans_width", 0.0)) <= 150.0
        assert 150.0 <= float(c.get("bass_first_mode_max_hz", 0.0)) <= 220.0
        assert float(c.get("mag_c_min", 0.0)) == 26.0
        assert float(c.get("low_bass_cut_hz", 0.0)) == 34.0
