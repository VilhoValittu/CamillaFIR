from camillafir.workflow.auto_flow import (
    _build_auto_finalize_status,
    _estimate_auto_progress_from_status,
)
from camillafir.auto_mode.orchestrator_finalize import _build_phase2_pareto_status


def test_auto_progress_advances_across_auto_search_phases():
    init_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: init (goal balanced, basis preset_objective_score, filter mixed, taps 65536)"
    )
    target_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: target preselect init (top-3, 10 trials/curve, fs 44100 Hz, taps 65536, -6 dB point 16.0 Hz, goal balanced)"
    )
    search_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: preset search init (phase1 100 + refine 24 trials @ 44100 Hz, -6 dB point 16.0 Hz, goal balanced, basis preset_objective_score, target Adaptive)"
    )
    phase1_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode [Adaptive] (-6 dB 16.0 Hz, low-cut 18.0 Hz, exc seed 24.0 Hz, hpf off): phase 1/2 50/100 (rank 90.000, ok 48/50)"
    )
    phase2_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode [Adaptive] (-6 dB 16.0 Hz, low-cut 18.0 Hz, exc seed 24.0 Hz, hpf off): phase 2/2 local center#2 6/12 (rank 92.000, ok 6/6)"
    )
    pareto_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: phase 2 pareto selected winner (rank_best 91.831 -> pareto 88.510, avg 84.335 -> 84.352, prepost 0.2200 -> 0.0900, mode_ripple 1.303 dB -> 0.780 dB)"
    )
    finalize_progress = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: finalize (winner rank 93.500/100, avg 88.300, boost 1.20 dB, events 0, via phase 2 pareto)"
    )

    assert init_progress is not None
    assert target_progress is not None
    assert search_progress is not None
    assert phase1_progress is not None
    assert phase2_progress is not None
    assert pareto_progress is not None
    assert finalize_progress is not None
    assert init_progress < target_progress < search_progress < phase1_progress < phase2_progress < pareto_progress < finalize_progress


def test_auto_progress_tracks_target_trials_inside_shortlist():
    first_trial = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: target trials (target 1/3 Harman6, trial 1/10, rank 88.100)"
    )
    late_same_target = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: target trials best improved (target 1/3 Harman6, trial 8/10, -6 dB point 16.0 Hz, goal balanced, rank 89.400, avg 84.200, fit 0.320, pre 0.410)"
    )
    next_target = _estimate_auto_progress_from_status(
        "CamillaFIR automatic mode: target trials (target 2/3 Studio, trial 1/10, rank 89.600)"
    )

    assert first_trial is not None
    assert late_same_target is not None
    assert next_target is not None
    assert first_trial < late_same_target < next_target


def test_auto_finalize_status_calls_out_special_winner_stage():
    text = _build_auto_finalize_status(
        {
            "rank_score": 88.510,
            "rank_score_official": 88.510,
            "avg_score": 84.352,
            "max_net_boost_db": -0.01,
            "events_total": 30,
        },
        winner_explanation={"phase_label": "phase 2 pareto"},
    )

    assert "winner rank 88.510/100" in text
    assert "via phase 2 pareto" in text


def test_phase2_pareto_status_explains_rank_tradeoff():
    text = _build_phase2_pareto_status(
        rank_best_metrics={
            "rank_score": 91.831,
            "avg_score": 84.335,
            "ir_pre_post_energy_ratio_max": 0.2200,
            "mode_ripple_db": 1.303,
        },
        winner_metrics={
            "rank_score": 88.510,
            "avg_score": 84.352,
            "ir_pre_post_energy_ratio_max": 0.0900,
            "mode_ripple_db": 0.780,
        },
    )

    assert "phase 2 pareto selected winner" in text
    assert "rank_best 91.831 -> pareto 88.510" in text
    assert "prepost 0.2200 -> 0.0900" in text
