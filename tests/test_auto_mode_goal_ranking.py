from camillafir.io.camillafir_automatic_mode import (
    AutoModeConfig,
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_optuna,
    _build_auto_mode_candidates_local,
    _auto_run_optuna_eval_loop,
    _auto_exc_penalty_bins_from_dbg,
    _auto_exc_zero_penalty_freq_hz_from_stats,
    _auto_goal,
    _auto_hybrid_mixed_freq_penalty,
    _auto_optuna_sampler_kwargs,
    _auto_optimizer_backend,
    _auto_rank_key,
    _auto_rank_key_goal,
    _auto_reject,
    _auto_trial_workers,
)
from types import SimpleNamespace


def test_auto_goal_defaults_to_balanced():
    assert _auto_goal({}) == "balanced"
    assert _auto_goal({"auto_goal": "unknown"}) == "balanced"
    assert _auto_goal({"auto_goal": "hybrid"}) == "low-ripple"


def test_auto_mode_config_from_base_data_reads_overrides():
    cfg = AutoModeConfig.from_base_data(
        {
            "auto_mode_trials": 77,
            "auto_mode_phase1_plateau_rounds": 4,
            "auto_mode_local_refine_enabled": False,
            "auto_mode_local_refine_top_k": 3,
            "auto_mode_local_refine_trials_per_top": 9,
            "auto_mode_optuna": True,
            "auto_mode_optuna_min_trials": 40,
            "auto_mode_optuna_startup_trials": 11,
            "auto_mode_optuna_multivariate": False,
            "auto_mode_optuna_group": True,
            "auto_mode_optuna_constant_liar": False,
        }
    )
    assert cfg.trials == 77
    assert cfg.phase1_plateau_rounds == 4
    assert cfg.local_refine_enabled is False
    assert cfg.local_refine_top_k == 3
    assert cfg.local_refine_trials_per_top == 9
    assert cfg.optuna_pilot_enabled is True
    assert cfg.optuna_pilot_min_trials == 40
    assert cfg.optuna_pilot_startup_trials == 11
    assert cfg.optuna_multivariate is False
    assert cfg.optuna_group is True
    assert cfg.optuna_constant_liar is False


def test_auto_optimizer_backend_selection(monkeypatch):
    monkeypatch.delenv("CAMILLAFIR_AUTO_MODE_OPTIMIZER", raising=False)
    assert _auto_optimizer_backend({"auto_mode_optimizer": "optuna"}) == "optuna"
    assert _auto_optimizer_backend({"auto_mode_optuna": True}) == "optuna"
    assert _auto_optimizer_backend({"auto_mode_optimizer": "builtin", "auto_mode_optuna": True}) == "builtin"

    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_OPTIMIZER", "builtin")
    assert _auto_optimizer_backend({"auto_mode_optimizer": "optuna", "auto_mode_optuna": True}) == "builtin"


def test_auto_optuna_sampler_kwargs_defaults_and_parallel_behavior():
    assert _auto_optuna_sampler_kwargs({}, workers=1) == {
        "multivariate": True,
        "group": False,
        "constant_liar": False,
    }
    assert _auto_optuna_sampler_kwargs({}, workers=4) == {
        "multivariate": True,
        "group": False,
        "constant_liar": True,
    }


def test_auto_optuna_sampler_kwargs_group_requires_multivariate():
    assert _auto_optuna_sampler_kwargs(
        {
            "auto_mode_optuna_multivariate": False,
            "auto_mode_optuna_group": True,
            "auto_mode_optuna_constant_liar": True,
        },
        workers=8,
    ) == {
        "multivariate": False,
        "group": False,
        "constant_liar": True,
    }


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


def test_build_auto_mode_candidates_varies_mag_and_low_cut():
    cands = _build_auto_mode_candidates(
        {
            "auto_goal": "balanced",
            "filter_type": "Asymmetric",
            "mag_c_min": 24.0,
            "low_bass_cut_hz": 32.0,
        },
        n_trials=24,
        seed=4321,
    )
    mags = {
        round(float(c.get("mag_c_min")), 1)
        for c in cands
        if isinstance(c, dict) and ("mag_c_min" in c)
    }
    lows = {
        round(float(c.get("low_bass_cut_hz")), 1)
        for c in cands
        if isinstance(c, dict) and ("low_bass_cut_hz" in c)
    }
    assert len(mags) > 1
    assert len(lows) > 1
    assert all(15.0 <= v <= 70.0 for v in mags)
    assert all(18.0 <= v <= 55.0 for v in lows)


def test_build_auto_mode_candidates_optuna_optional_backend():
    cands = _build_auto_mode_candidates_optuna(
        {
            "auto_goal": "balanced",
            "filter_type": "Mixed",
            "mag_c_min": 24.0,
            "low_bass_cut_hz": 32.0,
        },
        n_trials=8,
        seed=1234,
    )
    if cands is None:
        assert cands is None
        return
    assert isinstance(cands, list)
    assert len(cands) == 8


def test_auto_run_optuna_eval_loop_feeds_seed_trials_into_study():
    class _FakeTrial:
        def __init__(self, fixed=None):
            self.fixed = dict(fixed or {})

        def suggest_float(self, name, low, high):
            if name in self.fixed:
                return float(self.fixed[name])
            return float(low)

        def suggest_categorical(self, name, choices):
            if name in self.fixed:
                return self.fixed[name]
            return list(choices)[0]

    class _FakeStudy:
        def __init__(self):
            self.enqueued = []
            self.told = []

        def enqueue_trial(self, params):
            self.enqueued.append(dict(params or {}))

        def ask(self):
            if self.enqueued:
                return _FakeTrial(self.enqueued.pop(0))
            return _FakeTrial()

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    class _FakeTrialState:
        FAIL = "fail"

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

    seen = []

    def _build_preset(trial):
        return {
            "max_boost": float(trial.suggest_float("max_boost", 3.0, 8.0)),
            "reg_strength": float(trial.suggest_float("reg_strength", 15.0, 45.0)),
        }

    def _eval_one(idx, preset):
        seen.append((int(idx), dict(preset or {})))
        return {
            "idx": int(idx),
            "ok": True,
            "metrics": {"rank_score": float(80.0 + idx)},
        }

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=2,
        seed=123,
        startup_trials=2,
        base_data={},
        seed_presets=[{"max_boost": 6.7, "reg_strength": 22.5}],
        build_preset=_build_preset,
        eval_one=_eval_one,
        consume_one=lambda idx, out: False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=1,
        seed_to_params=lambda preset: {
            "max_boost": float(preset.get("max_boost", 4.0)),
            "reg_strength": float(preset.get("reg_strength", 30.0)),
        },
    )

    assert seen
    assert seen[0][1]["max_boost"] == 6.7
    assert seen[0][1]["reg_strength"] == 22.5


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
        assert 15.0 <= float(c.get("mag_c_min", 0.0)) <= 70.0
        assert 170.0 <= float(c.get("mag_c_max", 0.0)) <= 300.0
        assert 70.0 <= float(c.get("trans_width", 0.0)) <= 150.0
        assert 150.0 <= float(c.get("bass_first_mode_max_hz", 0.0)) <= 220.0
        assert 18.0 <= float(c.get("low_bass_cut_hz", 0.0)) <= 55.0


def test_build_auto_mode_candidates_local_varies_mag_and_low_cut():
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
    cands = _build_auto_mode_candidates_local(base, center, n_trials=16, seed=12345, shrink=0.60)
    mags = {round(float(c.get("mag_c_min", 0.0)), 1) for c in cands}
    lows = {round(float(c.get("low_bass_cut_hz", 0.0)), 1) for c in cands}
    assert len(mags) > 1
    assert len(lows) > 1


def test_auto_trial_workers_respects_auto_and_caps(monkeypatch):
    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_WORKERS", "")
    monkeypatch.setattr("camillafir.io.camillafir_automatic_mode.os.cpu_count", lambda: 8)
    assert _auto_trial_workers({"auto_mode_workers": 0}, 32) == 8
    assert _auto_trial_workers({"auto_mode_workers": 3}, 32) == 3
    assert _auto_trial_workers({"auto_mode_workers": 99}, 32) == 8


def test_auto_trial_workers_env_override_and_min_trials(monkeypatch):
    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_WORKERS", "2")
    monkeypatch.setattr("camillafir.io.camillafir_automatic_mode.os.cpu_count", lambda: 16)
    assert _auto_trial_workers({"auto_mode_workers": 8}, 32) == 2
    assert _auto_trial_workers({"auto_mode_workers": 8}, 4) == 1


def test_auto_exc_penalty_bins_from_dbg_prefers_pen_bins_field():
    assert _auto_exc_penalty_bins_from_dbg({"pen_bins": 1.25, "exc_bins": 99}) == 1.25
    assert abs(_auto_exc_penalty_bins_from_dbg({"exc_bins": 12}) - 1.2) < 1e-9


def test_auto_exc_zero_penalty_freq_from_stats_clips_to_limits():
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 10.0}) == 20.0
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 120.0}) == 80.0
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 42.5}) == 42.5


def test_auto_score_result_waives_exc_bins_using_zero_penalty_floor():
    st = {
        "exc_prot": True,
        "exc_freq": 24.0,
        "boost_candidate_bins_excprot": 6,
        "boost_candidate_min_hz": 18.2,
        "lf_boost_max_db": 0.0,
        "net_boost_peak_db": 0.0,
        "avg_confidence": 90.0,
        "freq_axis": [20.0, 30.0, 40.0],
        "measured_mags": [0.0, 0.0, 0.0],
        "target_mags": [0.0, 0.0, 0.0],
        "filter_mags": [0.0, 0.0, 0.0],
    }
    from camillafir.io.camillafir_automatic_mode import _auto_score_result

    out = _auto_score_result(
        SimpleNamespace(l_st=dict(st), r_st=dict(st)),
        auto_exc_freq_hz=24.0,
        base_data={},
    )
    assert out["auto_exc_zero_penalty_hz"] == 20.0
    assert out["exc_penalty_raw_total"] > 0.0
    assert out["exc_penalty_bins_raw"] > 0.0
    assert out["exc_penalty_bins_waived"] is True
    assert out["exc_penalty_raw"] == 0.0
