import numpy as np

from camillafir.config.results import FilterResult
from camillafir.ui import camillafir_export as export


def _result(fs: int, left_score: float, right_score: float, boost_db: float = 0.0, events: int = 0) -> FilterResult:
    zeros = np.zeros(8, dtype=float)
    refs = [{} for _ in range(int(events))]
    return FilterResult(
        fs=fs,
        taps=2048,
        l_ir=zeros,
        r_ir=zeros,
        l_mag=zeros,
        r_mag=zeros,
        l_phase=zeros,
        r_phase=zeros,
        freq_axis=zeros,
        l_st={"score": float(left_score), "net_boost_peak_db": float(boost_db), "reflections": list(refs)},
        r_st={"score": float(right_score), "net_boost_peak_db": float(boost_db), "reflections": list(refs)},
    )


def test_export_winner_summary_uses_official_rank_score():
    data = {
        "camillafir_automatic_mode": True,
        "_auto_mode_meta": {
            "trials_ok": 4,
            "trials_total": 6,
            "search_fs": 44100,
            "search_taps": 2048,
            "best_metrics": {
                "rank_score": 69.505,
                "rank_score_official": 91.264,
                "rank_score_components": {"rank_score": 91.264, "context": {"score_kind": "preset_objective_score"}},
                "avg_score": 81.700,
                "dsp_penalty": 0.80,
                "exc_penalty": 0.20,
                "max_net_boost_db": 2.40,
                "events_total": 1,
                "events_severity": 0.25,
            },
            "best_preset": {},
        },
    }

    text = export._append_dsp_effective_params("", data, 44100)

    assert export._export_winner_rank_score(data) == 91.264
    assert "Best rank score: 91.264/100" in text
    assert "Best rank score: 69.505/100" not in text


def test_export_run_ranking_uses_distinct_label(monkeypatch):
    monkeypatch.setattr(
        export.plots,
        "calc_ai_summary_from_stats",
        lambda stats: {"score": float(dict(stats or {}).get("score", 0.0))},
    )

    ranking = export._build_export_ranking(
        [
            _result(44100, 82.0, 81.0, boost_db=0.5, events=0),
            _result(48000, 78.0, 77.5, boost_db=3.5, events=2),
        ]
    )

    winner = dict(ranking["entries"][0])
    block = export._append_export_ranking("", 44100, ranking)

    assert winner["run_ranking_score"] == winner["run_ranking_score_components"]["rank_score"]
    assert "run_ranking_score=" in block
    assert "rank_score=" not in block
    assert "separate from automatic-mode Best rank score" in block
