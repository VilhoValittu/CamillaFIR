import logging
from dataclasses import dataclass, field

from .rank_score import attach_official_rank_score
from .scoring_ranking import _auto_build_winner_explanation

logger = logging.getLogger("CamillaFIR")


@dataclass
class _AutoModeSearchState:
    best_result: object | None = None
    best_metrics: dict | None = None
    best_preset: dict | None = None
    winner_explanation: dict | None = None
    scored: list[dict] = field(default_factory=list)
    phase2_pool: list[dict] = field(default_factory=list)


@dataclass
class _AutoModePhaseState:
    ok_n: int = 0
    tried_n: int = 0
    plateau_hit: bool = False
    no_improve_streak: int = 0
    improved_any: bool = False


def _auto_set_search_winner(
    search_state: _AutoModeSearchState,
    metrics: dict | None,
    preset: dict | None,
    *,
    prev_metrics: dict | None = None,
    phase_label: str | None = None,
    target_name: str | None = None,
) -> dict:
    search_state.best_metrics = attach_official_rank_score(metrics)
    search_state.best_preset = dict(preset or {})
    explanation = _auto_build_winner_explanation(
        search_state.best_metrics,
        prev_metrics,
        phase_label=phase_label,
        target_name=target_name,
    )
    search_state.winner_explanation = dict(explanation or {})
    logger.info(
        "Automatic mode winner explanation: %s",
        search_state.winner_explanation.get("summary", "n/a"),
    )
    return dict(search_state.winner_explanation)
