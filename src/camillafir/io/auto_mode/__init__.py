"""Backward-compatible shims for the relocated auto_mode package."""

from importlib import import_module
import sys

_SUBMODULES = (
    "cache_signature",
    "candidate_generation",
    "filter_priors",
    "materialize",
    "optuna_backend",
    "orchestrator_finalize",
    "orchestrator_refine",
    "orchestrator_target",
    "protection_seed",
    "rank_score",
    "refine_eval",
    "runtime_context",
    "scoring_metrics",
    "scoring_ranking",
    "search_entrypoints",
    "search_state",
    "shared",
    "target_preselection",
    "winner_polish",
)

for _name in _SUBMODULES:
    _module = import_module(f"...auto_mode.{_name}", __name__)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

del _module
del _name

