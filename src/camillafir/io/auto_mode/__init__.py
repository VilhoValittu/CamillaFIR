"""Backward-compatible shims for the relocated auto_mode package.

These imports exist only to preserve legacy entrypoints. Do not add new
business logic here; new functionality belongs in `camillafir.auto_mode`.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType

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

__all__ = list(_SUBMODULES)


class _LazyAliasModule(ModuleType):
    def __init__(self, alias_name: str, target_name: str) -> None:
        super().__init__(alias_name)
        self.__dict__["_alias_target_name"] = target_name

    def _load(self) -> ModuleType:
        module = import_module(self.__dict__["_alias_target_name"], __name__)
        sys.modules[self.__name__] = module
        globals()[self.__name__.rsplit(".", 1)[-1]] = module
        return module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


for _name in _SUBMODULES:
    _alias_name = f"{__name__}.{_name}"
    if _alias_name not in sys.modules:
        sys.modules[_alias_name] = _LazyAliasModule(
            _alias_name,
            f"...auto_mode.{_name}",
        )


def __getattr__(name: str):
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = sys.modules[f"{__name__}.{name}"]
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


del _alias_name
del _name
