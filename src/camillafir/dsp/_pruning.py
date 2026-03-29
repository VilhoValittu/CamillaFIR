"""Thread-local pruning hook for Optuna trial pruning.

The hook is injected by the auto-mode eval loop before each trial and
consumed inside _run_generate_filter_pipeline after the magnitude
correction stage (before the more expensive phase-IR stage).

Usage:
    set_pruning_hook(fn)   # call before eval_one
    ...                    # eval_one runs the DSP pipeline
    clear_pruning_hook()   # call in finally block

The hook callable receives a single float (partial score, higher = better).
It should raise optuna.TrialPruned() if the trial should be stopped early.
"""
from __future__ import annotations

import threading

_local = threading.local()


def set_pruning_hook(fn) -> None:
    _local.hook = fn


def get_pruning_hook():
    return getattr(_local, "hook", None)


def clear_pruning_hook() -> None:
    _local.hook = None
