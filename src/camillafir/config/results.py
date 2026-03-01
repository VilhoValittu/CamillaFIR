from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FilterResult:
    """
    Normalized single-run output container for DSP, export and UI layers.

    A single instance represents one sample-rate run.
    """

    fs: int
    taps: int
    l_ir: np.ndarray
    r_ir: np.ndarray
    l_mag: np.ndarray
    r_mag: np.ndarray
    l_phase: np.ndarray
    r_phase: np.ndarray
    freq_axis: np.ndarray
    l_st: dict
    r_st: dict
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Optional payloads for export/report/plot consumers.
    measurements: dict[str, np.ndarray] = field(default_factory=dict)
    cfg: Any = None
