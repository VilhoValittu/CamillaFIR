from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransferData:
    freqs_hz: np.ndarray
    complex_spec: np.ndarray
    mag_db: np.ndarray
    phase_deg: np.ndarray
    sample_rate: int
    label: str = ""


@dataclass(frozen=True)
class BassIntegrationBundle:
    l_main: TransferData
    r_main: TransferData
    l_sub: TransferData
    r_sub: TransferData
    l_total: TransferData
    r_total: TransferData
    avr_crossover_hz: float
    profile: str = "safe"
    diagnostics: dict[str, Any] = field(default_factory=dict)
