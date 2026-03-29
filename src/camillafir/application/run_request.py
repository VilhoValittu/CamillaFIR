from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRequest:
    raw_ui_data: dict[str, Any] = field(default_factory=dict)
    run_started_at: float | None = None
