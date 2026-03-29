from __future__ import annotations

from typing import Any

from ..config.camillafir_pipeline import collect_ui_data

from .run_request import RunRequest


def build_run_request_from_pin(
    pin_obj: Any,
    *,
    version: str,
    auto_mode_compat_version: str,
) -> RunRequest:
    data = collect_ui_data(pin_obj)
    data["bass_adaptive_isolation_mode"] = True
    data["bass_smooth_sigma_scale"] = 1.20
    data["bass_smooth_conf_floor"] = 0.25
    data["bass_smooth_w_gamma"] = 2.40
    data["bass_smooth_w_max"] = 0.45
    data["program_version"] = str(version)
    data["auto_mode_compat_version"] = str(auto_mode_compat_version)
    return RunRequest(raw_ui_data=dict(data))
