from __future__ import annotations

import time
import typing
from dataclasses import dataclass

from ..ui.ng_bridge import ProcessRunUiBridge
from .auto_flow import (
    _run_auto_mode_search_if_needed,
    _run_auto_mode_seed_phases,
)
from .pipeline_flow import _run_pipeline
from .run_finalize import _finalize_run_outputs
from .run_prepare import (
    _prepare_target_curve_and_run_context,
    _prepare_ui_and_measurements,
)


@dataclass(frozen=True)
class ProcessRunSupport:
    version: str
    max_safe_boost: float
    force_single_plot_fs_hz: int
    auto_target_mode_norm: typing.Callable[[typing.Any], str]
    auto_target_selection_method_text: typing.Callable[[typing.Any], str]
    pick_target_curve_label: typing.Callable[[dict], str]
    slugify_filename_token: typing.Callable[..., str]
    has_uploaded_target_file: typing.Callable[[dict], bool]
    ui_bridge: ProcessRunUiBridge


def run_process_flow(*, pin_obj, support: ProcessRunSupport):
    run_started_at = time.perf_counter()
    callbacks = support.ui_bridge.make_callbacks(run_started_at)

    ctx = _prepare_ui_and_measurements(
        pin_obj=pin_obj,
        callbacks=callbacks,
        support=support,
        run_started_at=run_started_at,
    )
    if ctx is None:
        return

    _run_auto_mode_seed_phases(
        ctx,
        pin_obj=pin_obj,
        callbacks=callbacks,
        support=support,
    )
    _prepare_target_curve_and_run_context(
        ctx,
        pin_obj=pin_obj,
        support=support,
    )
    _run_auto_mode_search_if_needed(
        ctx,
        pin_obj=pin_obj,
        callbacks=callbacks,
        support=support,
    )
    if not _run_pipeline(
        ctx,
        pin_obj=pin_obj,
        callbacks=callbacks,
        support=support,
    ):
        return
    _finalize_run_outputs(
        ctx,
        callbacks=callbacks,
        support=support,
    )


__all__ = ["ProcessRunSupport", "run_process_flow"]
