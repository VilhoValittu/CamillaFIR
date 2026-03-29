from __future__ import annotations

import logging
import time
import typing

from ..engine import build_config, run_pipeline, summarize_run
from ..resources.i8n.camillafir_i18n import t
from ..ui.camillafir_utils import scale_taps_with_fs
from ..ui.ng_bridge import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from .process_run_flow import ProcessRunSupport

logger = logging.getLogger("CamillaFIR")

_PIPELINE_PROGRESS_START = 0.20
_PIPELINE_PROGRESS_START_AUTO = 0.60
_PIPELINE_PROGRESS_END = 0.80


def _run_pipeline(ctx: dict, *, callbacks: ProcessRunCallbacks, support: ProcessRunSupport) -> bool:
    data = ctx["data"]
    measurements = ctx["measurements"]
    target_rates = ctx["target_rates"]
    xos = ctx["xos"]
    hpf = ctx["hpf"]
    hc_f = ctx["hc_f"]
    hc_m = ctx["hc_m"]
    taps_base = int(ctx["taps_base"])
    dash_fs = int(ctx["dash_fs"])
    results_by_fs = ctx["results_by_fs"]
    perf_stats = ctx["perf_stats"]
    per_fs_stats = ctx["per_fs_stats"]
    pipeline_start = (
        _PIPELINE_PROGRESS_START_AUTO
        if bool(ctx.get("auto_mode_enabled", False))
        else _PIPELINE_PROGRESS_START
    )
    pipeline_span = float(max(0.0, _PIPELINE_PROGRESS_END - pipeline_start))

    for index, fs_v in enumerate(target_rates):
        if bool(data.get("multi_rate_opt", False)):
            taps_v = scale_taps_with_fs(fs_v, base_taps=taps_base)
            logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
        else:
            taps_v = taps_base

        callbacks.status(f"{t('stat_calc')} {fs_v}Hz...")
        support.ui_bridge.set_progress(
            float(pipeline_start) + float(pipeline_span) * (float(index) / float(max(1, len(target_rates))))
        )
        data["enable_residual_pass"] = bool(data.get("enable_residual_pass", False))

        cfg = build_config(
            data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            max_safe_boost=float(support.max_safe_boost),
        )
        try:
            setattr(cfg, "bass_smooth_w_gamma", float(data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg, "bass_smooth_w_max", float(data.get("bass_smooth_w_max", 0.45)))
        except Exception:
            pass

        callbacks.status(f"{t('stat_calc')} {int(fs_v)} Hz")
        dsp_started_at = time.perf_counter()
        result = run_pipeline(cfg, measurements)
        result.metrics["summary"] = summarize_run(result)
        dsp_elapsed = max(0.0, float(time.perf_counter() - dsp_started_at))

        perf_stats["dsp_s"] += dsp_elapsed
        fs_key = int(fs_v)
        slot = per_fs_stats.setdefault(fs_key, {})
        slot["dsp_s"] = float(slot.get("dsp_s", 0.0)) + dsp_elapsed
        results_by_fs.append(result)

        if int(fs_v) == dash_fs:
            ctx["l_st_f"] = result.l_st
            ctx["r_st_f"] = result.r_st
            ctx["l_imp_f"] = result.l_ir
            ctx["r_imp_f"] = result.r_ir

    if not results_by_fs:
        support.ui_bridge.toast_measurement_files_missing()
        return False
    return True
