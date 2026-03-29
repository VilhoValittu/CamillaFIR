from __future__ import annotations

import logging
import math
import time
import typing
from datetime import datetime

import numpy as np

from ..application.health_service import compute_health
from ..application.house_curve_service import load_house_curve
from ..application.run_request import RunRequest
from ..common.result_postprocess import _irwin_tag
from ..config.camillafir_config import save_config
from ..config.camillafir_pipeline import (
    build_xos_hpf,
    choose_dash_fs,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
    log_df_smoothing_toggle,
)
from ..io.measurements_loader import load_measurements_lr
from ..io.measurements_txt import parse_measurements_from_path
from ..resources.i8n.camillafir_i18n import t
from ..ui.ng_bridge import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from .process_run_flow import ProcessRunSupport

logger = logging.getLogger("CamillaFIR")


def _prepare_ui_and_measurements(
    *,
    request: RunRequest,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
) -> dict | None:
    perf_stats = {
        "read_s": 0.0,
        "dsp_s": 0.0,
        "zip_png_s": 0.0,
    }
    per_fs_stats: dict[int, dict[str, float]] = {}

    data = dict(request.raw_ui_data or {})
    run_started_at = float(request.run_started_at or time.perf_counter())
    callbacks.set_auto_selected_bar("")

    try:
        mode = str(data.get("mode") or "BASIC").strip().upper()
        hr = compute_health(data, mode)
        if support.ui_bridge.toast_health_gate_result(hr, mode):
            return None
    except Exception:
        pass

    ir_export_window_mode = data.get("ir_export_window_mode")
    if not isinstance(ir_export_window_mode, str) or ir_export_window_mode.strip() == "":
        data["ir_export_window_mode"] = "auto"
    logger.info(f"UI ir_export_window_mode={data.get('ir_export_window_mode')}")

    try:
        sh = str(data.get("ir_export_window_shape", "hann") or "hann").strip().lower()
    except Exception:
        sh = "hann"
    if sh not in ("hann", "tukey"):
        sh = "hann"
    data["ir_export_window_shape"] = sh

    try:
        alpha = float(data.get("ir_export_tukey_alpha", 0.25))
    except Exception:
        alpha = 0.25
    if not math.isfinite(alpha):
        alpha = 0.25
    data["ir_export_tukey_alpha"] = float(np.clip(alpha, 0.0, 1.0))

    try:
        if filter_type_short(str(data.get("filter_type", "") or "")) == "Asymmetric":
            data["ir_export_window_mode"] = "rew_asym"
            data["ir_window_mode"] = "rew_asym"
            data["ir_export_window_shape"] = "tukey"
            data["ir_export_tukey_alpha"] = 0.25
    except Exception:
        pass

    taps_base = int(float(data.get("taps", 65536) or 65536))
    save_config(data)

    support.ui_bridge.ensure_progress_bar()

    read_started_at = time.perf_counter()
    callbacks.status(t("stat_reading"))
    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    perf_stats["read_s"] += max(0.0, float(time.perf_counter() - read_started_at))
    if f_l is None or f_r is None:
        support.ui_bridge.toast_measurement_files_missing()
        return None

    return {
        "run_started_at": run_started_at,
        "perf_stats": perf_stats,
        "per_fs_stats": per_fs_stats,
        "data": data,
        "taps_base": taps_base,
        "f_l": f_l,
        "m_l": m_l,
        "p_l": p_l,
        "f_r": f_r,
        "m_r": m_r,
        "p_r": p_r,
    }


def _prepare_target_curve_and_run_context(
    ctx: dict,
    *,
    support: ProcessRunSupport,
):
    data = ctx["data"]
    taps_base = int(ctx["taps_base"])
    f_l = ctx["f_l"]
    m_l = ctx["m_l"]
    p_l = ctx["p_l"]
    f_r = ctx["f_r"]
    m_r = ctx["m_r"]
    p_r = ctx["p_r"]

    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path,
    )
    data["hc_source"] = hc_source
    target_curve_name = support.pick_target_curve_label(data)
    target_curve_tag = support.slugify_filename_token(target_curve_name, default="target")
    data["target_curve_name"] = target_curve_name
    data["target_curve_tag"] = target_curve_tag
    logger.info(f"House curve source: {hc_source}")
    logger.info(f"Export target curve tag: {target_curve_tag} (from '{target_curve_name}')")
    try:
        if hc_f is not None and hc_m is not None:
            logger.info(
                f"HC: n={len(hc_f)} f=[{hc_f[0]:.2f}..{hc_f[-1]:.2f}] "
                f"m=[{float(np.min(hc_m)):.2f}..{float(np.max(hc_m)):.2f}] mean={float(np.mean(hc_m)):.2f}"
            )
    except Exception:
        pass
    xos, hpf = build_xos_hpf(data)
    try:
        if xos:
            xo_txt = ", ".join(
                [
                    f"{float(x.get('freq')):.1f}Hz/{int(x.get('slope', int(x.get('order', 1)) * 6))}dB/oct"
                    for x in xos
                ]
            )
            logger.info(f"XO (UI->CFG): {xo_txt}")
        else:
            logger.info("XO (UI->CFG): off")
        if isinstance(hpf, dict) and hpf.get("enabled"):
            hf = float(hpf.get("freq", 0.0) or 0.0)
            ho = int(hpf.get("order", 0) or 0)
            logger.info(f"HPF (UI->CFG): {hf:.1f}Hz/{int(ho * 6)}dB/oct")
        else:
            logger.info("HPF (UI->CFG): off")
    except Exception:
        pass
    log_df_smoothing_toggle(data, logger)

    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(
        target_rates,
        multi_rate_on=multi_rate_on,
        forced_plot_fs_hz=int(support.force_single_plot_fs_hz),
    )
    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    auto_mode_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    zip_dashboards_on = False

    ts = datetime.now().strftime("%d%m%y_%H%M")
    file_ts = datetime.now().strftime("%H%M_%d%m%y")
    ft_short = filter_type_short(data["filter_type"])
    logger.info(
        f"EXPORT IR (UI): shape={data.get('ir_export_window_shape')}, "
        f"alpha={data.get('ir_export_tukey_alpha')}"
    )

    val_raw = data.get("ir_export_window_mode", None)
    if not isinstance(val_raw, str) or val_raw.strip() == "":
        val_raw = data.get("ir_window_mode", "auto")
    irw_mode = str(val_raw or "auto").strip().lower()
    if irw_mode not in ("auto", "off", "rew_sym", "rew_asym"):
        irw_mode = "auto"
    data["ir_export_window_mode"] = irw_mode
    irw_tag = _irwin_tag(irw_mode)

    is_wav_source = detect_is_wav_source(data)
    data["_is_wav_source"] = bool(is_wav_source)

    measurements = {
        "f_l": np.asarray(f_l, dtype=float),
        "m_l": np.asarray(m_l, dtype=float),
        "p_l": np.asarray(p_l, dtype=float),
        "f_r": np.asarray(f_r, dtype=float),
        "m_r": np.asarray(m_r, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "hc_f": hc_f,
        "hc_m": hc_m,
        "ui_data": data,
        "is_wav_source": bool(is_wav_source),
    }

    ctx.update(
        {
            "hc_f": hc_f,
            "hc_m": hc_m,
            "xos": xos,
            "hpf": hpf,
            "target_curve_tag": target_curve_tag,
            "target_rates": target_rates,
            "dash_fs": dash_fs,
            "auto_mode_enabled": auto_mode_enabled,
            "zip_dashboards_on": zip_dashboards_on,
            "ts": ts,
            "file_ts": file_ts,
            "ft_short": ft_short,
            "irw_tag": irw_tag,
            "measurements": measurements,
            "results_by_fs": [],
            "l_st_f": None,
            "r_st_f": None,
            "l_imp_f": None,
            "r_imp_f": None,
            "ui_dashboards": {},
        }
    )
