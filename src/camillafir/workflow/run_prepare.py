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
from ..io.measurements_loader import (
    load_bass_integration_measurements,
    load_measurements_lr,
    load_raw_irs_lr,
    load_raw_ir_sub,
)
from ..io.measurements_txt import parse_measurements_from_path
from ..resources.i8n.camillafir_i18n import t
from ..ui.ng_bridge import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from .process_run_flow import ProcessRunSupport

logger = logging.getLogger("CamillaFIR")


def _direct_dac_filter_params(data: dict) -> tuple[int, float, int]:
    try:
        xo_order = max(1, int(round(float(data.get("sub_crossover_slope", 24) or 24.0))) // 6)
    except Exception:
        xo_order = 4
    try:
        sub_hpf_hz = float(data.get("sub_hpf_freq", 20.0) or 20.0)
    except Exception:
        sub_hpf_hz = 20.0
    try:
        sub_hpf_order = max(1, int(round(float(data.get("sub_hpf_slope", 12) or 12.0))) // 6)
    except Exception:
        sub_hpf_order = 2
    return int(xo_order), float(sub_hpf_hz), int(sub_hpf_order)


def _compute_crossover_recommendation(bundle: object, data: dict) -> float | None:
    """Return recommended AVR crossover Hz from standard candidates, or None on failure."""
    try:
        if bundle is None:
            return None
        profile = str(data.get("bass_integration_profile", "safe") or "safe")
        bi_mode = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        ).strip().lower()
        if bi_mode == "direct_dac":
            from ..dsp.bass_integration import recommend_direct_dac_crossover  # noqa: PLC0415
            xo_order, sub_hpf_hz, sub_hpf_order = _direct_dac_filter_params(data)
            result = recommend_direct_dac_crossover(
                bundle,
                profile=profile,
                main_hpf_order=int(xo_order),
                sub_lpf_order=int(xo_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
            )
        else:
            from ..dsp.bass_integration import recommend_avr_crossover  # noqa: PLC0415
            result = recommend_avr_crossover(bundle, profile=profile)
        rec = result.get("recommended_hz", None)
        if rec is not None and math.isfinite(float(rec)):
            return float(rec)
    except Exception:
        logger.debug("crossover recommendation failed", exc_info=True)
    return None


def _compute_direct_dac_allpass_recommendation(bundle: object, data: dict) -> dict:
    try:
        if bundle is None:
            return {
                "enabled": False,
                "freq_hz": 0.0,
                "q": 0.707,
                "baseline": {},
                "optimized": {},
                "improvement_score": 0.0,
                "reason": "No meaningful improvement found.",
            }
        from ..dsp.bass_integration import recommend_direct_dac_allpass  # noqa: PLC0415

        profile = str(data.get("bass_integration_profile", "safe") or "safe")
        fc_hz = float(
            data.get(
                "sub_crossover_hz",
                data.get("avr_crossover_hz", 80.0),
            )
            or 80.0
        )
        xo_order, sub_hpf_hz, sub_hpf_order = _direct_dac_filter_params(data)
        return dict(
            recommend_direct_dac_allpass(
                bundle,
                fc_hz=float(fc_hz),
                profile=profile,
                main_hpf_order=int(xo_order),
                sub_lpf_order=int(xo_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
            )
            or {}
        )
    except Exception:
        logger.debug("Direct-DAC allpass recommendation failed", exc_info=True)
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": {},
            "optimized": {},
            "improvement_score": 0.0,
            "reason": "No meaningful improvement found.",
        }


def _compute_selected_bass_integration_diagnostics(bundle: object, data: dict) -> dict:
    try:
        if bundle is None:
            return {}
        bi_mode = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        ).strip().lower()
        profile = str(data.get("bass_integration_profile", "safe") or "safe")
        fc_hz = float(data.get("avr_crossover_hz", 80.0) or 80.0)
        if not math.isfinite(fc_hz) or fc_hz <= 0.0:
            fc_hz = 80.0
        if bi_mode == "direct_dac":
            from ..dsp.bass_integration import compute_direct_dac_bass_integration_diagnostics  # noqa: PLC0415
            xo_order, sub_hpf_hz, sub_hpf_order = _direct_dac_filter_params(data)
            sub_allpass_freq_hz = None
            sub_allpass_q = None
            if bool(data.get("bass_integration_allpass_auto_applied", False)):
                try:
                    sub_allpass_freq_hz = float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0)
                except Exception:
                    sub_allpass_freq_hz = 0.0
                try:
                    sub_allpass_q = float(data.get("bass_integration_allpass_q", 0.707) or 0.707)
                except Exception:
                    sub_allpass_q = 0.707
            return dict(
                compute_direct_dac_bass_integration_diagnostics(
                    bundle,
                    float(fc_hz),
                    profile,
                    main_hpf_order=int(xo_order),
                    sub_lpf_order=int(xo_order),
                    sub_hpf_hz=float(sub_hpf_hz),
                    sub_hpf_order=int(sub_hpf_order),
                    sub_allpass_freq_hz=sub_allpass_freq_hz,
                    sub_allpass_q=sub_allpass_q,
                )
                or {}
            )
        return dict(getattr(bundle, "diagnostics", {}) or {})
    except Exception:
        logger.debug("Bass Integration diagnostics refresh failed", exc_info=True)
        return dict(getattr(bundle, "diagnostics", {}) or {}) if bundle is not None else {}


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
    bass_integration_bundle = None
    bass_integration_enabled = bool(data.get("bass_integration_enable", False))
    if bass_integration_enabled:
        (
            bass_integration_bundle,
            f_l,
            m_l,
            p_l,
            f_r,
            m_r,
            p_r,
        ) = load_bass_integration_measurements(data, logger=logger)
        if bass_integration_bundle is not None:
            logger.info(
                "Bass Integration: decomposed bass integration measurements loaded; "
                "predicted totals built; xo/hpf phase model retained"
            )
    else:
        f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    perf_stats["read_s"] += max(0.0, float(time.perf_counter() - read_started_at))
    if f_l is None or f_r is None:
        return None

    raw_ir_slot_keys = {}
    if bass_integration_enabled:
        raw_ir_slot_keys = {
            "file_key_l": "file_l_main",
            "path_key_l": "local_path_l_main",
            "file_key_r": "file_r_main",
            "path_key_r": "local_path_r_main",
        }

    raw_ir_l, raw_ir_fs_l, raw_ir_r, raw_ir_fs_r = load_raw_irs_lr(
        data,
        logger=logger,
        **raw_ir_slot_keys,
    )
    if bass_integration_enabled:
        raw_ir_sub, raw_ir_fs_sub = load_raw_ir_sub(data, logger=logger)
    else:
        raw_ir_sub, raw_ir_fs_sub = None, 0

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
        "bass_integration_bundle": bass_integration_bundle,
        "raw_ir_l": raw_ir_l,
        "raw_ir_fs_l": raw_ir_fs_l,
        "raw_ir_r": raw_ir_r,
        "raw_ir_fs_r": raw_ir_fs_r,
        "raw_ir_sub": raw_ir_sub,
        "raw_ir_fs_sub": raw_ir_fs_sub,
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

    bi_recommended_xo_hz = None
    bi_selected_diagnostics = {}
    bi_allpass_recommendation = {}
    if bool(data.get("bass_integration_enable", False)):
        bundle = ctx.get("bass_integration_bundle", None)
        bi_recommended_xo_hz = _compute_crossover_recommendation(bundle, data)
        bi_mode = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        ).strip().lower()
        data["bass_integration_allpass_auto_enable"] = bool(
            data.get("bass_integration_allpass_auto_enable", False)
        )
        data["bass_integration_allpass_auto_applied"] = False
        data["bass_integration_allpass_freq_hz"] = float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0)
        data["bass_integration_allpass_q"] = float(data.get("bass_integration_allpass_q", 0.707) or 0.707)
        data["bass_integration_allpass_reason"] = ""
        if bi_mode == "direct_dac":
            manual_xo_override = bool(data.get("sub_crossover_manual_override", False))
            if manual_xo_override:
                try:
                    manual_xo_hz = float(data.get("sub_crossover_hz", 80.0) or 80.0)
                except Exception:
                    manual_xo_hz = 80.0
                if not math.isfinite(manual_xo_hz) or manual_xo_hz <= 0.0:
                    manual_xo_hz = 80.0
                data["sub_crossover_hz"] = float(manual_xo_hz)
                data["avr_crossover_hz"] = float(manual_xo_hz)
                logger.info(
                    "Bass Integration Direct-DAC manual XO override: "
                    f"{float(manual_xo_hz):.1f} Hz"
                )
            elif bi_recommended_xo_hz is not None:
                data["sub_crossover_hz"] = float(bi_recommended_xo_hz)
                data["avr_crossover_hz"] = float(bi_recommended_xo_hz)
                logger.info(
                    "Bass Integration Direct-DAC auto XO selected: "
                    f"{float(bi_recommended_xo_hz):.1f} Hz"
                )
            if bool(data.get("bass_integration_allpass_auto_enable", False)):
                bi_allpass_recommendation = _compute_direct_dac_allpass_recommendation(bundle, data)
                data["bass_integration_allpass_auto_applied"] = bool(bi_allpass_recommendation.get("enabled", False))
                data["bass_integration_allpass_freq_hz"] = float(
                    bi_allpass_recommendation.get("freq_hz", 0.0) or 0.0
                )
                data["bass_integration_allpass_q"] = float(
                    bi_allpass_recommendation.get("q", 0.707) or 0.707
                )
                data["bass_integration_allpass_reason"] = str(
                    bi_allpass_recommendation.get("reason", "") or ""
                )
                if bool(data.get("bass_integration_allpass_auto_applied", False)):
                    logger.info(
                        "Bass Integration Direct-DAC auto allpass applied: "
                        f"{float(data.get('bass_integration_allpass_freq_hz', 0.0)):.1f} Hz, "
                        f"Q {float(data.get('bass_integration_allpass_q', 0.707)):.3f}"
                    )
                else:
                    logger.info(
                        "Bass Integration Direct-DAC auto allpass kept OFF: "
                        f"{str(data.get('bass_integration_allpass_reason', '') or 'No meaningful improvement found.')}"
                    )
            else:
                data["bass_integration_allpass_reason"] = "Auto allpass optimization is OFF."
        else:
            data["bass_integration_allpass_reason"] = "Direct DAC only."
        bi_selected_diagnostics = _compute_selected_bass_integration_diagnostics(bundle, data)

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
        "raw_ir_l": ctx.get("raw_ir_l"),
        "raw_ir_fs_l": ctx.get("raw_ir_fs_l", 0),
        "raw_ir_r": ctx.get("raw_ir_r"),
        "raw_ir_fs_r": ctx.get("raw_ir_fs_r", 0),
        "raw_ir_sub": ctx.get("raw_ir_sub"),
        "raw_ir_fs_sub": ctx.get("raw_ir_fs_sub", 0),
    }
    if bool(data.get("bass_integration_enable", False)):
        bundle = ctx.get("bass_integration_bundle", None)
        data["_bass_integration_meta"] = {
            "enabled": True,
            "mode": str(data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"),
            "profile": str(data.get("bass_integration_profile", "safe") or "safe"),
            "avr_crossover_hz": float(data.get("avr_crossover_hz", 80.0) or 80.0),
            "inputs": {
                "l_main": str(data.get("local_path_l_main", "") or "") or str(dict(data.get("file_l_main", {}) or {}).get("filename", "") or ""),
                "r_main": str(data.get("local_path_r_main", "") or "") or str(dict(data.get("file_r_main", {}) or {}).get("filename", "") or ""),
                "l_sub": str(data.get("local_path_l_sub", "") or "") or str(dict(data.get("file_l_sub", {}) or {}).get("filename", "") or ""),
                "r_sub": str(data.get("local_path_r_sub", "") or "") or str(dict(data.get("file_r_sub", {}) or {}).get("filename", "") or ""),
            },
            "diagnostics": dict(bi_selected_diagnostics or getattr(bundle, "diagnostics", {}) or {}),
            "recommended_crossover_hz": bi_recommended_xo_hz,
            "sub_crossover_manual_override": bool(data.get("sub_crossover_manual_override", False)),
            "recommended_allpass": {
                "enabled": bool(data.get("bass_integration_allpass_auto_applied", False)),
                "freq_hz": float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0),
                "q": float(data.get("bass_integration_allpass_q", 0.707) or 0.707),
                "improvement_score": float(bi_allpass_recommendation.get("improvement_score", 0.0) or 0.0),
                "reason": str(data.get("bass_integration_allpass_reason", "") or ""),
            },
            "allpass_baseline_metrics": dict(bi_allpass_recommendation.get("baseline", {}) or {}),
            "allpass_optimized_metrics": dict(bi_allpass_recommendation.get("optimized", {}) or {}),
        }
        measurements["bass_integration_enabled"] = True
        measurements["bass_integration_bundle"] = bundle
        measurements["bass_integration_mode"] = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        )
        measurements["avr_crossover_hz"] = float(data.get("avr_crossover_hz", 80.0) or 80.0)
        measurements["bass_integration_profile"] = str(
            data.get("bass_integration_profile", "safe") or "safe"
        )
        measurements["bass_integration_allpass_auto_enable"] = bool(
            data.get("bass_integration_allpass_auto_enable", False)
        )
        measurements["bass_integration_allpass_auto_applied"] = bool(
            data.get("bass_integration_allpass_auto_applied", False)
        )
        measurements["bass_integration_allpass_freq_hz"] = float(
            data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0
        )
        measurements["bass_integration_allpass_q"] = float(
            data.get("bass_integration_allpass_q", 0.707) or 0.707
        )

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
            "sub_ir_f": None,
            "sub_st_f": None,
            "sub_meas_f": {},
            "l_imp_f": None,
            "r_imp_f": None,
            "ui_dashboards": {},
        }
    )
