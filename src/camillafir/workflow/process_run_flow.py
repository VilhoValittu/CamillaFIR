from __future__ import annotations

import logging
import math
import time
import typing
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..common.result_postprocess import _irwin_tag
from ..config.camillafir_config import save_config
from ..config.camillafir_pipeline import (
    build_xos_hpf,
    choose_dash_fs,
    choose_target_rates,
    collect_ui_data,
    detect_is_wav_source,
    filter_type_short,
    log_df_smoothing_toggle,
)
from ..engine import build_config, run_pipeline, summarize_run
from ..io.auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ..io.auto_mode.shared import _auto_goal_forced_level_window
from ..io.camillafir_automatic_mode import (
    AUTO_MODE_COMPAT_VERSION,
    AUTO_MODE_EXC_FROM_F6_ADD_HZ,
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_REFINE_TRIALS,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    AUTO_MODE_TRIALS,
    _auto_goal_norm,
    _auto_optuna_storage_path,
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
    _run_auto_mode_search,
    get_auto_mode_cache_path,
)
from ..io.measurements_loader import load_measurements_lr
from ..io.measurements_txt import parse_measurements_from_path
from ..resources.i8n.camillafir_i18n import t
from ..ui.camillafir_utils import scale_taps_with_fs
from ..ui.process_run_bridge import ProcessRunCallbacks, ProcessRunUiBridge

logger = logging.getLogger("CamillaFIR")


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


def _build_auto_selected_text(run_data: dict) -> str:
    try:
        target_name = str(
            run_data.get(
                "target_curve_name",
                run_data.get("hc_mode", "n/a"),
            )
            or "n/a"
        ).strip()
    except Exception:
        target_name = "n/a"
    if not target_name:
        target_name = "n/a"

    f6_hz = _auto_safe_float(
        run_data.get("_auto_mag_c_min_hz", run_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f"{f6_hz:.1f} Hz" if np.isfinite(f6_hz) else "n/a"

    hpf_enabled = bool(run_data.get("hpf_enable", False))
    hpf_freq = _auto_safe_float(run_data.get("hpf_freq", float("nan")), float("nan"))
    hpf_slope = _auto_safe_float(run_data.get("hpf_slope", float("nan")), float("nan"))
    if hpf_enabled and np.isfinite(hpf_freq):
        if np.isfinite(hpf_slope):
            hpf_txt = f"{hpf_freq:.1f} Hz/{int(round(hpf_slope))} dB/oct"
        else:
            hpf_txt = f"{hpf_freq:.1f} Hz"
    else:
        hpf_txt = "off"

    ft_short = filter_type_short(str(run_data.get("filter_type", "") or ""))

    phase_hz = _auto_safe_float(run_data.get("phase_limit", float("nan")), float("nan"))
    phase_txt = f"{phase_hz:.1f} Hz" if np.isfinite(phase_hz) else "n/a"

    mixed_hz = _auto_safe_float(run_data.get("mixed_freq", float("nan")), float("nan"))
    mixed_txt = f"{mixed_hz:.1f} Hz" if np.isfinite(mixed_hz) else "n/a"

    best_metrics = attach_official_rank_score(run_data.get("best_metrics", {}))
    rank_score = official_rank_score(best_metrics)
    rank_txt = f", winner rank {rank_score:.3f}/100" if np.isfinite(rank_score) else ""

    detail_txt = ""
    if ft_short in ("Linear", "Asymmetric"):
        detail_txt = f", phase limit {phase_txt}"
    elif ft_short == "Mixed":
        detail_txt = f", mixed freq {mixed_txt}"

    return (
        "Chosen (Automatic mode): "
        f"target {target_name}, HPF {hpf_txt}, -6 dB {f6_txt}{detail_txt}{rank_txt}"
    )


def _prepare_ui_and_measurements(
    *,
    pin_obj,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
    run_started_at: float,
) -> dict | None:
    perf_stats = {
        "read_s": 0.0,
        "dsp_s": 0.0,
        "zip_png_s": 0.0,
    }
    per_fs_stats: dict[int, dict[str, float]] = {}

    data = collect_ui_data(pin_obj)
    callbacks.set_auto_selected_bar("")
    data["bass_adaptive_isolation_mode"] = True
    data["bass_smooth_sigma_scale"] = 1.20
    data["bass_smooth_conf_floor"] = 0.25
    data["bass_smooth_w_gamma"] = 2.40
    data["bass_smooth_w_max"] = 0.45
    data["program_version"] = support.version
    data["auto_mode_compat_version"] = AUTO_MODE_COMPAT_VERSION

    try:
        mode = str(data.get("mode") or "BASIC").strip().upper()
        hr = support.ui_bridge.compute_health(data, mode)
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


def _run_auto_mode_seed_phases(
    ctx: dict,
    *,
    pin_obj,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
):
    data = ctx["data"]
    taps_base = int(ctx["taps_base"])
    f_l = ctx["f_l"]
    m_l = ctx["m_l"]
    f_r = ctx["f_r"]
    m_r = ctx["m_r"]
    p_l = ctx["p_l"]
    p_r = ctx["p_r"]

    try:
        auto_mode_preview = bool(
            str(data.get("mode", "BASIC") or "BASIC").strip().upper() == "AUTO"
            or data.get("camillafir_automatic_mode", False)
        )
    except Exception:
        auto_mode_preview = False
    auto_goal = _auto_goal_norm(str(data.get("auto_goal", "balanced") or "balanced"))
    data["auto_goal"] = str(auto_goal)
    forced_level_window = _auto_goal_forced_level_window(auto_goal) if auto_mode_preview else None
    if forced_level_window is not None:
        data["lvl_min"] = float(forced_level_window[0])
        data["lvl_max"] = float(forced_level_window[1])
    auto_basis = "preset_objective_score"
    logger.info(f"Automatic mode goal: {auto_goal} (basis: {auto_basis})")

    if auto_mode_preview:
        try:
            ft = str(data.get("filter_type", "mixed") or "mixed")
        except Exception:
            ft = "mixed"
        callbacks.status(
            "CamillaFIR automatic mode: init "
            f"(goal {auto_goal}, basis {auto_basis}, filter {ft}, taps {int(taps_base)})"
        )
        if forced_level_window is not None:
            callbacks.status(
                "CamillaFIR automatic mode: forced Smart Scan range "
                f"{float(forced_level_window[0]):.0f}-{float(forced_level_window[1]):.0f} Hz "
                f"(goal {auto_goal})"
            )
    if auto_mode_preview:
        try:
            est_mag_c_min = _estimate_auto_mag_c_min_hz(
                f_l,
                m_l,
                f_r,
                m_r,
                default_hz=_auto_safe_float(data.get("mag_c_min", 25.0), 25.0),
            )
            data["mag_c_min"] = float(est_mag_c_min)
            data["_auto_mag_c_min_hz"] = float(est_mag_c_min)
            est_low_bass_cut = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                )
            )
            est_exc_freq = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_EXC_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_EXC_MIN_HZ),
                    float(AUTO_MODE_EXC_MAX_HZ),
                )
            )
            data["low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["exc_freq"] = float(round(est_exc_freq, 1))
            data["_auto_low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["_auto_exc_freq_hz"] = float(round(est_exc_freq, 1))
            data["_auto_exc_seed_freq_hz"] = float(round(est_exc_freq, 1))
            callbacks.status(
                "CamillaFIR automatic mode: protection seed "
                f"(smoothed -6 dB point {float(est_mag_c_min):.1f} Hz -> "
                f"mag_c_min {float(data['mag_c_min']):.1f} Hz, "
                f"low-cut {float(data['low_bass_cut_hz']):.1f} Hz, "
                f"exc seed {float(data['exc_freq']):.1f} Hz, "
                "final exc, mag_c_min and low_bass_cut values auto-tuned in preset search)"
            )
            user_hpf_enabled = bool(data.get("hpf_enable", False))
            auto_hpf = _estimate_auto_hpf_from_response(
                f_l,
                m_l,
                f_r,
                m_r,
                default_freq_hz=_auto_safe_float(data.get("hpf_freq", 20.0), 20.0),
                default_slope_db_oct=int(_auto_safe_float(data.get("hpf_slope", 24), 24.0)),
            )
            if isinstance(auto_hpf, dict):
                auto_hpf = _resolve_auto_hpf_application(
                    auto_hpf,
                    user_hpf_enabled=bool(user_hpf_enabled),
                )
                auto_hpf_conf = _auto_safe_float(auto_hpf.get("confidence", 0.0), 0.0)
                auto_hpf_method = str(auto_hpf.get("method", "") or "").strip().lower()
                auto_hpf_apply = bool(auto_hpf.get("applied", False))
                auto_hpf_decision = str(auto_hpf.get("decision", "") or "").strip().lower()
                data["_auto_hpf_meta"] = dict(auto_hpf)
                if auto_hpf_apply:
                    auto_hpf_freq = _auto_safe_float(
                        auto_hpf.get("freq", data.get("hpf_freq", 20.0)),
                        _auto_safe_float(data.get("hpf_freq", 20.0), 20.0),
                    )
                    auto_hpf_slope = int(
                        round(
                            _auto_safe_float(
                                auto_hpf.get("slope_db_oct", data.get("hpf_slope", 24)),
                                _auto_safe_float(data.get("hpf_slope", 24), 24.0),
                            )
                        )
                    )
                    data["hpf_enable"] = True
                    data["hpf_freq"] = float(round(auto_hpf_freq, 1))
                    data["hpf_slope"] = int(max(6, auto_hpf_slope))
                    callbacks.status(
                        "CamillaFIR automatic mode: HPF auto-fit applied "
                        f"{float(data['hpf_freq']):.1f} Hz/{int(data['hpf_slope'])} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}, "
                        f"user_hpf={'on' if bool(user_hpf_enabled) else 'off'})"
                    )
                elif auto_hpf_decision == "suggest_only":
                    auto_hpf_freq = _auto_safe_float(
                        auto_hpf.get("freq", data.get("hpf_freq", 20.0)),
                        _auto_safe_float(data.get("hpf_freq", 20.0), 20.0),
                    )
                    auto_hpf_slope = int(
                        round(
                            _auto_safe_float(
                                auto_hpf.get("slope_db_oct", data.get("hpf_slope", 24)),
                                _auto_safe_float(data.get("hpf_slope", 24), 24.0),
                            )
                        )
                    )
                    callbacks.status(
                        "CamillaFIR automatic mode: HPF auto-fit suggestion "
                        f"{float(auto_hpf_freq):.1f} Hz/{int(max(6, auto_hpf_slope))} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        "not auto-applied"
                    )
                elif user_hpf_enabled:
                    callbacks.status(
                        "CamillaFIR automatic mode: HPF auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        "keeping user HPF settings"
                    )
                else:
                    callbacks.status(
                        "CamillaFIR automatic mode: HPF auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f})"
                    )

            auto_target_mode = support.auto_target_mode_norm(data.get("auto_target_mode", "auto"))
            data["auto_target_mode"] = str(auto_target_mode)
            hc_mode_raw = str(data.get("hc_mode", "") or "").strip().lower()
            has_local_target = bool(str(data.get("local_path_house", "") or "").strip())
            has_upload_target = support.has_uploaded_target_file(data)
            wants_custom_target = bool(("upload" in hc_mode_raw) or ("custom" in hc_mode_raw))
            use_user_target = bool(auto_target_mode == "selected")
            if not use_user_target:
                use_user_target = bool(
                    has_local_target
                    or (wants_custom_target and has_upload_target)
                )
            if use_user_target:
                data.pop("_auto_target_seed_preset", None)
                selected_hc = str(data.get("hc_mode", "n/a") or "n/a")
                if auto_target_mode == "selected":
                    callbacks.status(
                        "CamillaFIR automatic mode: target curve mode=selected "
                        f"(using {selected_hc}, auto target comparison disabled)"
                    )
                else:
                    callbacks.status(
                        "CamillaFIR automatic mode: target curve mode=user "
                        f"(using {selected_hc}, skip built-in target comparison)"
                    )
            else:
                if wants_custom_target and not has_local_target:
                    callbacks.status(
                        "CamillaFIR automatic mode: custom target selected but no file found, "
                        "using built-in target comparison"
                    )
                pre_target_rates = choose_target_rates(data)
                pre_fs = int(pre_target_rates[0]) if pre_target_rates else int(data.get("fs", 44100) or 44100)
                if bool(data.get("multi_rate_opt", False)):
                    pre_taps = int(scale_taps_with_fs(pre_fs, base_taps=taps_base))
                else:
                    pre_taps = int(taps_base)
                pre_xos, pre_hpf = build_xos_hpf(data)
                pre_measurements = {
                    "f_l": np.asarray(f_l, dtype=float),
                    "m_l": np.asarray(m_l, dtype=float),
                    "p_l": np.asarray(p_l, dtype=float),
                    "f_r": np.asarray(f_r, dtype=float),
                    "m_r": np.asarray(m_r, dtype=float),
                    "p_r": np.asarray(p_r, dtype=float),
                    "is_wav_source": bool(detect_is_wav_source(data)),
                }
                callbacks.status(
                    "CamillaFIR automatic mode: target preselect init "
                    f"(top-{AUTO_MODE_TARGET_TOP_N}, {AUTO_MODE_TARGET_TRIALS_PER_CURVE} trials/curve, "
                    f"fs {int(pre_fs)} Hz, taps {int(pre_taps)}, "
                    f"-6 dB point {float(est_mag_c_min):.1f} Hz, goal {auto_goal})"
                )
                tc_pick = _auto_select_target_curve_with_trials(
                    base_data=data,
                    measurements=pre_measurements,
                    fs_v=int(pre_fs),
                    taps_v=int(pre_taps),
                    xos=pre_xos,
                    hpf=pre_hpf,
                    pin_obj=pin_obj,
                    status_cb=callbacks.status,
                    top_n=int(AUTO_MODE_TARGET_TOP_N),
                    trials_per_curve=int(AUTO_MODE_TARGET_TRIALS_PER_CURVE),
                )
                if not isinstance(tc_pick, dict):
                    tc_pick = _auto_select_builtin_target_curve(
                        data,
                        f_l=f_l,
                        m_l=m_l,
                        f_r=f_r,
                        m_r=m_r,
                    )
                if isinstance(tc_pick, dict):
                    chosen_hc = str(tc_pick.get("selected_hc_mode", "Harman6") or "Harman6")
                    prev_hc = str(data.get("hc_mode", "") or "")
                    method_raw = str(tc_pick.get("selection_method", "fit_rms") or "fit_rms")
                    method_txt = support.auto_target_selection_method_text(method_raw)
                    data["hc_mode"] = chosen_hc
                    target_seed_preset = dict(tc_pick.get("best_preset", {}) or {})
                    if target_seed_preset:
                        data["_auto_target_seed_preset"] = dict(target_seed_preset)
                    else:
                        data.pop("_auto_target_seed_preset", None)
                    if chosen_hc == "Adaptive" and "_synth_hc_f" in tc_pick:
                        data["_synth_hc_f"] = tc_pick["_synth_hc_f"]
                        data["_synth_hc_m"] = tc_pick["_synth_hc_m"]
                    else:
                        data.pop("_synth_hc_f", None)
                        data.pop("_synth_hc_m", None)
                    data["local_path_house"] = ""
                    data["_auto_target_curve_meta"] = dict(tc_pick)
                    if chosen_hc == "Adaptive":
                        callbacks.status(
                            "CamillaFIR automatic mode: adaptive target selected "
                            "(synthesized from room measurements — bass buildup, tilt and HF roll-off)"
                        )
                    else:
                        callbacks.status(
                            "CamillaFIR automatic mode: target preselect winner "
                            f"{chosen_hc} (method {method_txt}, "
                            f"fit_rms {float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                            f"tested {int(tc_pick.get('top_n', 0) or 0)} curves x "
                            f"{int(tc_pick.get('trials_per_curve', 0) or 0)} trials)"
                        )
                    logger.info(
                        f"Automatic mode target select: {chosen_hc} "
                        f"(fit_rms={float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                        f"method={method_txt}, "
                        f"prev={prev_hc or 'n/a'})"
                    )
                    if target_seed_preset:
                        logger.info(
                            "Automatic mode target seed preset: "
                            + ", ".join([f"{k}={v}" for k, v in target_seed_preset.items()])
                        )
        except Exception as exc:
            logger.warning(f"Automatic mode target preselect failed: {type(exc).__name__}: {exc}")

    ctx["auto_mode_preview"] = auto_mode_preview
    ctx["auto_goal"] = auto_goal
    ctx["auto_basis"] = auto_basis


def _prepare_target_curve_and_run_context(
    ctx: dict,
    *,
    pin_obj,
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

    hc_f, hc_m, hc_source = support.ui_bridge.load_house_curve(
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
    log_df_smoothing_toggle(pin_obj, logger)

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


def _run_auto_mode_search_if_needed(
    ctx: dict,
    *,
    pin_obj,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
):
    if not bool(ctx["auto_mode_enabled"]):
        return

    data = ctx["data"]
    measurements = ctx["measurements"]
    target_rates = ctx["target_rates"]
    xos = ctx["xos"]
    hpf = ctx["hpf"]
    hc_f = ctx["hc_f"]
    hc_m = ctx["hc_m"]
    taps_base = int(ctx["taps_base"])
    auto_goal = str(ctx["auto_goal"])
    auto_basis = str(ctx["auto_basis"])

    try:
        data["comparison_mode"] = True
        auto_search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
        if bool(data.get("multi_rate_opt", False)):
            auto_search_taps = int(scale_taps_with_fs(auto_search_fs, base_taps=taps_base))
        else:
            auto_search_taps = int(taps_base)
        support.ui_bridge.set_progress(0.10)
        f6_hz = _auto_safe_float(
            data.get("_auto_mag_c_min_hz", data.get("mag_c_min", float("nan"))),
            float("nan"),
        )
        f6_txt = f", -6 dB point {f6_hz:.1f} Hz" if np.isfinite(f6_hz) else ""
        phase2_hint = int(AUTO_MODE_REFINE_TRIALS)
        if bool(AUTO_MODE_LOCAL_REFINE_ENABLED) and str(auto_goal) in ("balanced", "room-safe", "low-ripple", "subwoofers"):
            phase2_hint = int(AUTO_MODE_LOCAL_REFINE_TOP_K * AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
        callbacks.status(
            f"CamillaFIR automatic mode: preset search init "
            f"(phase1 {AUTO_MODE_TRIALS} + refine {phase2_hint} trials @ {auto_search_fs} Hz{f6_txt}, "
            f"goal {auto_goal}, basis {auto_basis}, target {str(data.get('hc_mode', 'n/a') or 'n/a')})"
        )
        auto_res = _run_auto_mode_search(
            base_data=dict(data),
            measurements=measurements,
            fs_v=int(auto_search_fs),
            taps_v=int(auto_search_taps),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin_obj=pin_obj,
            status_cb=callbacks.status,
            n_trials=int(AUTO_MODE_TRIALS),
        )
        if isinstance(auto_res, dict):
            best_preset = dict(auto_res.get("best_preset", {}) or {})
            best_applied_preset = dict(auto_res.get("best_applied_preset", best_preset) or {})
            best_metrics = attach_official_rank_score(auto_res.get("best_metrics", {}))
            winner_meta = dict(auto_res.get("winner", {}) or {})
            if best_applied_preset:
                data.update(best_applied_preset)
                data["program_version"] = support.version
                data["auto_mode_compat_version"] = AUTO_MODE_COMPAT_VERSION
                measurements["ui_data"] = data
            best_auto_exc_hz = _auto_safe_float(
                auto_res.get(
                    "best_auto_exc_freq_hz",
                    best_metrics.get("auto_exc_zero_penalty_hz", float("nan")),
                ),
                float("nan"),
            )
            if np.isfinite(best_auto_exc_hz):
                best_auto_exc_hz = float(
                    np.clip(
                        float(best_auto_exc_hz),
                        float(AUTO_MODE_EXC_MIN_HZ),
                        float(AUTO_MODE_EXC_MAX_HZ),
                    )
                )
                data["exc_freq"] = float(round(best_auto_exc_hz, 1))
                data["_auto_exc_freq_hz"] = float(round(best_auto_exc_hz, 1))
                logger.info(
                    "CamillaFIR automatic mode: excursion protection auto-tuned "
                    f"to zero-penalty floor {float(data['_auto_exc_freq_hz']):.1f} Hz"
                )
            sel_goal = str(auto_res.get("auto_goal", data.get("auto_goal", "balanced")) or "balanced")
            sel_basis = str(auto_res.get("selection_basis", "rank_score") or "rank_score")
            data["_auto_mode_meta"] = {
                "enabled": True,
                "auto_goal": str(sel_goal),
                "selection_basis": str(sel_basis),
                "optimizer_backend": str(auto_res.get("optimizer_backend", "builtin") or "builtin"),
                "trials_total": int(auto_res.get("trials_total", AUTO_MODE_TRIALS)),
                "trials_ok": int(auto_res.get("trials_ok", 0)),
                "trials_phase1_total": int(auto_res.get("trials_phase1_total", AUTO_MODE_TRIALS)),
                "trials_phase1_ok": int(auto_res.get("trials_phase1_ok", 0)),
                "phase1_plateau_hit": bool(auto_res.get("phase1_plateau_hit", False)),
                "trials_phase2_total": int(auto_res.get("trials_phase2_total", AUTO_MODE_REFINE_TRIALS)),
                "trials_phase2_ok": int(auto_res.get("trials_phase2_ok", 0)),
                "phase2_plateau_hit": bool(auto_res.get("phase2_plateau_hit", False)),
                "search_fs": int(auto_res.get("search_fs", auto_search_fs)),
                "search_taps": int(auto_res.get("search_taps", auto_search_taps)),
                "auto_exc_seed_freq_hz": (
                    float(data.get("_auto_exc_seed_freq_hz"))
                    if np.isfinite(_auto_safe_float(data.get("_auto_exc_seed_freq_hz", float("nan")), float("nan")))
                    else float("nan")
                ),
                "best_auto_exc_freq_hz": (
                    float(data.get("_auto_exc_freq_hz"))
                    if np.isfinite(_auto_safe_float(data.get("_auto_exc_freq_hz", float("nan")), float("nan")))
                    else float("nan")
                ),
                "best_metrics": best_metrics,
                "best_preset": best_preset,
                "phase_limit_winner_polish": dict(auto_res.get("phase_limit_winner_polish", {}) or {}),
                "mag_c_min_winner_polish": dict(auto_res.get("mag_c_min_winner_polish", {}) or {}),
                "winner": {
                    "rank_score_official": float(
                        _auto_safe_float(
                            winner_meta.get("rank_score_official", best_metrics.get("rank_score_official", float("nan"))),
                            float("nan"),
                        )
                    ),
                    "rank_score_components": dict(
                        winner_meta.get("rank_score_components", best_metrics.get("rank_score_components", {}) or {})
                    ),
                },
                "top": list(auto_res.get("top", []) or []),
            }
            best_rank_official = official_rank_score(best_metrics)
            callbacks.status(
                "CamillaFIR automatic mode: finalize "
                f"(winner rank {best_rank_official:.3f}/100, "
                f"avg {_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
                f"boost {_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
                f"events {int(best_metrics.get('events_total', 0) or 0)})"
            )
            callbacks.set_auto_selected_bar(_build_auto_selected_text(data))
            logger.info(
                "Automatic mode best: "
                f"goal={sel_goal}, basis={sel_basis}, "
                f"rank={best_rank_official:.3f}/100, "
                f"avg={_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
                f"dsp_pen={_auto_safe_float(best_metrics.get('dsp_penalty'), 0.0):.3f}, "
                f"exc_pen={_auto_safe_float(best_metrics.get('exc_penalty'), 0.0):.3f}, "
                f"boost={_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
                f"events={int(best_metrics.get('events_total', 0) or 0)}, "
                f"event_sev={_auto_safe_float(best_metrics.get('events_severity'), 0.0):.2f}"
            )
        else:
            logger.warning("Automatic mode could not produce a valid best preset; using current settings.")
        support.ui_bridge.set_progress(0.18)
    except Exception as exc:
        logger.warning(f"Automatic mode failed: {type(exc).__name__}: {exc}")


def _run_pipeline(ctx: dict, *, pin_obj, callbacks: ProcessRunCallbacks, support: ProcessRunSupport) -> bool:
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

    for index, fs_v in enumerate(target_rates):
        if bool(data.get("multi_rate_opt", False)):
            taps_v = scale_taps_with_fs(fs_v, base_taps=taps_base)
            logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
        else:
            taps_v = taps_base

        callbacks.status(f"{t('stat_calc')} {fs_v}Hz...")
        support.ui_bridge.set_progress(0.2 + 0.6 * (index / len(target_rates)))
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


def _finalize_run_outputs(ctx: dict, *, callbacks: ProcessRunCallbacks, support: ProcessRunSupport):
    data = ctx["data"]
    results_by_fs = ctx["results_by_fs"]
    perf_stats = ctx["perf_stats"]
    per_fs_stats = ctx["per_fs_stats"]
    ft_short = ctx["ft_short"]
    file_ts = ctx["file_ts"]
    irw_tag = ctx["irw_tag"]
    dash_fs = int(ctx["dash_fs"])
    ts = ctx["ts"]
    target_curve_tag = ctx["target_curve_tag"]

    zip_started_at = time.perf_counter()
    zip_buffer, ui_dashboards, zip_perf = support.ui_bridge.build_export_zip(
        data=data,
        results=results_by_fs,
        ft_short=ft_short,
        file_ts=file_ts,
        irw_tag=irw_tag,
        write_dashboards=bool(ctx["zip_dashboards_on"]),
        dash_fs=dash_fs,
    )
    ctx["ui_dashboards"] = ui_dashboards
    zip_elapsed = max(0.0, float(time.perf_counter() - zip_started_at))
    perf_stats["zip_png_s"] += max(float(zip_perf.get("zip_png_s", 0.0) or 0.0), zip_elapsed)
    for fs_key, st in (zip_perf.get("per_fs_stats", {}) or {}).items():
        slot = per_fs_stats.setdefault(int(fs_key), {})
        slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + float(st.get("zip_png_s", 0.0) or 0.0)

    fname, saved_filters_dir, _save_msg = support.ui_bridge.save_export_bundle(
        zip_buffer,
        data=data,
        ft_short=ft_short,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        ts=ts,
        program_version=str(data.get("program_version", support.version) or support.version),
    )
    auto_cache_path = None
    optuna_storage_path = None
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    if bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False)):
        try:
            auto_cache_path = str(
                get_auto_mode_cache_path(
                    compat_version=str(
                        data.get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
                        or AUTO_MODE_COMPAT_VERSION
                    ),
                )
            )
        except Exception:
            auto_cache_path = None
        try:
            if bool(data.get("auto_mode_optuna_persistent_study", True)):
                optuna_storage_path = str(
                    _auto_optuna_storage_path(
                        compat_version=str(
                            data.get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
                            or AUTO_MODE_COMPAT_VERSION
                        ),
                    )
                )
            else:
                optuna_storage_path = None
        except Exception:
            optuna_storage_path = None

    l_st_f = ctx["l_st_f"]
    r_st_f = ctx["r_st_f"]
    l_imp_f = ctx["l_imp_f"]
    r_imp_f = ctx["r_imp_f"]
    if l_st_f is None or r_st_f is None or l_imp_f is None or r_imp_f is None:
        fallback = results_by_fs[-1]
        l_st_f, r_st_f = fallback.l_st, fallback.r_st
        l_imp_f, r_imp_f = fallback.l_ir, fallback.r_ir

    logger.info(
        f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
        f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis', []))}/{len(l_st_f.get('cmp_measured_mags', []))}/{len(l_st_f.get('cmp_target_mags', []))}"
    )

    support.ui_bridge.render_results(
        data,
        ctx["f_l"],
        ctx["m_l"],
        ctx["p_l"],
        ctx["f_r"],
        ctx["m_r"],
        ctx["p_r"],
        l_imp_f,
        r_imp_f,
        l_st_f,
        r_st_f,
        fname,
        zip_buffer,
        dash_html_l=ui_dashboards.get("left_html"),
        dash_html_r=ui_dashboards.get("right_html"),
        run_started_at=ctx["run_started_at"],
        perf_stats=perf_stats,
        per_fs_stats=per_fs_stats,
        saved_filters_dir=saved_filters_dir,
        auto_cache_path=auto_cache_path,
        optuna_storage_path=optuna_storage_path,
    )


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
