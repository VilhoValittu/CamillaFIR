from __future__ import annotations

import logging
import re
import typing

import numpy as np

from ..auto_mode.api import (
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
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
    _run_auto_mode_search,
)
from ..auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ..auto_mode.shared import _auto_goal_forced_level_window
from ..config.camillafir_pipeline import (
    build_xos_hpf,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
)
from ..ui.camillafir_utils import scale_taps_with_fs
from ..ui.ng_bridge import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from .process_run_flow import ProcessRunSupport

logger = logging.getLogger("CamillaFIR")

_AUTO_PROGRESS_INIT = 0.06
_AUTO_PROGRESS_TARGET_MODE = 0.10
_AUTO_PROGRESS_TARGET_PRESELECT = 0.12
_AUTO_PROGRESS_TARGET_TRIALS_START = 0.16
_AUTO_PROGRESS_TARGET_TRIALS_END = 0.24
_AUTO_PROGRESS_PRESET_SEARCH_START = 0.26
_AUTO_PROGRESS_PHASE1_START = 0.28
_AUTO_PROGRESS_PHASE1_END = 0.42
_AUTO_PROGRESS_PHASE2_START = 0.42
_AUTO_PROGRESS_PHASE2_END = 0.56
_AUTO_PROGRESS_PHASE3_START = 0.56
_AUTO_PROGRESS_PHASE3_END = 0.58
_AUTO_PROGRESS_FINALIZE = 0.60


def _progress_lerp(start: float, end: float, fraction: float) -> float:
    frac = float(np.clip(_auto_safe_float(fraction, 0.0), 0.0, 1.0))
    return float(start + (end - start) * frac)


def _auto_progress_fraction(
    done: int,
    total: int,
    *,
    base_done: int = 0,
    base_total: int = 1,
) -> float:
    total_i = max(1, int(total))
    base_total_i = max(1, int(base_total))
    done_i = int(np.clip(int(done), 0, total_i))
    base_done_i = int(np.clip(int(base_done), 0, base_total_i))
    overall = float(base_done_i) + (float(done_i) / float(total_i))
    return float(np.clip(overall / float(base_total_i), 0.0, 1.0))


def _estimate_auto_progress_from_status(msg: str) -> float | None:
    text = str(msg or "").strip()
    if not text:
        return None
    lower = text.lower()
    if "camillafir automatic mode" not in lower:
        return None

    if "automatic mode: finalize " in lower:
        return float(_AUTO_PROGRESS_FINALIZE)
    if (
        "phase2 summary" in lower
        or "phase 2 pareto selected winner" in lower
        or "phase_limit winner polish" in lower
        or "mag_c_min winner polish" in lower
    ):
        return float(_AUTO_PROGRESS_FINALIZE - 0.01)

    match = re.search(r"phase 3/3 micro\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE3_START,
            _AUTO_PROGRESS_PHASE3_END,
            _auto_progress_fraction(done, total),
        )
    if "phase3 micro summary" in lower:
        return float(_AUTO_PROGRESS_PHASE3_END)
    if "phase3 micro " in lower:
        return float(_AUTO_PROGRESS_PHASE3_START)

    match = re.search(
        r"phase 2/2 local center#(\d+)\s+(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        center_idx, done, total = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE2_START,
            _AUTO_PROGRESS_PHASE2_END,
            _auto_progress_fraction(
                done,
                total,
                base_done=max(0, center_idx - 1),
                base_total=max(1, int(AUTO_MODE_LOCAL_REFINE_TOP_K)),
            ),
        )
    match = re.search(r"local refine summary .*center #(\d+)", text, flags=re.IGNORECASE)
    if match:
        center_idx = int(match.group(1))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE2_START,
            _AUTO_PROGRESS_PHASE2_END,
            float(center_idx) / float(max(1, int(AUTO_MODE_LOCAL_REFINE_TOP_K))),
        )
    if "automatic mode: local refine " in lower:
        return float(_AUTO_PROGRESS_PHASE2_START)

    match = re.search(r"phase 1/2\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE1_START,
            _AUTO_PROGRESS_PHASE1_END,
            _auto_progress_fraction(done, total),
        )
    if "phase1 done" in lower:
        return float(_AUTO_PROGRESS_PHASE1_END)

    match = re.search(
        r"target\s+(\d+)/(\d+).*?trial\s+(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        target_done, target_total, trial_done, trial_total = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(
                trial_done,
                trial_total,
                base_done=max(0, target_done - 1),
                base_total=target_total,
            ),
        )
    match = re.search(r"best improved\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match and "selecting target curve" in lower:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(done, total),
        )
    match = re.search(
        r"selecting target curve\s+\(testing .*?(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(max(0, done - 1), total),
        )

    if (
        "target finalize" in lower
        or "target preselect winner" in lower
        or "adaptive target selected" in lower
        or "target loaded directly from cache" in lower
        or "adaptive target synthesized from measurements" in lower
    ):
        return float(_AUTO_PROGRESS_TARGET_TRIALS_END)
    if "target shortlist" in lower:
        return float(_AUTO_PROGRESS_TARGET_TRIALS_START - 0.01)
    if "target preselect top-3" in lower or "target preselect cache seed" in lower:
        return float(_AUTO_PROGRESS_TARGET_PRESELECT)
    if "target preselect init" in lower:
        return float(_AUTO_PROGRESS_TARGET_PRESELECT)
    if (
        "target curve mode=" in lower
        or "custom target selected but no file found" in lower
    ):
        return float(_AUTO_PROGRESS_TARGET_MODE)

    match = re.search(r"cache refine round\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        round_idx, round_total = (int(match.group(1)), int(match.group(2)))
        inner_match = re.search(r"(\d+)/(\d+)", text[match.end() :], flags=re.IGNORECASE)
        if inner_match:
            inner_done, inner_total = (int(inner_match.group(1)), int(inner_match.group(2)))
        else:
            inner_done, inner_total = (0, 1)
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE1_START,
            _AUTO_PROGRESS_PHASE2_END,
            _auto_progress_fraction(
                inner_done,
                inner_total,
                base_done=max(0, round_idx - 1),
                base_total=round_total,
            ),
        )
    if "cache refine init" in lower or "preset loaded from cache" in lower:
        return float(_AUTO_PROGRESS_PRESET_SEARCH_START)
    if "cache refine summary" in lower:
        return float(_AUTO_PROGRESS_PHASE2_END)

    if "preset search init" in lower:
        return float(_AUTO_PROGRESS_PRESET_SEARCH_START)
    if "hpf auto-fit" in lower or "protection seed" in lower:
        return float(_AUTO_PROGRESS_TARGET_MODE - 0.01)
    if "automatic mode: init" in lower:
        return float(_AUTO_PROGRESS_INIT)
    return None


def _set_auto_progress(ctx: dict, *, support: ProcessRunSupport, value: float) -> None:
    state = ctx.setdefault("_auto_progress_state", {})
    current = _auto_safe_float(state.get("value", 0.0), 0.0)
    next_value = float(np.clip(_auto_safe_float(value, current), current, _AUTO_PROGRESS_FINALIZE))
    if next_value <= float(current) + 1e-9:
        return
    state["value"] = float(next_value)
    support.ui_bridge.set_progress(float(next_value))


def _get_auto_status_callback(
    ctx: dict,
    *,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
) -> typing.Callable[[str], None]:
    cb = ctx.get("_auto_status_callback")
    if callable(cb):
        return cb

    def _status(msg: str) -> None:
        progress = _estimate_auto_progress_from_status(msg)
        if progress is not None:
            _set_auto_progress(ctx, support=support, value=float(progress))
        callbacks.status(msg)

    ctx["_auto_status_callback"] = _status
    return _status


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


def _auto_finalize_status_suffix(winner_explanation: dict | None) -> str:
    phase_label = str(dict(winner_explanation or {}).get("phase_label", "") or "").strip()
    if not phase_label:
        return ""
    lower = phase_label.lower()
    if not any(token in lower for token in ("pareto", "tie-break", "winner polish", "cache")):
        return ""
    return f", via {phase_label}"


def _build_auto_finalize_status(
    best_metrics: dict | None,
    *,
    winner_explanation: dict | None = None,
) -> str:
    metrics = attach_official_rank_score(best_metrics or {})
    best_rank_official = official_rank_score(metrics)
    return (
        "CamillaFIR automatic mode: finalize "
        f"(winner rank {best_rank_official:.3f}/100, "
        f"avg {_auto_safe_float(metrics.get('avg_score'), 0.0):.3f}, "
        f"boost {_auto_safe_float(metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
        f"events {int(metrics.get('events_total', 0) or 0)}"
        f"{_auto_finalize_status_suffix(winner_explanation)})"
    )


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
    auto_status = (
        _get_auto_status_callback(ctx, callbacks=callbacks, support=support)
        if auto_mode_preview
        else callbacks.status
    )

    if auto_mode_preview:
        try:
            ft = str(data.get("filter_type", "mixed") or "mixed")
        except Exception:
            ft = "mixed"
        auto_status(
            "CamillaFIR automatic mode: init "
            f"(goal {auto_goal}, basis {auto_basis}, filter {ft}, taps {int(taps_base)})"
        )
        if forced_level_window is not None:
            auto_status(
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
            auto_status(
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
                    auto_status(
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
                    auto_status(
                        "CamillaFIR automatic mode: HPF auto-fit suggestion "
                        f"{float(auto_hpf_freq):.1f} Hz/{int(max(6, auto_hpf_slope))} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        "not auto-applied"
                    )
                elif user_hpf_enabled:
                    auto_status(
                        "CamillaFIR automatic mode: HPF auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        "keeping user HPF settings"
                    )
                else:
                    auto_status(
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
                    auto_status(
                        "CamillaFIR automatic mode: target curve mode=selected "
                        f"(using {selected_hc}, auto target comparison disabled)"
                    )
                else:
                    auto_status(
                        "CamillaFIR automatic mode: target curve mode=user "
                        f"(using {selected_hc}, skip built-in target comparison)"
                    )
            else:
                if wants_custom_target and not has_local_target:
                    auto_status(
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
                auto_status(
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
                    status_cb=auto_status,
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
                        auto_status(
                            "CamillaFIR automatic mode: adaptive target selected "
                            "(synthesized from room measurements, bass buildup, tilt and HF roll-off)"
                        )
                    else:
                        auto_status(
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
    auto_status = _get_auto_status_callback(ctx, callbacks=callbacks, support=support)

    try:
        data["comparison_mode"] = True
        auto_search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
        if bool(data.get("multi_rate_opt", False)):
            auto_search_taps = int(scale_taps_with_fs(auto_search_fs, base_taps=taps_base))
        else:
            auto_search_taps = int(taps_base)
        _set_auto_progress(ctx, support=support, value=_AUTO_PROGRESS_PRESET_SEARCH_START)
        f6_hz = _auto_safe_float(
            data.get("_auto_mag_c_min_hz", data.get("mag_c_min", float("nan"))),
            float("nan"),
        )
        f6_txt = f", -6 dB point {f6_hz:.1f} Hz" if np.isfinite(f6_hz) else ""
        phase2_hint = int(AUTO_MODE_REFINE_TRIALS)
        if bool(AUTO_MODE_LOCAL_REFINE_ENABLED) and str(auto_goal) in ("balanced", "room-safe", "low-ripple", "subwoofers"):
            phase2_hint = int(AUTO_MODE_LOCAL_REFINE_TOP_K * AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
        auto_status(
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
            status_cb=auto_status,
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
            auto_status(
                _build_auto_finalize_status(
                    best_metrics,
                    winner_explanation=dict(auto_res.get("winner_explanation", {}) or {}),
                )
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
        _set_auto_progress(ctx, support=support, value=_AUTO_PROGRESS_FINALIZE)
    except Exception as exc:
        logger.warning(f"Automatic mode failed: {type(exc).__name__}: {exc}")
