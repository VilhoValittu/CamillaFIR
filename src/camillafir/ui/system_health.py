from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Literal, Optional

from ..resources.i8n.camillafir_i18n import t

Level = Literal["ok", "warn", "crit"]
_TOAST_LAST_SHOWN: Dict[str, float] = {}
_TOAST_EDGE_STATE: Dict[str, bool] = {}


def _prune_toast_cache(
    *,
    now: float,
    max_items: int = 256,
    max_age_s: float = 300.0,
) -> None:
    """Sisainen apufunktio: prune toast cache."""
    try:
        if not _TOAST_LAST_SHOWN:
            return

        if max_age_s > 0:
            old_keys = [k for k, ts in _TOAST_LAST_SHOWN.items() if (now - float(ts)) > float(max_age_s)]
            for k in old_keys:
                _TOAST_LAST_SHOWN.pop(k, None)

        if max_items > 0 and len(_TOAST_LAST_SHOWN) > int(max_items):
            items: List[Tuple[str, float]] = [(k, float(v)) for k, v in _TOAST_LAST_SHOWN.items()]
            items.sort(key=lambda kv: kv[1], reverse=True)
            keep = set(k for k, _ in items[: int(max_items)])
            for k in list(_TOAST_LAST_SHOWN.keys()):
                if k not in keep:
                    _TOAST_LAST_SHOWN.pop(k, None)
    except Exception:
        return


@dataclass(frozen=True)
class Issue:
    level: Level
    title: str
    detail: str = ""


@dataclass(frozen=True)
class HealthResult:
    overall: Level
    blocked: bool
    issues: List[Issue]


def _tr(key: str, **kwargs: Any) -> str:
    text = t(key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _clean_path_text(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    return p.strip().strip('"').strip("'")


def _has_path_text(p: Any) -> bool:
    return bool(_clean_path_text(p))


def _valid_path(p: Any) -> bool:
    path = _clean_path_text(p)
    if not path:
        return False
    return os.path.isfile(path)


def _has_uploaded_file_name(v: Any) -> bool:
    try:
        if isinstance(v, dict):
            for key in ("filename", "name", "file_name"):
                name = str(v.get(key, "") or "").strip()
                if name:
                    return True
        if isinstance(v, list):
            return any(_has_uploaded_file_name(x) for x in v)
    except Exception:
        return False
    return False


def _has_uploaded_file(v: Any) -> bool:
    """Sisainen apufunktio: has uploaded file."""
    try:
        if isinstance(v, dict):
            if not _has_uploaded_file_name(v):
                return False
            content = v.get("content", None)
            if content is None:
                return False
            if isinstance(content, (bytes, bytearray, str)):
                return len(content) > 0
            return True
        if isinstance(v, list):
            return any(_has_uploaded_file(x) for x in v)
    except Exception:
        return False
    return False


def _get_toast_callable():
    return None


def show_toast(
    msg: str,
    *,
    duration: float = 5.0,
    color: Optional[str] = None,
    dedupe_key: Optional[str] = None,
    dedupe_window_s: float = 0.75,
) -> bool:
    key = str(dedupe_key or msg or "").strip()
    if key:
        now = time.monotonic()
        _prune_toast_cache(now=now)
        last = float(_TOAST_LAST_SHOWN.get(key, 0.0) or 0.0)
        if (now - last) < float(dedupe_window_s):
            return False
        _TOAST_LAST_SHOWN[key] = now

    fn = _get_toast_callable()
    if not callable(fn):
        return False
    try:
        if color is None:
            fn(msg, duration=duration)
        else:
            fn(msg, duration=duration, color=color)
        return True
    except Exception:
        return False


def _toast_on_edge(
    *,
    edge_key: str,
    active: bool,
    msg: str,
    duration: float,
    color: Optional[str] = None,
    dedupe_key: Optional[str] = None,
) -> bool:
    prev = bool(_TOAST_EDGE_STATE.get(edge_key, False))
    _TOAST_EDGE_STATE[edge_key] = bool(active)
    if active and not prev:
        return show_toast(
            msg,
            duration=duration,
            color=color,
            dedupe_key=dedupe_key or edge_key,
            dedupe_window_s=0.25,
        )
    return False


def _health_toast_color(level: str, mode_u: str) -> Optional[str]:
    if level == "warn":
        mode_s = str(mode_u or "").strip().upper()
        return "warn" if mode_s in ("BASIC", "AUTO") else "info"
    if level == "crit":
        return "error"
    return None


def compute_health(data: Dict[str, Any], mode: str) -> HealthResult:
    issues: List[Issue] = []
    mode_u = str(mode or "BASIC").strip().upper()

    raw_up_l = _has_uploaded_file_name(data.get("file_l", None))
    raw_up_r = _has_uploaded_file_name(data.get("file_r", None))
    up_l = _has_uploaded_file(data.get("file_l", None))
    up_r = _has_uploaded_file(data.get("file_r", None))
    raw_lp_l = _has_path_text(data.get("local_path_l", None))
    raw_lp_r = _has_path_text(data.get("local_path_r", None))
    lp_l = _valid_path(data.get("local_path_l", None))
    lp_r = _valid_path(data.get("local_path_r", None))

    has_upload_pair = bool(up_l and up_r)
    has_local_pair = bool(lp_l and lp_r)

    if has_upload_pair or has_local_pair:
        issues.append(Issue("ok", _tr("health_measurements"), _tr("health_upload_source_provided")))
    elif raw_up_l or raw_up_r or raw_lp_l or raw_lp_r:
        issues.append(Issue("warn", _tr("health_measurements"), _tr("health_measurements_missing_pair")))
    else:
        issues.append(Issue("warn", _tr("health_measurements"), _tr("health_measurements_missing")))

    hc_mode = str(data.get("hc_mode") or "").strip()
    if hc_mode.lower() == "upload":
        upload_ok = _has_uploaded_file(data.get("hc_custom_file", None))
        local_ok = _valid_path(data.get("local_path_house", None))
        if upload_ok or local_ok:
            issues.append(Issue("ok", _tr("health_target_curve"), _tr("health_upload_source_provided")))
        else:
            issues.append(
                Issue(
                    "warn",
                    _tr("health_target_curve_upload_selected"),
                    _tr("health_target_curve_upload_missing"),
                )
            )
    elif hc_mode:
        issues.append(Issue("ok", _tr("health_target_curve"), hc_mode))
    else:
        issues.append(Issue("warn", _tr("health_target_curve"), _tr("health_not_set")))

    mag_on = bool(data.get("mag_correct", True))
    fmin = _as_float(data.get("mag_c_min", None))
    fmax = _as_float(data.get("mag_c_max", None))
    if not mag_on:
        issues.append(Issue("ok", _tr("health_magnitude_correction"), _tr("health_disabled")))
    elif (fmin is not None) and (fmax is not None):
        if fmin >= fmax:
            issues.append(Issue("crit", _tr("health_correction_range_invalid")))
        else:
            issues.append(Issue("ok", _tr("health_correction_range"), f"{fmin:.0f}-{fmax:.0f} Hz"))
            if fmax > 350.0:
                issues.append(Issue("warn", _tr("health_correction_max_very_high")))
    else:
        issues.append(Issue("warn", _tr("health_correction_range_set_bounds")))

    fs = _as_int(data.get("fs", None))
    taps = _as_int(data.get("taps", None))
    if fs and taps and fs > 0 and taps > 0:
        latency_ms = (taps / 2.0) / float(fs) * 1000.0
        bin_hz = float(fs) / float(taps)
        ir_window_left_ms = _as_float(data.get("ir_window_left", data.get("ir_window", None)))
        short_left_window = (ir_window_left_ms is not None) and (ir_window_left_ms < 120.0)
        ftype = str(data.get("filter_type") or "").lower()
        is_min = "min" in ftype
        is_asym = "asym" in ftype

        if is_min or is_asym:
            issues.append(Issue("ok", _tr("health_latency"), _tr("health_low_latency_mode")))
        else:
            issues.append(Issue("ok", _tr("health_latency"), f"{latency_ms:.0f} ms"))
        issues.append(Issue("ok", _tr("health_resolution"), f"{bin_hz:.2f} Hz/bin"))

        if (not is_min and not is_asym) and latency_ms > 150 and not short_left_window:
            issues.append(Issue("warn", _tr("health_taps_count_high")))
    else:
        issues.append(Issue("warn", _tr("health_engine_metrics_set")))

    lvl_min = _as_float(data.get("lvl_min", None))
    lvl_max = _as_float(data.get("lvl_max", None))
    if (lvl_min is not None) and (lvl_max is not None):
        if lvl_min >= lvl_max:
            issues.append(Issue("crit", _tr("health_leveling_range_invalid")))
        elif (lvl_max - lvl_min) < 200.0:
            issues.append(Issue("warn", _tr("health_leveling_range_narrow")))

    max_boost = _as_float(data.get("max_boost", None))
    if max_boost is None:
        issues.append(Issue("ok", _tr("health_max_boost"), _tr("health_default")))
    else:
        issues.append(Issue("ok", _tr("health_max_boost"), f"{max_boost:.1f} dB"))
        if max_boost > 8.0:
            issues.append(Issue("warn", _tr("health_max_boost_high")))

    phase_limit = _as_float(data.get("phase_limit", None))
    if (phase_limit is not None) and (phase_limit > 800.0):
        issues.append(Issue("warn", _tr("health_phase_limit_high"), f"{phase_limit:.0f} Hz"))

    exc_on = bool(data.get("exc_prot", False))
    if not exc_on:
        issues.append(Issue("warn", _tr("health_excursion_protection_off")))
    else:
        issues.append(Issue("ok", _tr("health_excursion_protection"), _tr("health_enabled")))

    hpf_on = bool(data.get("hpf_enable", False))
    if hpf_on:
        hpf_f = _as_float(data.get("hpf_freq", None))
        hpf_s = _as_float(data.get("hpf_slope", None))
        if (hpf_f is None) or (hpf_f <= 0.0):
            issues.append(Issue("crit", _tr("health_hpf_freq_invalid")))
        elif (hpf_f is not None) and (fs is not None) and (fs > 0) and (hpf_f >= 0.45 * fs):
            issues.append(Issue("warn", _tr("health_hpf_freq_high")))

        if (hpf_s is None) or (hpf_s <= 0.0):
            issues.append(Issue("crit", _tr("health_hpf_slope_invalid")))

    ftype = str(data.get("filter_type") or "").lower()
    if "mixed" in ftype:
        mixed_f = _as_float(data.get("mixed_freq", None))
        trans_w = _as_float(data.get("trans_width", None))

        if (mixed_f is None) or (mixed_f <= 0.0):
            issues.append(Issue("crit", _tr("health_mixed_split_invalid")))
        else:
            if mixed_f < 40.0:
                issues.append(Issue("warn", _tr("health_mixed_split_low")))
            if (fs is not None) and (fs > 0) and (mixed_f > 0.45 * fs):
                issues.append(Issue("warn", _tr("health_mixed_split_high")))

        if (trans_w is not None) and (trans_w < 0.0):
            issues.append(Issue("crit", _tr("health_transition_width_nonnegative")))
        if (mixed_f is not None) and (mixed_f > 0.0) and (trans_w is not None) and (trans_w > mixed_f):
            issues.append(Issue("warn", _tr("health_transition_width_wider")))

    has_crit = any(i.level == "crit" for i in issues)
    has_warn = any(i.level == "warn" for i in issues)
    overall: Level = "crit" if has_crit else ("warn" if has_warn else "ok")

    blocked = (mode_u in ("BASIC", "AUTO")) and has_crit
    return HealthResult(overall=overall, blocked=blocked, issues=issues)


def toast_health_gate_result(hr: HealthResult, mode: str) -> bool:
    mode_u = str(mode or "BASIC").strip().upper()
    if hr.blocked:
        msg = format_health_summary(hr) or "Fix errors before running (blocked in BASIC/AUTO)."
        show_toast(
            msg,
            duration=15.0,
            color=_health_toast_color("crit", mode_u),
            dedupe_key=f"health_blocked:{msg}",
            dedupe_window_s=1.0,
        )
        return True

    if hr.overall in ("warn", "crit"):
        msg = format_health_summary(hr)
        if msg:
            show_toast(
                msg,
                duration=10.0,
                color=_health_toast_color(hr.overall, mode_u),
                dedupe_key=f"health_summary:{mode_u}:{hr.overall}:{msg}",
                dedupe_window_s=1.0,
            )
    return False


def toast_measurement_files_missing() -> None:
    show_toast(
        "Measurement files missing! Load Left/Right or give local.",
        duration=6.0,
        color="error",
        dedupe_key="measurement_files_missing",
    )


def toast_mode_defaults_applied(mode: str) -> None:
    try:
        msg = _tr("mode_defaults_applied_toast", mode=mode)
    except Exception:
        msg = f"Mode defaults applied: {mode}"
    show_toast(
        msg,
        color="success",
        duration=2.0,
        dedupe_key=f"mode_defaults_applied:{mode}",
    )


def toast_tdc_preset_applied(name: str) -> None:
    show_toast(
        f"TDC preset applied: {name}",
        color="success",
        duration=1.5,
        dedupe_key=f"tdc_preset:{name}",
    )


def toast_afdw_preset_applied(name: str) -> None:
    show_toast(
        f"A-FDW preset applied: {name}",
        color="success",
        duration=1.5,
        dedupe_key=f"afdw_preset:{name}",
    )


def toast_max_boost_over_cap(value: Any, max_safe_boost: float) -> None:
    try:
        v = float(value)
    except Exception:
        _TOAST_EDGE_STATE["max_boost_over_cap"] = False
        return

    cap = float(max_safe_boost if max_safe_boost is not None else 0.0)
    over = (cap > 0.0) and (v > cap + 1e-9)
    if not over:
        _TOAST_EDGE_STATE["max_boost_over_cap"] = False
        return

    try:
        cap_suffix = _tr("max_boost_help_cap", value=f"{cap:.1f}")
    except Exception:
        cap_suffix = f" (capped to {cap:.1f} dB)"
    msg = f"{_tr('max_boost')}: {v:.1f} dB > {cap:.1f} dB{cap_suffix}"
    _toast_on_edge(
        edge_key="max_boost_over_cap",
        active=True,
        msg=msg,
        duration=5.0,
        dedupe_key="max_boost_over_cap",
    )


def toast_taps_over_cap(value: Any, max_safe_taps: int) -> None:
    try:
        v = int(value)
    except Exception:
        _TOAST_EDGE_STATE["taps_over_cap"] = False
        return

    cap = int(max_safe_taps if max_safe_taps is not None else 0)
    over = v > cap
    if not over:
        _TOAST_EDGE_STATE["taps_over_cap"] = False
        return

    try:
        msg = _tr("taps_warn_over", value=cap)
    except Exception:
        msg = f"Taps > {cap}: very high latency and diminishing returns."
    _toast_on_edge(
        edge_key="taps_over_cap",
        active=True,
        msg=msg,
        duration=6.0,
        dedupe_key="taps_over_cap",
    )


def format_health_summary(hr: HealthResult, max_items: int = 3) -> str:
    """Jasentaa tai muotoilee: format health summary."""
    crits = [i for i in hr.issues if i.level == "crit"]
    warns = [i for i in hr.issues if i.level == "warn"]

    if crits:
        head = _tr("health_summary_errors", count=len(crits))
        items = crits[:max_items]
        lines = [head] + [f"- {i.title}" for i in items]
        if len(crits) > max_items:
            lines.append(f"- {_tr('health_summary_more', count=len(crits) - max_items)}")
        return "\n".join(lines)

    if warns:
        head = _tr("health_summary_warnings", count=len(warns))
        items = warns[:max_items]
        lines = [head] + [f"- {i.title}" for i in items]
        if len(warns) > max_items:
            lines.append(f"- {_tr('health_summary_more', count=len(warns) - max_items)}")
        return "\n".join(lines)

    return ""
