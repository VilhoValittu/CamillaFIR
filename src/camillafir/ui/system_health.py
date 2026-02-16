from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Level = Literal["ok", "warn", "crit"]


@dataclass(frozen=True)
class Issue:
    level: Level
    title: str
    detail: str = ""


@dataclass(frozen=True)
class HealthResult:
    overall: Level
    blocked: bool  # BASIC blocks start if True
    issues: List[Issue]


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


def _valid_path(p: Any) -> bool:
    if not isinstance(p, str):
        return False
    p = p.strip()
    if not p:
        return False
    # lightweight sanity: looks like a path
    return (":" in p) or ("/" in p) or ("\\" in p)


def compute_health(data: Dict[str, Any], mode: str) -> HealthResult:
    issues: List[Issue] = []
    mode_u = str(mode or "BASIC").strip().upper()

    # --- Measurements (based on collect_ui_data keys) ---
    layout = str(data.get("ir_window_mode") or data.get("layout") or "").lower()
    # You may not have 'layout' in collect; stereo can be inferred from stereo_link only.
    stereo_link = bool(data.get("stereo_link", False))
    # If you're running stereo processing, you almost certainly expect R as well.
    # We'll treat "stereo_link" as the stereo intent signal.
    is_stereo = stereo_link

    path_l = data.get("local_path_l", None)
    path_r = data.get("local_path_r", None)
    has_l = _valid_path(path_l)
    has_r = _valid_path(path_r)

    if not has_l:
        issues.append(Issue("crit", "Left measurement missing", "Set 'local_path_l' (or load L measurement)."))
    else:
        issues.append(Issue("ok", "Left measurement", "Loaded (local path)"))

    if is_stereo:
        if not has_r:
            issues.append(Issue("crit", "Right measurement missing", "Stereo requires 'local_path_r'."))
        else:
            issues.append(Issue("ok", "Right measurement", "Loaded (local path)"))
    else:
        if has_r:
            issues.append(Issue("ok", "Right measurement", "Provided (mono mode)"))

    # --- Target selection (limited: collect_ui_data doesn't include target file) ---
    hc_mode = str(data.get("hc_mode") or "").strip()
    if hc_mode.lower() == "upload":
        # Cannot verify file existence without hc_custom_file in collect_ui_data
        issues.append(Issue("warn", "Target curve: Upload selected", "Target file presence is not tracked by collect_ui_data."))
    elif hc_mode:
        issues.append(Issue("ok", "Target curve", hc_mode))
    else:
        issues.append(Issue("warn", "Target curve", "Not set"))

    # --- Correction range sanity ---
    fmin = _as_float(data.get("mag_c_min", None))
    fmax = _as_float(data.get("mag_c_max", None))
    if (fmin is not None) and (fmax is not None):
        if fmin >= fmax:
            issues.append(Issue("crit", "Correction range invalid. mag_c_min must be < mag_c_max."))
        else:
            issues.append(Issue("ok", "Correction range", f"{fmin:.0f}–{fmax:.0f} Hz"))
            # Optional user-safety warning (does NOT block)
            if fmax > 350.0:
                issues.append(Issue(
                    "warn",
                    "Correction max is very high. Above ~300 Hz room correction becomes speaker/measurement dependent.",
                    
                ))
    else:
        issues.append(Issue("warn", "Correction range. Set mag_c_min and mag_c_max."))

    # --- Engine metrics ---
    fs = _as_int(data.get("fs", None))
    taps = _as_int(data.get("taps", None))
    if fs and taps and fs > 0 and taps > 0:
        latency_ms = (taps / 2.0) / float(fs) * 1000.0
        bin_hz = float(fs) / float(taps)
        issues.append(Issue("ok", "Latency", f"{latency_ms:.0f} ms"))
        issues.append(Issue("ok", "Resolution", f"{bin_hz:.2f} Hz/bin"))
        if latency_ms > 150:
            issues.append(Issue("warn", "Taps count is high. May affect AV sync / usability."))
    else:
        issues.append(Issue("warn", "Engine metrics. Set fs and taps."))

    # --- Boost safety (warn only; DSP may clamp) ---
    max_boost = _as_float(data.get("max_boost", None))
    if max_boost is None:
        issues.append(Issue("ok", "Max boost", "Default"))
    else:
        issues.append(Issue("ok", "Max boost", f"{max_boost:.1f} dB"))
        if max_boost > 8.0:
            issues.append(Issue("warn", "Max boost is high. Can stress drivers."))

    # --- Protection hints ---
    exc_on = bool(data.get("exc_prot", False))
    if not exc_on:
        issues.append(Issue("warn", "Excursion protection off. Recommended for bass-heavy correction."))
    else:
        issues.append(Issue("ok", "Excursion protection", "Enabled"))

    # --- Overall & BASIC gating ---
    has_crit = any(i.level == "crit" for i in issues)
    has_warn = any(i.level == "warn" for i in issues)
    overall: Level = "crit" if has_crit else ("warn" if has_warn else "ok")

    blocked = (mode_u == "BASIC") and has_crit
    return HealthResult(overall=overall, blocked=blocked, issues=issues)

def format_health_summary(hr: HealthResult, max_items: int = 3) -> str:
    """
    Production-style short summary for toast / status.
    Example:
      ⚠ System warnings (3):
      • Correction max is very high
      • Latency is high
      • Max boost is high
    """
    crits = [i for i in hr.issues if i.level == "crit"]
    warns = [i for i in hr.issues if i.level == "warn"]

    if crits:
        head = f"❌ System errors ({len(crits)}):"
        items = crits[:max_items]
        lines = [head] + [f"• {i.title}" for i in items]
        if len(crits) > max_items:
            lines.append(f"• …and {len(crits) - max_items} more")
        return "\n".join(lines)

    if warns:
        head = f"⚠ System warnings ({len(warns)}):"
        items = warns[:max_items]
        lines = [head] + [f"• {i.title}" for i in items]
        if len(warns) > max_items:
            lines.append(f"• …and {len(warns) - max_items} more")
        return "\n".join(lines)

    return ""

