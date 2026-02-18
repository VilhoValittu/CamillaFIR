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


def _has_uploaded_file(v: Any) -> bool:
    """Best-effort check for PyWebIO file upload payload."""
    try:
        if isinstance(v, dict):
            name = str(v.get("filename", "") or "").strip()
            return bool(name)
        if isinstance(v, list):
            return any(_has_uploaded_file(x) for x in v)
    except Exception:
        return False
    return False


def compute_health(data: Dict[str, Any], mode: str) -> HealthResult:
    issues: List[Issue] = []
    mode_u = str(mode or "BASIC").strip().upper()

    # --- Target selection ---
    hc_mode = str(data.get("hc_mode") or "").strip()
    if hc_mode.lower() == "upload":
        upload_ok = _has_uploaded_file(data.get("hc_custom_file", None))
        local_ok = _valid_path(data.get("local_path_house", None))
        if upload_ok or local_ok:
            issues.append(Issue("ok", "Target curve", "Upload source provided"))
        else:
            issues.append(Issue("warn", "Target curve: Upload selected", "No uploaded target file or local target path found."))
    elif hc_mode:
        issues.append(Issue("ok", "Target curve", hc_mode))
    else:
        issues.append(Issue("warn", "Target curve", "Not set"))

    # --- Correction range sanity ---
    mag_on = bool(data.get("mag_correct", True))
    fmin = _as_float(data.get("mag_c_min", None))
    fmax = _as_float(data.get("mag_c_max", None))
    if not mag_on:
        issues.append(Issue("ok", "Magnitude correction", "Disabled"))
    elif (fmin is not None) and (fmax is not None):
        if fmin >= fmax:
            issues.append(Issue("crit", "Correction range invalid. mag_c_min must be < mag_c_max."))
        else:
            issues.append(Issue("ok", "Correction range", f"{fmin:.0f}-{fmax:.0f} Hz"))
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
        ftype = str(data.get("filter_type") or "").lower()
        is_min = ("min" in ftype)
        is_asym = ("asym" in ftype)

        if is_min or is_asym:
            issues.append(Issue("ok", "Latency", "Low-latency mode (Min/Asym)"))
        else:
            issues.append(Issue("ok", "Latency", f"{latency_ms:.0f} ms"))
        issues.append(Issue("ok", "Resolution", f"{bin_hz:.2f} Hz/bin"))

        if (not is_min and not is_asym) and latency_ms > 150:
            issues.append(Issue("warn", "Taps count is high. May affect AV sync / usability."))
    else:
        issues.append(Issue("warn", "Engine metrics. Set fs and taps."))

    # --- Leveling range sanity ---
    lvl_min = _as_float(data.get("lvl_min", None))
    lvl_max = _as_float(data.get("lvl_max", None))
    if (lvl_min is not None) and (lvl_max is not None):
        if lvl_min >= lvl_max:
            issues.append(Issue("crit", "Leveling range invalid. lvl_min must be < lvl_max."))
        elif (lvl_max - lvl_min) < 200.0:
            issues.append(Issue("warn", "Leveling range is narrow. Level estimate may be unstable."))

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

    # --- HPF sanity ---
    hpf_on = bool(data.get("hpf_enable", False))
    if hpf_on:
        hpf_f = _as_float(data.get("hpf_freq", None))
        hpf_s = _as_float(data.get("hpf_slope", None))
        if (hpf_f is None) or (hpf_f <= 0.0):
            issues.append(Issue("crit", "HPF enabled but frequency is invalid."))
        elif (hpf_f is not None) and (fs is not None) and (fs > 0) and (hpf_f >= 0.45 * fs):
            issues.append(Issue("warn", "HPF frequency is very high vs sample rate."))

        if (hpf_s is None) or (hpf_s <= 0.0):
            issues.append(Issue("crit", "HPF enabled but slope/order is invalid."))

    # --- Mixed phase sanity ---
    ftype = str(data.get("filter_type") or "").lower()
    if "mixed" in ftype:
        mixed_f = _as_float(data.get("mixed_freq", None))
        trans_w = _as_float(data.get("trans_width", None))

        if (mixed_f is None) or (mixed_f <= 0.0):
            issues.append(Issue("crit", "Mixed filter split frequency is invalid."))
        else:
            if mixed_f < 40.0:
                issues.append(Issue("warn", "Mixed split is very low. Effect may be minimal."))
            if (fs is not None) and (fs > 0) and (mixed_f > 0.45 * fs):
                issues.append(Issue("warn", "Mixed split is very high vs sample rate."))

        if (trans_w is not None) and (trans_w < 0.0):
            issues.append(Issue("crit", "Transition width must be >= 0."))
        if (mixed_f is not None) and (mixed_f > 0.0) and (trans_w is not None) and (trans_w > mixed_f):
            issues.append(Issue("warn", "Transition width is wider than split frequency."))

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
