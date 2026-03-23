from __future__ import annotations

from typing import Any

import numpy as np

from .phase_ir_utils import _resolve_ir_energy_split

_PRE_ENERGY_EPS = 1e-20
_PRE_ENERGY_WINDOW_MS = 10.0
_PRE_ENERGY_MIN_WINDOW_MS = 0.75


def format_pre_energy_status(reason_code: str | None, debug: bool = False) -> str:
    reason = str(reason_code or "").strip().lower()
    mapping = {
        "empty_ir": "n/a (empty IR)",
        "not_applicable_phase_mode": "n/a (not applicable for this phase mode)",
        "insufficient_pre_window": "n/a (insufficient pre-window)",
        "insufficient_post_window": "n/a (insufficient post-window)",
        "near_zero_total_energy": "n/a (IR energy too low for this metric)",
        "near_zero_post_energy": "n/a (insufficient post energy)",
        "anchor_near_boundary": "n/a (not reliable after alignment)",
        "split_ir_structure": "n/a (not meaningful for this IR shape)",
    }
    if bool(debug):
        mapping = {
            "empty_ir": "n/a (empty IR)",
            "not_applicable_phase_mode": "n/a (not applicable for this phase mode; asymmetric/minimum-phase IR)",
            "insufficient_pre_window": "n/a (insufficient pre-window after alignment)",
            "insufficient_post_window": "n/a (insufficient post-window after alignment)",
            "near_zero_total_energy": "n/a (IR energy too low for pre/post metric)",
            "near_zero_post_energy": "n/a (post energy too low for reliable ratio)",
            "anchor_near_boundary": "n/a (anchor too close to boundary after alignment)",
            "split_ir_structure": "n/a (IR structure does not support a stable pre/post split)",
        }
    return str(mapping.get(reason, "n/a"))


def format_pre_energy_metric_note(
    *,
    valid: bool,
    reason_code: str | None = None,
) -> str:
    if bool(valid):
        return "ok"
    return format_pre_energy_status(reason_code, debug=False)


def compute_pre_post_energy_metrics(
    ir,
    fs,
    peak_idx=None,
    filter_type=None,
    phase_mode=None,
) -> dict[str, Any]:
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    fs_hz = float(fs) if fs is not None else 0.0
    filter_txt = str(filter_type or "").strip().lower()
    phase_txt = str(phase_mode or "").strip().lower()
    if n <= 0:
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "empty_ir",
            "anchor_idx": 0,
            "window_pre_samples": 0,
            "window_post_samples": 0,
        }

    anchor_guess = peak_idx
    if anchor_guess is None:
        anchor_guess = int(np.argmax(np.abs(x)))
    anchor_idx = int(_resolve_ir_energy_split(x, anchor_guess))
    total_energy = float(np.sum(x * x))
    if total_energy <= float(_PRE_ENERGY_EPS):
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "near_zero_total_energy",
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": 0,
            "window_post_samples": 0,
        }

    if ("min" in filter_txt) or ("minimum" in phase_txt):
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "not_applicable_phase_mode",
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": 0,
            "window_post_samples": 0,
        }

    min_window = int(max(16, round(max(1.0, fs_hz) * (_PRE_ENERGY_MIN_WINDOW_MS / 1000.0))))
    target_window = int(max(min_window, round(max(1.0, fs_hz) * (_PRE_ENERGY_WINDOW_MS / 1000.0))))
    pre_available = int(max(0, anchor_idx))
    post_available = int(max(0, n - anchor_idx))

    if pre_available < min_window or post_available < min_window:
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "anchor_near_boundary",
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": int(pre_available),
            "window_post_samples": int(post_available),
        }

    window_pre = int(min(pre_available, target_window))
    window_post = int(min(post_available, target_window))
    if window_pre < min_window:
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "insufficient_pre_window",
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": int(window_pre),
            "window_post_samples": int(window_post),
        }
    if window_post < min_window:
        return {
            "valid": False,
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": "insufficient_post_window",
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": int(window_pre),
            "window_post_samples": int(window_post),
        }

    pre = np.asarray(x[anchor_idx - window_pre:anchor_idx], dtype=float)
    post = np.asarray(x[anchor_idx:anchor_idx + window_post], dtype=float)
    pre_energy = float(np.sum(pre * pre))
    post_energy = float(np.sum(post * post))
    if post_energy <= float(_PRE_ENERGY_EPS):
        reason_code = "near_zero_post_energy"
        if pre_energy >= (0.9 * max(total_energy, _PRE_ENERGY_EPS)):
            reason_code = "split_ir_structure"
        return {
            "valid": False,
            "pre_energy": float(pre_energy),
            "post_energy": float(post_energy),
            "ratio": float("nan"),
            "pre_ringing_db": float("nan"),
            "reason_code": str(reason_code),
            "anchor_idx": int(anchor_idx),
            "window_pre_samples": int(window_pre),
            "window_post_samples": int(window_post),
        }

    ratio = float(pre_energy / max(post_energy, _PRE_ENERGY_EPS))
    pre_ringing_db = float(10.0 * np.log10(max(ratio, _PRE_ENERGY_EPS)))
    return {
        "valid": True,
        "pre_energy": float(pre_energy),
        "post_energy": float(post_energy),
        "ratio": float(ratio),
        "pre_ringing_db": float(pre_ringing_db),
        "reason_code": "ok",
        "anchor_idx": int(anchor_idx),
        "window_pre_samples": int(window_pre),
        "window_post_samples": int(window_post),
    }


def _summarize_ir_metrics(ir_final, cfg, st) -> dict[str, Any]:
    x = np.asarray(ir_final, dtype=float)
    split_hint = None
    fs_hz = None
    filter_type = ""
    phase_mode = ""
    try:
        fs_hz = getattr(cfg, "fs", None)
    except (AttributeError, TypeError, ValueError):
        fs_hz = None
    try:
        filter_type = str(getattr(cfg, "filter_type_str", "") or "")
    except (AttributeError, TypeError, ValueError):
        filter_type = ""
    try:
        phase_mode = str(getattr(cfg, "ir_anchor_mode", "") or "")
    except (AttributeError, TypeError, ValueError):
        phase_mode = ""
    try:
        if isinstance(st, dict):
            split_hint = st.get("ir_energy_split_samples", None)
    except (TypeError, ValueError):
        split_hint = None

    if x.size == 0:
        out = {
            "ir_len": 0,
            "ir_peak_samples": 0,
            "ir_peak_db": float("-inf"),
            "ir_rms": 0.0,
            "ir_pre_ringing_db": float("nan"),
            "ir_pre_post_ratio": float("nan"),
            "ir_pre_ringing_db_raw": float("nan"),
            "ir_pre_post_ratio_raw": float("nan"),
            "ir_pre_energy_split_samples": 0,
            "pre_energy_metric_valid": False,
            "pre_energy_metric_suspect": True,
            "pre_energy_metric_reason_code": "empty_ir",
            "pre_energy_metric_note": format_pre_energy_metric_note(valid=False, reason_code="empty_ir"),
            "ir_pre_energy": 0.0,
            "ir_post_energy": 0.0,
            "ir_pre_energy_window_pre_samples": 0,
            "ir_pre_energy_window_post_samples": 0,
        }
    else:
        peak_idx = int(np.argmax(np.abs(x)))
        peak = float(np.max(np.abs(x)))
        energy_metrics = compute_pre_post_energy_metrics(
            x,
            fs_hz,
            peak_idx=split_hint if split_hint is not None else peak_idx,
            filter_type=filter_type,
            phase_mode=phase_mode,
        )
        valid = bool(energy_metrics.get("valid", False))
        ratio = float(energy_metrics.get("ratio", float("nan")))
        pre_db = float(energy_metrics.get("pre_ringing_db", float("nan")))
        reason_code = str(energy_metrics.get("reason_code", "ok") or "ok")
        out = {
            "ir_len": int(x.size),
            "ir_peak_samples": int(peak_idx),
            "ir_peak_db": float(20.0 * np.log10(peak + 1e-30)),
            "ir_rms": float(np.sqrt(np.mean(x * x))),
            "ir_pre_ringing_db": float(pre_db) if valid else float("nan"),
            "ir_pre_post_ratio": float(ratio) if valid else float("nan"),
            "ir_pre_ringing_db_raw": float(pre_db),
            "ir_pre_post_ratio_raw": float(ratio),
            "ir_pre_energy_split_samples": int(energy_metrics.get("anchor_idx", peak_idx)),
            "pre_energy_metric_valid": bool(valid),
            "pre_energy_metric_suspect": bool(not valid),
            "pre_energy_metric_reason_code": str(reason_code),
            "pre_energy_metric_note": format_pre_energy_metric_note(valid=valid, reason_code=reason_code),
            "ir_pre_energy": float(energy_metrics.get("pre_energy", 0.0)),
            "ir_post_energy": float(energy_metrics.get("post_energy", 0.0)),
            "ir_pre_energy_window_pre_samples": int(energy_metrics.get("window_pre_samples", 0)),
            "ir_pre_energy_window_post_samples": int(energy_metrics.get("window_post_samples", 0)),
        }
    try:
        if isinstance(st, dict):
            for k, v in out.items():
                st[k] = v
    except (TypeError, ValueError):
        pass
    return out
