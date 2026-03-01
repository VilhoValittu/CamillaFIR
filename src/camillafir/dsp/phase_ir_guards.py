from __future__ import annotations

from typing import Any

import numpy as np

from .phase_ir_utils import (
    _pre_post_energy_ratio,
    _pre_ringing_db,
    _resolve_ir_anchor_mode,
    _resolve_ir_energy_split,
)


def _resolve_guard_split_index(x: np.ndarray, cfg, st) -> int:
    n = int(x.size)
    if n <= 1:
        return 0
    candidates: list[int] = []
    if isinstance(st, dict):
        for k in (
            "ir_energy_split_samples",
            "post_align_anchor_after_samples",
            "post_align_peak_after_samples",
            "pre_align_anchor_after_samples",
            "pre_align_peak_after_samples",
        ):
            try:
                v = int(st.get(k))
            except Exception:
                continue
            candidates.append(v)
    for idx in candidates:
        if 0 < idx < n:
            return int(_resolve_ir_energy_split(x, idx))

    anchor_mode = _resolve_ir_anchor_mode(cfg)
    if anchor_mode == "centroid":
        w = np.abs(x)
        w_sum = float(np.sum(w))
        if np.isfinite(w_sum) and w_sum > 1e-30:
            idx = float(np.sum(np.arange(n, dtype=float) * w) / w_sum)
            if np.isfinite(idx):
                return int(_resolve_ir_energy_split(x, int(round(idx))))
    return int(_resolve_ir_energy_split(x, int(np.argmax(np.abs(x)))))


def _pre_energy_guard(ir, cfg, st) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(ir, dtype=float)
    try:
        ratio_max = float(getattr(cfg, "pre_energy_ratio_max", 0.25) or 0.25)
    except Exception:
        ratio_max = 0.25
    if not np.isfinite(ratio_max):
        ratio_max = 0.25
    ratio_max = float(max(0.0, ratio_max))
    try:
        strength = float(getattr(cfg, "pre_energy_guard_strength", 0.8) or 0.8)
    except Exception:
        strength = 0.8
    if not np.isfinite(strength):
        strength = 0.8
    strength = float(np.clip(strength, 0.0, 1.0))
    split_idx = int(_resolve_guard_split_index(x, cfg, st))
    split_idx = int(_resolve_ir_energy_split(x, split_idx))
    try:
        min_pre_scale = float(getattr(cfg, "pre_energy_guard_min_scale", 0.08) or 0.08)
    except Exception:
        min_pre_scale = 0.08
    min_pre_scale = float(np.clip(min_pre_scale, 1e-3, 1.0))
    try:
        taper_floor = float(getattr(cfg, "pre_energy_guard_taper_floor", 0.05) or 0.05)
    except Exception:
        taper_floor = 0.05
    taper_floor = float(np.clip(taper_floor, 1e-6, 1.0))
    try:
        taper_power = float(getattr(cfg, "pre_energy_guard_taper_power", 2.0) or 2.0)
    except Exception:
        taper_power = 2.0
    taper_power = float(np.clip(taper_power, 1.0, 4.0))
    info = {
        "enabled": bool(getattr(cfg, "enable_ir_pre_energy_guard", False)),
        "before_db": float(_pre_ringing_db(x, split=split_idx)),
        "after_db": float(_pre_ringing_db(x, split=split_idx)),
        "before_ratio": 0.0,
        "after_ratio": 0.0,
        "ratio_max": float(ratio_max),
        "strength": float(strength),
        "split_samples": int(split_idx),
        "scale": 1.0,
        "triggered": False,
        "reason": "not_triggered",
    }

    def _store_guard_info() -> None:
        try:
            if isinstance(st, dict):
                st["ir_pre_energy_guard_enabled"] = bool(info["enabled"])
                st["ir_pre_energy_guard_triggered"] = bool(info["triggered"])
                st["ir_pre_energy_guard_scale"] = float(info["scale"])
                st["ir_pre_energy_guard_before_db"] = float(info["before_db"])
                st["ir_pre_energy_guard_after_db"] = float(info["after_db"])
                st["ir_pre_energy_guard_before_ratio"] = float(info["before_ratio"])
                st["ir_pre_energy_guard_after_ratio"] = float(info["after_ratio"])
                st["ir_pre_energy_guard_ratio_max"] = float(info["ratio_max"])
                st["ir_pre_energy_guard_strength"] = float(info["strength"])
                st["ir_pre_energy_guard_split_samples"] = int(info["split_samples"])
                st["ir_pre_energy_guard_reason"] = str(info["reason"])
        except Exception:
            pass

    if not info["enabled"]:
        info["reason"] = "disabled"
        _store_guard_info()
        return x, info
    try:
        limit_db = float(getattr(cfg, "max_pre_ringing_db", -35.0) or -35.0)
    except Exception:
        limit_db = -35.0
    if not np.isfinite(limit_db):
        info["reason"] = "bad_limit_db"
        _store_guard_info()
        return x, info
    limit_db = float(min(limit_db, 0.0))
    split = int(split_idx)
    if split <= 0 or split >= int(x.size):
        info["reason"] = "split<=0"
        _store_guard_info()
        return x, info
    ratio_now = float(_pre_post_energy_ratio(x, split=split, eps=1e-20))
    before_db = float(_pre_ringing_db(x, split=split))
    if (not np.isfinite(ratio_now)) or (not np.isfinite(before_db)):
        info["reason"] = "missing_data"
        _store_guard_info()
        return x, info
    info["before_db"] = before_db
    info["before_ratio"] = float(ratio_now)
    info["after_ratio"] = float(ratio_now)
    db_ratio_target = float(10.0 ** (limit_db / 10.0))
    if ratio_max > 0.0:
        ratio_target = float(ratio_max)
    else:
        ratio_target = float(max(db_ratio_target, 1e-12))
    info["ratio_max"] = float(ratio_target)
    trigger = bool(ratio_now > ratio_target)
    if trigger:
        full_scale = float(np.clip(np.sqrt(ratio_target / max(ratio_now, 1e-30)), min_pre_scale, 1.0))
        y = x.copy()
        y_full = x.copy()
        y_full[:split] *= full_scale
        if split > 1:
            progress = np.linspace(0.0, 1.0, split, endpoint=True, dtype=float)
            taper = (0.5 + 0.5 * np.cos(np.pi * progress)).astype(float)
            taper = np.clip(taper, taper_floor, 1.0)
            taper = taper**taper_power
            y_full[:split] *= taper
        y[:split] = (1.0 - strength) * y[:split] + strength * y_full[:split]
        ratio_after = float(_pre_post_energy_ratio(y, split=split, eps=1e-20))
        if np.isfinite(ratio_after) and ratio_after > ratio_target:
            extra = float(np.clip(np.sqrt(ratio_target / max(ratio_after, 1e-30)), min_pre_scale, 1.0))
            y[:split] *= extra
            ratio_after = float(_pre_post_energy_ratio(y, split=split, eps=1e-20))
            info["scale"] = float(extra)
        in_pre_rms = float(np.sqrt(np.mean(x[:split] * x[:split])))
        out_pre_rms = float(np.sqrt(np.mean(y[:split] * y[:split])))
        if in_pre_rms > 1e-20:
            info["scale"] = float(out_pre_rms / in_pre_rms)
        # Guard should not hard-zero pre segment unless input was already zero there.
        if np.all(np.abs(y[:split]) < 1e-15):
            y[:split] = (1.0 - min_pre_scale) * y[:split] + min_pre_scale * x[:split]
            ratio_after = float(_pre_post_energy_ratio(y, split=split, eps=1e-20))
        info["triggered"] = True
        info["after_ratio"] = float(ratio_after)
        info["after_db"] = float(_pre_ringing_db(y, split=split))
        if full_scale <= (min_pre_scale + 1e-12) and np.isfinite(ratio_after) and ratio_after > ratio_target:
            info["reason"] = "applied_floor_limited"
        else:
            info["reason"] = "applied"
        _store_guard_info()
        return y, info
    info["after_db"] = float(before_db)
    info["reason"] = "within_limit"
    _store_guard_info()
    return x, info


def _tdc_postprocess(ir, cfg, st) -> np.ndarray:
    out = np.asarray(ir, dtype=float)
    try:
        if isinstance(st, dict):
            st["tdc_postprocess_applied"] = False
    except Exception:
        pass
    return out
