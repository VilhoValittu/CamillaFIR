from __future__ import annotations

import numpy as np


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _cosine_fade_out_01(x: np.ndarray) -> np.ndarray:
    """
    x in [0..1] -> weight in [1..0] with cosine shape.
    0 -> 1.0
    1 -> 0.0
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * x)


def _mixed_excess_weight(freqs: np.ndarray, full_hz: float, none_hz: float) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    w = np.zeros_like(f, dtype=float)
    valid = np.isfinite(f) & (f > 0.0)
    if not np.any(valid):
        return w

    f_full = float(max(20.0, full_hz))
    f_none = float(max(f_full + 1.0, none_hz))

    low = valid & (f <= f_full)
    w[low] = 1.0
    mid = valid & (f > f_full) & (f < f_none)
    if np.any(mid):
        x = (f[mid] - f_full) / (f_none - f_full)
        w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))
    return np.clip(w, 0.0, 1.0)


def _max_abs_group_delay_ms(freqs: np.ndarray, phase_rad: np.ndarray, mask: np.ndarray | None = None) -> float:
    f = np.asarray(freqs, dtype=float)
    p = np.unwrap(np.asarray(phase_rad, dtype=float))
    if f.size < 4 or p.size != f.size:
        return 0.0
    w = 2.0 * np.pi * f
    dw = np.gradient(w) + 1e-30
    gd_ms = (-np.gradient(p) / dw) * 1000.0
    if mask is None:
        sel = np.isfinite(gd_ms)
    else:
        sel = np.asarray(mask, dtype=bool) & np.isfinite(gd_ms)
    if not np.any(sel):
        return 0.0
    try:
        return float(np.max(np.abs(gd_ms[sel])))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0.0


def _resolve_ir_energy_split(ir: np.ndarray, split: int | None = None) -> int:
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return 0
    if split is None:
        split = int(np.argmax(np.abs(x)))
    try:
        s = int(split)
    except (TypeError, ValueError, OverflowError):
        s = int(np.argmax(np.abs(x)))
    if n <= 16:
        return int(np.clip(s, 1, max(1, n - 1)))
    return int(np.clip(s, 8, n - 8))


def _pre_post_energy_ratio(ir: np.ndarray, split: int | None = None, *, eps: float = 1e-20) -> float:
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    if n < 32:
        return float("nan")
    s = int(_resolve_ir_energy_split(x, split))
    if s <= 0 or s >= n:
        return float("nan")
    pre = x[:s]
    post = x[s:]
    if pre.size == 0 or post.size == 0:
        return float("nan")
    epre = float(np.mean(pre * pre))
    epost = float(np.mean(post * post))
    eps_v = float(max(float(eps), 1e-20))
    return float(epre / max(epost, eps_v))


def _pre_ringing_db(ir: np.ndarray, split: int | None = None) -> float:
    ratio = float(_pre_post_energy_ratio(ir, split=split, eps=1e-20))
    if not np.isfinite(ratio):
        return float("nan")
    return float(10.0 * np.log10(max(ratio, 1e-20)))


def _shift_zeropad(x: np.ndarray, shift_samp: int) -> np.ndarray:
    n_loc = int(len(x))
    if n_loc == 0 or int(shift_samp) == 0:
        return x
    if shift_samp > 0:
        if shift_samp >= n_loc:
            return np.zeros_like(x)
        return np.concatenate((np.zeros(shift_samp, dtype=x.dtype), x[:-shift_samp]))
    s = -int(shift_samp)
    if s >= n_loc:
        return np.zeros_like(x)
    return np.concatenate((x[s:], np.zeros(s, dtype=x.dtype)))


def _ms_value(cfg, name_ms: str, name_alias: str, default: float = 0.0) -> float:
    v = getattr(cfg, name_ms, None)
    if v is None:
        v = getattr(cfg, name_alias, default)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _resolve_ir_window_mode(cfg, logger=None) -> tuple[str, str, bool]:
    requested_win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
    is_min_filter = ("Min" in cfg.filter_type_str)
    min_strict_off = bool(getattr(cfg, "min_strict_off", False))
    win_mode = "off" if (is_min_filter and min_strict_off) else requested_win_mode
    if is_min_filter and min_strict_off and requested_win_mode != "off" and logger is not None:
        logger.info(
            f"IR export: forcing mode=off for Minimum filter (requested={requested_win_mode}, strict)"
        )
    return requested_win_mode, win_mode, is_min_filter


def _resolve_ir_anchor_mode(cfg) -> str:
    mode = str(getattr(cfg, "ir_anchor_mode", "peak") or "peak").strip().lower()
    if mode in ("peak", "centroid", "min_causal"):
        return mode
    return "peak"
