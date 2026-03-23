from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.ndimage

from .smoothing import psycho_smooth_safe_gain


def apply_bass_boost_post_restore(
    gain_db: np.ndarray,
    target_db: np.ndarray,
    boost_cap_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    hz_lo: float = 20.0,
    hz_hi: float = 200.0,
    strength: float = 0.6,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    out = np.asarray(gain_db, dtype=float).copy()
    tgt = np.asarray(target_db, dtype=float)
    cap = np.asarray(boost_cap_db, dtype=float)
    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(mask_c, dtype=bool)
    n = int(min(out.size, tgt.size, cap.size, f.size, m.size))
    meta: dict[str, float | int | bool] = {
        "enabled": False,
        "bins": 0,
        "delta_rms_20_200": 0.0,
        "delta_max_20_200": 0.0,
    }
    if n < 8:
        return out, meta
    out = out[:n].copy()
    tgt = tgt[:n]
    cap = cap[:n]
    f = f[:n]
    m = m[:n]
    s = float(np.clip(float(strength), 0.0, 1.0))
    if s <= 0.0:
        return out, meta
    valid = np.isfinite(out) & np.isfinite(tgt) & np.isfinite(cap) & np.isfinite(f)
    rmask = (
        valid
        & m
        & (f >= float(hz_lo))
        & (f <= float(hz_hi))
        & (cap > out + 1e-9)
        & (tgt > out + 1e-9)
    )
    if not np.any(rmask):
        return out, meta
    pre = out.copy()
    out[rmask] = out[rmask] + s * (tgt[rmask] - out[rmask])
    out[rmask] = np.minimum(out[rmask], cap[rmask])
    b20 = valid & (f >= 20.0) & (f <= 200.0)
    d = out - pre
    if np.any(b20):
        meta["delta_rms_20_200"] = float(np.sqrt(np.mean(d[b20] * d[b20])))
        meta["delta_max_20_200"] = float(np.max(np.abs(d[b20])))
    meta["enabled"] = True
    meta["bins"] = int(np.count_nonzero(rmask))
    return out, meta


def _ps_invalid(ps: np.ndarray | None, mask: np.ndarray) -> bool:
    return ps is None or ps.shape != mask.shape


def apply_confpull_post_slope(
    gain_db_in: np.ndarray,
    mask_c_in: np.ndarray,
    measured_ref_db: np.ndarray | None,
    *,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray,
    freq_axis: np.ndarray,
    logger: Any,
    apply_confidence_weighted_target_pull: Callable[..., Any],
) -> np.ndarray:
    try:
        if gain_db_in is None or mask_c_in is None:
            return gain_db_in
        if not (isinstance(gain_db_in, np.ndarray) and isinstance(mask_c_in, np.ndarray)):
            return gain_db_in
        if gain_db_in.size < 16 or gain_db_in.shape != mask_c_in.shape or not np.any(mask_c_in):
            return gain_db_in
        conf_floor = float(getattr(cfg, "conf_pull_floor", 0.05) or 0.05)
        conf_ceil = float(getattr(cfg, "conf_pull_ceil", 0.95) or 0.95)
        conf_max_hz = getattr(cfg, "conf_pull_max_hz", 200.0)
        conf_max_hz = None if conf_max_hz is None else float(conf_max_hz)
        gamma_cut = float(getattr(cfg, "conf_pull_gamma_cut", 0.55) or 0.55)
        gamma_boost = float(getattr(cfg, "conf_pull_gamma_boost", 1.35) or 1.35)
        conf_sigma = float(getattr(cfg, "conf_pull_conf_smooth_sigma", 2.0) or 2.0)
        bass_floor_hz = float(getattr(cfg, "conf_pull_bass_floor_hz", 120.0) or 120.0)
        bass_floor_min = float(getattr(cfg, "conf_pull_bass_floor_min", 0.25) or 0.25)
        bass_boost_floor_hz = float(getattr(cfg, "conf_pull_bass_boost_floor_hz", 200.0) or 200.0)
        bass_boost_floor_min = float(getattr(cfg, "conf_pull_bass_boost_floor_min", 0.45) or 0.45)
        bass_boost_restore = float(getattr(cfg, "conf_pull_bass_boost_restore", 0.55) or 0.0)
        bass_adaptive_isolation_mode = bool(getattr(cfg, "bass_adaptive_isolation_mode", False))
        if bass_adaptive_isolation_mode:
            bass_boost_floor_hz = 0.0
            bass_boost_floor_min = 0.0
            bass_boost_restore = 0.0
        if not np.isfinite(conf_sigma) or conf_sigma < 0.0:
            conf_sigma = 0.0
        bass_floor_hz = float(np.clip(bass_floor_hz if np.isfinite(bass_floor_hz) else 0.0, 0.0, np.inf))
        bass_floor_min = float(np.clip(bass_floor_min if np.isfinite(bass_floor_min) else 0.0, 0.0, 1.0))
        bass_boost_floor_hz = float(np.clip(bass_boost_floor_hz if np.isfinite(bass_boost_floor_hz) else 0.0, 0.0, np.inf))
        bass_boost_floor_min = float(np.clip(bass_boost_floor_min if np.isfinite(bass_boost_floor_min) else 0.0, 0.0, 1.0))
        bass_boost_restore = float(np.clip(bass_boost_restore if np.isfinite(bass_boost_restore) else 0.0, 0.0, 1.0))
        try:
            if isinstance(st, dict):
                st["bass_adaptive_isolation_mode"] = bool(bass_adaptive_isolation_mode)
                st["conf_pull_post_bass_boost_floor_hz"] = float(bass_boost_floor_hz)
                st["conf_pull_post_bass_boost_floor_min"] = float(bass_boost_floor_min)
                st["conf_pull_post_bass_boost_restore"] = float(bass_boost_restore)
        except (TypeError, ValueError):
            pass
        conf_for_pull = conf_mask
        try:
            c0 = np.asarray(conf_mask, dtype=float)
            if c0.shape == gain_db_in.shape:
                if conf_sigma > 0.0:
                    c0 = scipy.ndimage.gaussian_filter1d(c0, sigma=float(conf_sigma))
                c0 = np.clip(c0, 0.0, 1.0)
                if bass_floor_hz > 0.0 and bass_floor_min > 0.0:
                    f0 = np.asarray(freq_axis, dtype=float)
                    bm = (f0 > 0.0) & (f0 <= float(bass_floor_hz))
                    if np.any(bm):
                        c0[bm] = np.maximum(c0[bm], float(bass_floor_min))
                if bass_boost_floor_hz > 0.0 and bass_boost_floor_min > 0.0:
                    f0 = np.asarray(freq_axis, dtype=float)
                    g0 = np.asarray(gain_db_in, dtype=float)
                    bb = (f0 > 0.0) & (f0 <= float(bass_boost_floor_hz)) & (g0 > 0.0)
                    if np.any(bb):
                        c0[bb] = np.maximum(c0[bb], float(bass_boost_floor_min))
                conf_for_pull = np.clip(c0, 0.0, 1.0)
        except (TypeError, ValueError, FloatingPointError, IndexError):
            conf_for_pull = conf_mask
        try:
            g_ref = np.asarray(measured_ref_db, dtype=float) if measured_ref_db is not None else None
            if g_ref is not None and g_ref.shape != gain_db_in.shape:
                g_ref = None
        except (TypeError, ValueError):
            g_ref = None
        if g_ref is None:
            try:
                g_in = np.asarray(gain_db_in, dtype=float).copy()
                idx = np.where(mask_c_in)[0]
                i0, i1 = int(idx[0]), int(idx[-1])
                if i0 > 0:
                    g_in[:i0] = g_in[i0]
                if i1 < (g_in.size - 1):
                    g_in[i1 + 1:] = g_in[i1]
                g_ref = psycho_smooth_safe_gain(freq_axis, g_in)
            except (TypeError, ValueError, FloatingPointError, IndexError):
                g_ref = np.asarray(gain_db_in, dtype=float)
        g_ref = np.where(mask_c_in, np.asarray(g_ref, dtype=float), gain_db_in)
        try:
            g_cur = np.asarray(gain_db_in, dtype=float)
            g_ref = np.where(g_cur > 0.0, np.minimum(g_ref, g_cur), np.minimum(g_ref, 0.0))
        except (TypeError, ValueError, FloatingPointError):
            pass
        out = apply_confidence_weighted_target_pull(
            target_db=gain_db_in,
            measured_db=g_ref,
            confidence_mask=conf_for_pull,
            conf_floor=conf_floor,
            conf_ceil=conf_ceil,
            freq_axis=freq_axis,
            freq_limit_hz=conf_max_hz,
            gamma_cut=gamma_cut,
            gamma_boost=gamma_boost,
            return_telemetry=True,
        )
        gain_out, tel = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
        gain_out = np.where(mask_c_in, np.asarray(gain_out, dtype=float), gain_db_in)
        try:
            if bass_boost_floor_hz > 0.0 and bass_boost_restore > 0.0:
                f0 = np.asarray(freq_axis, dtype=float)
                gt = np.asarray(gain_db_in, dtype=float)
                go = np.asarray(gain_out, dtype=float)
                cp = np.asarray(conf_for_pull, dtype=float)
                bb = mask_c_in & (f0 > 0.0) & (f0 <= float(bass_boost_floor_hz)) & (gt > 0.0)
                if np.any(bb):
                    w = np.clip(
                        (cp[bb] - float(bass_boost_floor_min)) / max(1e-9, 1.0 - float(bass_boost_floor_min)),
                        0.0,
                        1.0,
                    )
                    restore = np.clip(float(bass_boost_restore) * w, 0.0, 1.0)
                    gb = go[bb]
                    go[bb] = gb + restore * (gt[bb] - gb)
                    gain_out = np.where(mask_c_in, go, gain_db_in)
                    if isinstance(st, dict):
                        st["conf_pull_post_bass_boost_restore"] = float(bass_boost_restore)
                        st["conf_pull_post_bass_boost_restore_mean_eff"] = float(np.mean(restore))
                        st["conf_pull_post_bass_boost_restore_max_eff"] = float(np.max(restore))
                        st["conf_pull_post_bass_boost_restore_bins"] = int(np.count_nonzero(bb))
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass
        try:
            w_eff = np.asarray(tel.get("w_eff"), dtype=float) if isinstance(tel, dict) and tel.get("w_eff") is not None else None
            pm = np.asarray(tel.get("pull_mask"), dtype=bool) if isinstance(tel, dict) and tel.get("pull_mask") is not None else None
            ps = np.asarray(tel.get("pull_strength"), dtype=float) if isinstance(tel, dict) and tel.get("pull_strength") is not None else None
            pm2 = mask_c_in if (pm is None or pm.shape != mask_c_in.shape) else (pm & mask_c_in)
            if (w_eff is not None) and (w_eff.shape == pm2.shape) and np.any(pm2):
                wv = w_eff[pm2]
                pv = np.clip(1.0 - wv, 0.0, 1.0) if _ps_invalid(ps, pm2) else ps[pm2]
                act = pv > 0.05
                n_mask = int(np.count_nonzero(pm2))
                n_act = int(np.count_nonzero(act))
                act_pct = 100.0 * n_act / max(1, n_mask)
                w_mean = float(np.mean(wv))
                p_mean = float(np.mean(pv))
                p_max = float(np.max(pv))
                f_pull_max = None
                try:
                    idxs = np.where(pm2)[0]
                    f_pull_max = float(freq_axis[int(idxs[int(np.argmax(pv))])])
                except (TypeError, ValueError, IndexError):
                    f_pull_max = None
                logger.info(
                    "ConfPullPost: "
                    f"mask_bins={n_mask}, active_bins={n_act} ({act_pct:.1f}%), "
                    f"w_eff(mean={w_mean:.3f}), pull_strength(mean={p_mean:.3f}, max={p_max:.3f}), "
                    f"floor={conf_floor:.3f}, ceil={conf_ceil:.3f}, max_hz={conf_max_hz}, "
                    f"gamma_cut={gamma_cut:.2f}, gamma_boost={gamma_boost:.2f}"
                )
                if isinstance(st, dict):
                    st["conf_pull_post_floor"] = float(conf_floor)
                    st["conf_pull_post_ceil"] = float(conf_ceil)
                    st["conf_pull_post_max_hz"] = None if conf_max_hz is None else float(conf_max_hz)
                    st["conf_pull_post_gamma_cut"] = float(gamma_cut)
                    st["conf_pull_post_gamma_boost"] = float(gamma_boost)
                    st["conf_pull_post_active_pct"] = float(act_pct)
                    st["conf_pull_post_w_eff_mean"] = float(w_mean)
                    st["conf_pull_post_strength_mean"] = float(p_mean)
                    st["conf_pull_post_strength_max"] = float(p_max)
                    st["conf_pull_post_strength_max_hz"] = float(f_pull_max) if f_pull_max is not None else None
                    st["conf_pull_post_conf_smooth_sigma"] = float(conf_sigma)
                    st["conf_pull_post_bass_floor_hz"] = float(bass_floor_hz)
                    st["conf_pull_post_bass_floor_min"] = float(bass_floor_min)
                    st["conf_pull_post_bass_boost_floor_hz"] = float(bass_boost_floor_hz)
                    st["conf_pull_post_bass_boost_floor_min"] = float(bass_boost_floor_min)
                    st["conf_pull_post_bass_boost_restore"] = float(bass_boost_restore)
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass
        return gain_out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return gain_db_in
