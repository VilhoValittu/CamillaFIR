from __future__ import annotations

import numpy as np

from ..dsp import camillafir_dsp as dsp
from . import shared


def _estimate_auto_mag_c_min_hz(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    default_hz: float = 25.0,
) -> float:
    def _sorted_xy(f, y):
        try:
            ff = np.asarray(f, dtype=float).reshape(-1)
            yy = np.asarray(y, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        if ff.size != yy.size or ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        idx = np.argsort(ff)
        ff = ff[idx]
        yy = yy[idx]
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        ff = ff[m]
        yy = yy[m]
        if ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return ff, yy

    def _f6(ff: np.ndarray, mm: np.ndarray) -> float | None:
        if ff.size < 32 or mm.size != ff.size:
            return None
        try:
            mm_sm, _ = dsp.apply_smoothing_std(
                ff,
                mm,
                np.zeros_like(mm),
                float(shared.AUTO_MODE_MAG_C_MIN_SMOOTH_OCT),
            )
            mm_use = np.asarray(mm_sm, dtype=float)
        except Exception:
            mm_use = np.asarray(mm, dtype=float)

        ref_mask = (
            (ff >= float(shared.AUTO_MODE_MAG_C_MIN_REF_MIN_HZ))
            & (ff <= float(shared.AUTO_MODE_MAG_C_MIN_REF_MAX_HZ))
        )
        if int(np.count_nonzero(ref_mask)) < 8:
            ref_mask = (ff >= 63.0) & (ff <= 250.0)
        if int(np.count_nonzero(ref_mask)) < 8:
            return None

        ref_slice = np.asarray(mm_use[ref_mask], dtype=float)
        ref_slice = ref_slice[np.isfinite(ref_slice)]
        if ref_slice.size < 6:
            return None
        ref_db = float(np.quantile(ref_slice, 0.75))
        thr_db = float(ref_db - 6.0)

        lf_hi = float(
            min(
                float(shared.AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ),
                float(shared.AUTO_MODE_MAG_C_MIN_REF_MIN_HZ),
            )
        )
        lf_mask = (ff >= float(shared.AUTO_MODE_MAG_C_MIN_MIN_HZ)) & (ff <= lf_hi)
        if int(np.count_nonzero(lf_mask)) < 8:
            return None
        f_lo = ff[lf_mask]
        m_lo = mm_use[lf_mask]
        m_env = np.maximum.accumulate(np.asarray(m_lo, dtype=float))
        above = np.asarray(m_env >= thr_db, dtype=bool)
        if not np.any(above):
            return float(shared._auto_safe_float(default_hz, 25.0))

        try:
            df = float(np.median(np.diff(f_lo))) if f_lo.size > 2 else float("nan")
        except Exception:
            df = float("nan")
        if np.isfinite(df) and df > 0.0:
            n_cons = int(max(3, round(3.0 / df)))
        else:
            n_cons = 3
        n_cons = int(np.clip(n_cons, 3, 12))

        i1 = None
        if above.size >= n_cons:
            for i in range(0, int(above.size) - n_cons + 1):
                if bool(np.all(above[i : i + n_cons])):
                    i1 = int(i)
                    break
        if i1 is None:
            return float(shared._auto_safe_float(default_hz, 25.0))
        if i1 <= 0:
            return float(shared.AUTO_MODE_MAG_C_MIN_MIN_HZ)

        x1, y1 = float(f_lo[i1 - 1]), float(m_env[i1 - 1])
        x2, y2 = float(f_lo[i1]), float(m_env[i1])
        if np.isfinite(y2 - y1) and abs(float(y2 - y1)) > 1e-9:
            f6 = float(x1 + (thr_db - y1) * (x2 - x1) / (y2 - y1))
        else:
            f6 = float(x2)
        return float(f6)

    fl, ml = _sorted_xy(f_l, m_l)
    fr, mr = _sorted_xy(f_r, m_r)
    f6_l = _f6(fl, ml)
    f6_r = _f6(fr, mr)
    if f6_l is None or not np.isfinite(f6_l):
        f6_l = None
    if f6_r is None or not np.isfinite(f6_r):
        f6_r = None

    if f6_l is None and f6_r is None:
        est = shared._auto_safe_float(default_hz, 25.0)
    elif f6_l is None:
        est = float(f6_r)
    elif f6_r is None:
        est = float(f6_l)
    elif abs(float(f6_l) - float(f6_r)) <= 8.0:
        est = 0.5 * (float(f6_l) + float(f6_r))
    else:
        est = max(float(f6_l), float(f6_r))

    est = float(
        np.clip(
            est,
            float(shared.AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(shared.AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    return float(round(est, 1))


def _estimate_auto_hpf_from_response(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    default_freq_hz: float = 20.0,
    default_slope_db_oct: int = 24,
) -> dict:
    allowed_slopes = sorted(
        {
            int(v)
            for v in tuple(shared.AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT)
            if int(v) > 0
        }
    )
    if not allowed_slopes:
        allowed_slopes = [24]
    min_hz = float(shared.AUTO_MODE_HPF_MIN_HZ)
    max_hz = float(shared.AUTO_MODE_HPF_MAX_HZ)
    ref_min = float(shared.AUTO_MODE_HPF_REF_MIN_HZ)
    ref_max = float(shared.AUTO_MODE_HPF_REF_MAX_HZ)
    search_max = float(shared.AUTO_MODE_HPF_SEARCH_MAX_HZ)

    def _nearest_slope(v: float) -> int:
        try:
            x = float(v)
        except Exception:
            x = float(default_slope_db_oct)
        return int(min(allowed_slopes, key=lambda s: abs(float(s) - x)))

    def _sorted_xy(f, y):
        try:
            ff = np.asarray(f, dtype=float).reshape(-1)
            yy = np.asarray(y, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        if ff.size != yy.size or ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        idx = np.argsort(ff)
        ff = ff[idx]
        yy = yy[idx]
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        ff = ff[m]
        yy = yy[m]
        if ff.size < 16:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return ff, yy

    def _stable_crossing_hz(fx: np.ndarray, yx: np.ndarray, thr_db: float) -> float | None:
        if fx.size < 8 or yx.size != fx.size:
            return None
        above = np.asarray(yx >= float(thr_db), dtype=bool)
        if not np.any(above):
            return None
        try:
            df = float(np.median(np.diff(fx))) if fx.size > 2 else float("nan")
        except Exception:
            df = float("nan")
        if np.isfinite(df) and df > 0.0:
            n_cons = int(max(3, round(3.0 / df)))
        else:
            n_cons = 3
        n_cons = int(np.clip(n_cons, 3, 12))

        i1 = None
        if above.size >= n_cons:
            for i in range(0, int(above.size) - n_cons + 1):
                if bool(np.all(above[i : i + n_cons])):
                    i1 = int(i)
                    break
        if i1 is None:
            return None
        if i1 <= 0:
            return float(fx[0])
        x1, y1 = float(fx[i1 - 1]), float(yx[i1 - 1])
        x2, y2 = float(fx[i1]), float(yx[i1])
        if np.isfinite(y2 - y1) and abs(float(y2 - y1)) > 1e-9:
            return float(x1 + (float(thr_db) - y1) * (x2 - x1) / (y2 - y1))
        return float(x2)

    def _butter_mag_db(freqs: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
        f = np.asarray(freqs, dtype=float)
        fc = float(cutoff_hz)
        n = int(order)
        with np.errstate(divide="ignore"):
            return -10.0 * np.log10(1.0 + np.power(fc / np.maximum(f, 1e-9), 2 * n))

    def _fit_one_channel(ff: np.ndarray, mm: np.ndarray, *, ch_name: str) -> dict | None:
        if ff.size < 32 or mm.size != ff.size:
            return None
        try:
            mm_sm, _ = dsp.apply_smoothing_std(
                ff,
                mm,
                np.zeros_like(mm),
                float(shared.AUTO_MODE_HPF_SMOOTH_OCT),
            )
            mm_use = np.asarray(mm_sm, dtype=float)
        except Exception:
            mm_use = np.asarray(mm, dtype=float)

        ref_mask = (ff >= ref_min) & (ff <= ref_max)
        if int(np.count_nonzero(ref_mask)) < 8:
            ref_mask = (ff >= 80.0) & (ff <= 320.0)
        if int(np.count_nonzero(ref_mask)) < 8:
            return None
        ref_slice = np.asarray(mm_use[ref_mask], dtype=float)
        ref_slice = ref_slice[np.isfinite(ref_slice)]
        if ref_slice.size < 6:
            return None
        ref_db = float(np.quantile(ref_slice, 0.75))

        lf_hi = float(min(search_max, ref_min))
        lf_mask = (ff >= min_hz) & (ff <= lf_hi)
        if int(np.count_nonzero(lf_mask)) < 10:
            return None
        f_lo = np.asarray(ff[lf_mask], dtype=float)
        m_lo = np.asarray(mm_use[lf_mask], dtype=float)
        m_env = np.maximum.accumulate(m_lo)

        thr3 = float(ref_db - 3.0)
        thr6 = float(ref_db - 6.0)
        thr12 = float(ref_db - 12.0)
        f3 = _stable_crossing_hz(f_lo, m_env, thr3)
        f6 = _stable_crossing_hz(f_lo, m_env, thr6)
        f12 = _stable_crossing_hz(f_lo, m_env, thr12)
        if f3 is None or f6 is None:
            return None
        if not (np.isfinite(f3) and np.isfinite(f6)) or float(f6) >= float(f3):
            return None

        c6 = float((10.0 ** 0.6) - 1.0)
        c12 = float((10.0 ** 1.2) - 1.0)
        n_estimates = []
        try:
            den6 = float(np.log(float(f3) / float(f6)))
            if np.isfinite(den6) and den6 > 1e-9:
                n_estimates.append(float(np.log(c6) / (2.0 * den6)))
        except Exception:
            pass
        if f12 is not None and np.isfinite(f12) and float(f12) < float(f3):
            try:
                den12 = float(np.log(float(f3) / float(f12)))
                if np.isfinite(den12) and den12 > 1e-9:
                    n_estimates.append(float(np.log(c12) / (2.0 * den12)))
            except Exception:
                pass
        if n_estimates:
            n_est = float(np.median(np.asarray(n_estimates, dtype=float)))
        else:
            fit_mask = (
                (f_lo >= min_hz)
                & (f_lo <= min(float(f3) * 0.95, float(np.max(f_lo))))
                & (m_env <= float(ref_db - 6.0))
            )
            if int(np.count_nonzero(fit_mask)) >= 8:
                x = np.log2(np.asarray(f_lo[fit_mask], dtype=float))
                y = np.asarray(m_env[fit_mask], dtype=float)
                try:
                    p = np.polyfit(x, y, 1)
                    slope_db_oct = abs(float(p[0]))
                    n_est = float(np.clip(slope_db_oct / 6.0, 1.0, 8.0))
                except Exception:
                    n_est = float(default_slope_db_oct) / 6.0
            else:
                n_est = float(default_slope_db_oct) / 6.0

        if not np.isfinite(n_est):
            n_est = float(default_slope_db_oct) / 6.0
        n_est = float(np.clip(n_est, 1.0, 8.0))
        slope_db_oct = int(_nearest_slope(6.0 * n_est))
        order = max(1, int(round(float(slope_db_oct) / 6.0)))
        if (f12 is not None and np.isfinite(f12) and float(f12) <= (min_hz * 1.05)) or float(f6) <= (min_hz + 1.8):
            slope_db_oct = int(min(int(slope_db_oct), 24))
            order = max(1, int(round(float(slope_db_oct) / 6.0)))

        ratio6 = float(np.power(c6, 1.0 / (2.0 * float(order))))
        ratio12 = float(np.power(c12, 1.0 / (2.0 * float(order))))
        fc_candidates = []
        if np.isfinite(f3):
            fc_candidates.append(float(f3))
        if np.isfinite(f6):
            fc_candidates.append(float(f6) * ratio6)
        if f12 is not None and np.isfinite(f12):
            fc_candidates.append(float(f12) * ratio12)
        if not fc_candidates:
            return None
        fc_hz = float(np.median(np.asarray(fc_candidates, dtype=float)))
        fc_hz = float(np.clip(fc_hz, min_hz, max_hz))

        fit_mask = (f_lo >= max(min_hz, fc_hz / 4.0)) & (f_lo <= min(search_max, fc_hz * 2.8))
        if int(np.count_nonzero(fit_mask)) < 12:
            fit_mask = (f_lo >= min_hz) & (f_lo <= search_max)
        model = float(ref_db) + _butter_mag_db(f_lo, fc_hz, order)
        if int(np.count_nonzero(fit_mask)) >= 4:
            fit_meas = np.asarray(m_env[fit_mask], dtype=float)
            fit_mod = np.asarray(model[fit_mask], dtype=float)
            delta_db = float(np.median(fit_meas - fit_mod))
            fit_mod = fit_mod + delta_db
            err = np.asarray(fit_mod - fit_meas, dtype=float)
            if err.size >= 8:
                q_lo, q_hi = np.quantile(err, [0.10, 0.90])
                keep = (err >= float(q_lo)) & (err <= float(q_hi))
                err_use = err[keep] if int(np.count_nonzero(keep)) >= 4 else err
            else:
                err_use = err
            fit_rmse_db = float(np.sqrt(np.mean(np.square(err_use))))
        else:
            fit_rmse_db = float("nan")

        conf = 1.0
        if np.isfinite(fit_rmse_db):
            conf *= float(np.clip(1.0 - (fit_rmse_db / 10.0), 0.0, 1.0))
        else:
            conf *= 0.35
        if not (f12 is not None and np.isfinite(f12)):
            conf *= 0.88
        if float(f3) <= (min_hz + 0.5):
            conf *= 0.82
        try:
            expected_ratio = float(np.power(c6, 1.0 / (2.0 * float(order))))
            obs_ratio = float(max(1e-9, float(f3)) / max(1e-9, float(f6)))
            ratio_err = abs(np.log2(max(1e-9, obs_ratio / expected_ratio)))
            conf *= float(np.clip(1.0 - ratio_err / 1.5, 0.45, 1.0))
        except Exception:
            pass
        conf = float(np.clip(conf, 0.0, 1.0))

        return {
            "channel": str(ch_name),
            "freq": float(fc_hz),
            "slope_db_oct": int(slope_db_oct),
            "order": int(order),
            "f3_hz": float(f3),
            "f6_hz": float(f6),
            "f12_hz": float(f12) if (f12 is not None and np.isfinite(f12)) else float("nan"),
            "fit_rmse_db": float(fit_rmse_db),
            "confidence": float(conf),
        }

    fl, ml = _sorted_xy(f_l, m_l)
    fr, mr = _sorted_xy(f_r, m_r)
    ch_l = _fit_one_channel(fl, ml, ch_name="L")
    ch_r = _fit_one_channel(fr, mr, ch_name="R")
    valid = [c for c in (ch_l, ch_r) if isinstance(c, dict)]

    default_freq = float(np.clip(shared._auto_safe_float(default_freq_hz, 20.0), min_hz, max_hz))
    default_slope = int(_nearest_slope(float(default_slope_db_oct)))
    if not valid:
        return {
            "enabled": False,
            "freq": float(round(default_freq, 1)),
            "slope_db_oct": int(default_slope),
            "order": int(max(1, round(float(default_slope) / 6.0))),
            "confidence": 0.0,
            "fit_rmse_db": float("nan"),
            "method": "fallback_default",
            "channels": [],
        }

    if len(valid) == 1:
        v = dict(valid[0])
        freq_hz = float(v.get("freq", default_freq))
        slope_db = int(_nearest_slope(float(v.get("slope_db_oct", default_slope))))
        conf = float(np.clip(shared._auto_safe_float(v.get("confidence", 0.0), 0.0), 0.0, 1.0))
        fit_rmse_db = shared._auto_safe_float(v.get("fit_rmse_db", float("nan")), float("nan"))
    else:
        f1 = float(valid[0].get("freq", default_freq))
        f2 = float(valid[1].get("freq", default_freq))
        c1 = float(np.clip(shared._auto_safe_float(valid[0].get("confidence", 0.0), 0.0), 0.05, 1.0))
        c2 = float(np.clip(shared._auto_safe_float(valid[1].get("confidence", 0.0), 0.0), 0.05, 1.0))
        cw = c1 + c2
        freq_hz = float((f1 * c1 + f2 * c2) / cw) if cw > 0.0 else float(0.5 * (f1 + f2))
        denom = max(1e-9, float(0.5 * (f1 + f2)))
        freq_delta_rel = abs(f1 - f2) / denom
        if np.isfinite(freq_delta_rel) and float(freq_delta_rel) > 0.28:
            freq_hz = float(max(f1, f2))
            conf_agree = 0.78
        else:
            conf_agree = 1.0

        s1 = float(valid[0].get("slope_db_oct", default_slope))
        s2 = float(valid[1].get("slope_db_oct", default_slope))
        slope_w = float((s1 * c1 + s2 * c2) / cw) if cw > 0.0 else float(0.5 * (s1 + s2))
        slope_db = int(_nearest_slope(slope_w))

        rmse_vals = [
            shared._auto_safe_float(valid[0].get("fit_rmse_db", float("nan")), float("nan")),
            shared._auto_safe_float(valid[1].get("fit_rmse_db", float("nan")), float("nan")),
        ]
        rmse_ok = [float(x) for x in rmse_vals if np.isfinite(x)]
        fit_rmse_db = float(np.mean(rmse_ok)) if rmse_ok else float("nan")
        conf = float(np.clip(0.5 * (c1 + c2) * float(conf_agree), 0.0, 1.0))

    freq_hz = float(np.clip(freq_hz, min_hz, max_hz))
    auto_enable = bool(
        np.isfinite(freq_hz)
        and (conf >= float(shared.AUTO_MODE_HPF_AUTO_ENABLE_MIN_CONF))
        and (min_hz <= freq_hz <= max_hz)
    )

    return {
        "enabled": bool(auto_enable),
        "freq": float(round(freq_hz, 1)),
        "slope_db_oct": int(slope_db),
        "order": int(max(1, round(float(slope_db) / 6.0))),
        "confidence": float(round(conf, 3)),
        "fit_rmse_db": float(fit_rmse_db) if np.isfinite(fit_rmse_db) else float("nan"),
        "method": "response_fit",
        "channels": valid,
    }


def _resolve_auto_hpf_application(auto_hpf: dict | None, *, user_hpf_enabled: bool) -> dict:
    meta = dict(auto_hpf or {})
    method = str(meta.get("method", "") or "").strip().lower()
    enabled = bool(meta.get("enabled", False))

    if bool(user_hpf_enabled) and method == "response_fit":
        applied = True
        decision = "apply_user_hpf_fit"
    elif enabled and method == "response_fit":
        applied = False
        decision = "suggest_only"
    else:
        applied = False
        decision = "skip"

    meta["applied"] = bool(applied)
    meta["decision"] = str(decision)
    return meta


__all__ = [
    "_estimate_auto_hpf_from_response",
    "_estimate_auto_mag_c_min_hz",
    "_resolve_auto_hpf_application",
]
