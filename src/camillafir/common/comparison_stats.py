from __future__ import annotations

import numpy as np


def _make_comparison_stats(stats: dict, ref_fs: int = 44100, ref_taps: int = 65536) -> dict:
    stats = stats or {}
    out = dict(stats)

    if str(out.get("analysis_mode", "native")).lower() == "comparison" and ("cmp_freq_axis" in out):
        return out
    f = out.get("freq_axis", None)
    m = out.get("measured_mags", None)
    t = out.get("target_mags", None)
    g = out.get("filter_mags", None)
    c = out.get("confidence_mask", None)
    mm = out.get("mag_mask", out.get("mask_c", None))

    if f is None or m is None or t is None:
        return out

    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if g is not None:
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    if c is not None:
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    if mm is not None:
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        f = np.asarray(f, dtype=float)
        m = np.asarray(m, dtype=float)
        t = np.asarray(t, dtype=float)
        g = np.asarray(g, dtype=float) if g is not None else None
        c = np.asarray(c, dtype=float) if c is not None else None
        mm = np.asarray(mm, dtype=float) if mm is not None else None
    except Exception:
        return out

    if f.ndim != 1 or f.size < 32 or m is None or t is None:
        return out
    if (m.ndim != 1) or (t.ndim != 1) or (m.size != f.size) or (t.size != f.size):
        return out
    if (g is not None) and ((g.ndim != 1) or (g.size != f.size)):
        g = None
    if (c is not None) and ((c.ndim != 1) or (c.size != f.size)):
        c = None
    if (mm is not None) and ((mm.ndim != 1) or (mm.size != f.size)):
        mm = None

    nfft = int(ref_taps)
    if nfft < 1024:
        nfft = 1024
    if (nfft % 2) != 0:
        nfft += 1
    fmax = min(float(ref_fs) / 2.0, float(np.max(f)))
    if fmax <= 10.0:
        return out

    freq_cmp = np.linspace(0.0, fmax, nfft // 2 + 1)

    def _interp(y):
        y = np.asarray(y, dtype=float)
        if y.shape != f.shape:
            return None
        return np.interp(freq_cmp, f, y)

    m_cmp = _interp(m)
    t_cmp = _interp(t)
    g_cmp = _interp(g) if g is not None and g.shape == f.shape else None
    c_cmp = _interp(c) if c is not None and c.shape == f.shape else None
    mm_cmp = _interp(mm) if mm is not None and mm.shape == f.shape else None
    if m_cmp is None or t_cmp is None:
        return out

    rng = out.get("smart_scan_range", None)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        fmin, fmax_rng = float(rng[0]), float(rng[1])
    else:
        fmin, fmax_rng = 200.0, 5000.0
    mask = (freq_cmp >= fmin) & (freq_cmp <= fmax_rng)
    if np.count_nonzero(mask) >= 20:
        cmp_offset_db = float(np.median((m_cmp - t_cmp)[mask]))
    else:
        cmp_offset_db = 0.0
    m_cmp = m_cmp - cmp_offset_db

    out["analysis_mode"] = "comparison"
    out["cmp_ref_fs"] = int(ref_fs)
    out["cmp_ref_taps"] = int(ref_taps)
    out["cmp_freq_axis"] = freq_cmp.tolist()
    out["cmp_measured_mags"] = m_cmp.tolist()
    out["cmp_target_mags"] = t_cmp.tolist()
    if g_cmp is not None:
        out["cmp_filter_mags"] = g_cmp.tolist()
    if c_cmp is not None:
        out["cmp_confidence_mask"] = np.clip(c_cmp, 0.0, 1.0).tolist()
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)
    if mm_cmp is not None:
        out["cmp_mag_mask"] = (np.asarray(mm_cmp, dtype=float) > 0.5).astype(float).tolist()
    bw = out.get("afdw_bw_oct", None)
    bw_cmp = _interp(bw) if bw is not None and np.asarray(bw).shape == f.shape else None
    if bw_cmp is not None:
        out["cmp_afdw_bw_oct"] = np.clip(bw_cmp, 1.0 / 96.0, 1.0 / 3.0).tolist()
        out["cmp_offset_db"] = float(cmp_offset_db)

    if "smart_scan_range" in out and isinstance(out["smart_scan_range"], (list, tuple)) and len(out["smart_scan_range"]) == 2:
        out["cmp_smart_scan_range"] = [float(out["smart_scan_range"][0]), float(out["smart_scan_range"][1])]

    if c_cmp is not None:
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)

    if "eff_target_db" in stats and stats.get("eff_target_db") is not None:
        try:
            v = float(stats.get("eff_target_db"))
            if np.isfinite(v):
                out["eff_target_db"] = v
                out["cmp_eff_target_db"] = v
        except Exception:
            pass

    if "target_level_db_window" in stats:
        out["cmp_target_level_db_window"] = stats.get("target_level_db_window")

    return out


__all__ = ["_make_comparison_stats"]
