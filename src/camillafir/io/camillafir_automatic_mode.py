import logging
import math
import json
import os
import hashlib
import time

import numpy as np

from ..dsp import camillafir_dsp as dsp
from ..dsp.target_match import target_match_from_stats
from ..engine import build_config, run_pipeline, summarize_run
from ..ui import camillafir_plot as plots
from ..ui.camillafir_housecurve import get_house_curve_by_name

logger = logging.getLogger("CamillaFIR")
MAX_SAFE_BOOST = 8.0
AUTO_MODE_TRIALS = 100
AUTO_MODE_REFINE_TRIALS = 50

# --- Auto-mode preset cache ---
# Stores best preset per (measurement + key settings) signature, so next run can start from it.
AUTO_MODE_CACHE_ENABLED = True
AUTO_MODE_CACHE_MAX_ITEMS = 64
AUTO_MODE_CACHE_FILENAME = "camillafir_auto_mode_cache.json"


def _auto_cache_path() -> str:
    # Cross-platform: ~/.camillafir/cache.json
    base = os.path.join(os.path.expanduser("~"), ".camillafir")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, AUTO_MODE_CACHE_FILENAME)


AUTO_MODE_PHASE1_PLATEAU_ROUNDS = 5
AUTO_MODE_PHASE2_PLATEAU_ROUNDS = 8
AUTO_MODE_TARGET_TOP_N = 3
AUTO_MODE_TARGET_TRIALS_PER_CURVE = 10
AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT = 0.75
AUTO_MODE_TARGET_PREFER_MILDER_STEP = True
AUTO_MODE_TARGET_MILDER_MAX_RANK_DROP = 1.50
AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB = 0.25
AUTO_MODE_MAG_C_MIN_MIN_HZ = 15.0
AUTO_MODE_MAG_C_MIN_MAX_HZ = 70.0
AUTO_MODE_MAG_C_MIN_REF_MIN_HZ = 80.0
AUTO_MODE_MAG_C_MIN_REF_MAX_HZ = 200.0
AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ = 80.0
AUTO_MODE_MAG_C_MIN_SMOOTH_OCT = 1.0
AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ = 2.0
AUTO_MODE_LOW_BASS_MIN_HZ = 18.0
AUTO_MODE_LOW_BASS_MAX_HZ = 55.0
AUTO_MODE_EXC_FROM_F6_ADD_HZ = 8.0
AUTO_MODE_EXC_MIN_HZ = 20.0
AUTO_MODE_EXC_MAX_HZ = 80.0
AUTO_MODE_BUILTIN_TARGETS = (
    "Harman6",
    "Harman8",
    "Harman4",
    "Harman10",
    "Harman12",
    "Studio",
    "Nearfield",
    "HiFi",
    "Speech",
    "Toole",
    "BK_Light",
    "BK_Medium",
    "BK_Strong",
    "Flat",
    "Cinema",
)


def _auto_hash_array(a: np.ndarray, *, decimals: int = 4, max_len: int = 1200) -> str:
    """
    Stable-ish hash for numeric arrays:
      - flatten
      - drop non-finite
      - downsample to max_len points
      - round
      - sha256
    """
    try:
        x = np.asarray(a, dtype=float).reshape(-1)
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    m = np.isfinite(x)
    x = x[m]
    if x.size <= 0:
        return ""
    if x.size > int(max_len):
        idx = np.linspace(0, x.size - 1, int(max_len)).astype(int)
        x = x[idx]
    x = np.round(x, int(decimals))
    b = x.astype(np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()


def _auto_measurement_signature(measurements: dict) -> str:
    fL = measurements.get("f_l")
    mL = measurements.get("m_l")
    fR = measurements.get("f_r")
    mR = measurements.get("m_r")
    h = hashlib.sha256()
    h.update(_auto_hash_array(np.asarray(fL) if fL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(mL) if mL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(fR) if fR is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array(np.asarray(mR) if mR is not None else np.asarray([])).encode("ascii", "ignore"))
    return h.hexdigest()


def _auto_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> str:
    """
    Signature for caching:
      - measurement response (f/m arrays L+R)
      - key settings that affect search space and result
    """
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    h = hashlib.sha256()
    h.update(_auto_measurement_signature(measurements).encode("ascii", "ignore"))
    keys = {
        "fs": int(fs_v),
        "taps": int(taps_v),
        "filter_type": ft,
        "enable_tdc": bool(base_data.get("enable_tdc", True)),
        "enable_afdw": bool(base_data.get("enable_afdw", True)),
        "bass_first_ai": bool(base_data.get("bass_first_ai", True)),
        "mag_c_max": float(_auto_safe_float(base_data.get("mag_c_max", 250.0), 250.0)),
        "_auto_mag_c_min_hz": float(_auto_safe_float(base_data.get("_auto_mag_c_min_hz", float("nan")), float("nan"))),
        "_auto_low_bass_cut_hz": float(_auto_safe_float(base_data.get("_auto_low_bass_cut_hz", float("nan")), float("nan"))),
        "_auto_exc_freq_hz": float(_auto_safe_float(base_data.get("_auto_exc_freq_hz", float("nan")), float("nan"))),
        "xos": xos if isinstance(xos, list) else [],
        "hpf": hpf if isinstance(hpf, dict) or hpf is None else str(hpf),
    }
    if bool(include_hc_mode):
        keys["hc_mode"] = str(hc_mode or base_data.get("hc_mode", "") or "").strip()
    try:
        h.update(json.dumps(keys, sort_keys=True, default=str).encode("utf-8"))
    except Exception:
        h.update(str(keys).encode("utf-8", "ignore"))
    return h.hexdigest()


def _auto_cache_load() -> dict:
    path = _auto_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _auto_cache_save(cache: dict) -> None:
    path = _auto_cache_path()
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        # best-effort only
        return


def _auto_cache_get_entry(sig: str) -> dict | None:
    if not sig:
        return None
    cache = _auto_cache_load()
    items = cache.get("items", {})
    if not isinstance(items, dict):
        return None
    entry = items.get(sig)
    return dict(entry) if isinstance(entry, dict) else None


def _auto_cache_get_best(sig: str) -> dict | None:
    entry = _auto_cache_get_entry(sig)
    if not isinstance(entry, dict):
        return None
    preset = entry.get("best_preset")
    return dict(preset) if isinstance(preset, dict) else None


def _auto_cache_get_best_target(sig: str) -> str | None:
    entry = _auto_cache_get_entry(sig)
    if not isinstance(entry, dict):
        return None
    hc = str(entry.get("best_target_curve", entry.get("best_hc_mode", "")) or "").strip()
    return hc or None


def _auto_cache_get_target_for_measurements(measurements: dict) -> dict | None:
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return None
    cache = _auto_cache_load()

    # Preferred map: target selected by measurement signature.
    target_map = cache.get("target_by_measurement", {})
    if isinstance(target_map, dict):
        direct = target_map.get(msig)
        if isinstance(direct, dict):
            return dict(direct)

    # Backward compatibility: check legacy item entries if they carry measurement_sig.
    items = cache.get("items", {})
    if not isinstance(items, dict):
        return None
    best = None
    best_t = -1
    for entry in items.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("measurement_sig", "") or "") != str(msig):
            continue
        try:
            t = int(entry.get("t", 0) or 0)
        except Exception:
            t = 0
        if t >= best_t:
            best_t = int(t)
            best = dict(entry)
    return dict(best) if isinstance(best, dict) else None


def _auto_cache_put_target_for_measurements(
    *,
    measurements: dict,
    best_hc_mode: str | None,
    best_preset: dict,
    best_metrics: dict | None = None,
) -> None:
    hc_val = str(best_hc_mode or "").strip()
    if not hc_val:
        return
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return
    cache = _auto_cache_load()
    target_map = cache.get("target_by_measurement", {})
    if not isinstance(target_map, dict):
        target_map = {}
    target_map[str(msig)] = {
        "t": int(time.time()),
        "measurement_sig": str(msig),
        "best_target_curve": hc_val,
        "best_hc_mode": hc_val,
        "best_preset": dict(best_preset or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    try:
        if len(target_map) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                target_map.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            target_map = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        pass
    cache["target_by_measurement"] = target_map
    cache["v"] = 2
    _auto_cache_save(cache)


def _auto_cache_put_best(
    sig: str,
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
) -> None:
    if not sig or not isinstance(best_preset, dict):
        return
    cache = _auto_cache_load()
    items = cache.get("items", {})
    if not isinstance(items, dict):
        items = {}
    entry = {
        "t": int(time.time()),
        "best_preset": dict(best_preset),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
    items[str(sig)] = entry
    try:
        if len(items) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                items.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            items = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        pass
    cache["items"] = items
    cache["v"] = 2
    _auto_cache_save(cache)


def _auto_safe_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _auto_safe_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        s = str(value or "").strip().lower()
    except Exception:
        return bool(default)
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _auto_target_one_step_milder(hc_name: str) -> str | None:
    name = str(hc_name or "").strip()
    if not name:
        return None
    ladders = (
        ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
        ("BK_Light", "BK_Medium", "BK_Strong"),
    )
    for ladder in ladders:
        if name not in ladder:
            continue
        idx = int(ladder.index(name))
        if idx <= 0:
            return None
        return str(ladder[idx - 1])
    return None


def _auto_collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []


def _auto_event_severity(refs: list | None) -> float:
    refs = refs or []
    if not isinstance(refs, list) or not refs:
        return 0.0

    vals = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        v = _auto_safe_float(r.get("gd_error", 0.0), 0.0)
        if np.isfinite(v):
            vals.append(abs(float(v)))
    if not vals:
        return 0.0

    vals = sorted(vals, reverse=True)[:5]
    weights = (1.00, 0.75, 0.55, 0.40, 0.30)
    sev = 0.0
    for i, v in enumerate(vals):
        # Ignore very small GD irregularities; focus on meaningful events.
        sev += float(weights[i]) * max(0.0, float(v) - 2.0)
    return float(max(0.0, sev))


def _auto_pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _auto_select_builtin_target_curve(
    data: dict,
    *,
    f_l,
    m_l,
    f_r,
    m_r,
) -> dict | None:
    try:
        fl = np.asarray(f_l, dtype=float).reshape(-1)
        ml = np.asarray(m_l, dtype=float).reshape(-1)
        fr = np.asarray(f_r, dtype=float).reshape(-1)
        mr = np.asarray(m_r, dtype=float).reshape(-1)
    except Exception:
        return None

    if fl.size < 32 or fr.size < 32 or ml.size != fl.size or mr.size != fr.size:
        return None

    def _sorted_xy(f, y):
        idx = np.argsort(f)
        ff = np.asarray(f[idx], dtype=float)
        yy = np.asarray(y[idx], dtype=float)
        m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
        return ff[m], yy[m]

    fl, ml = _sorted_xy(fl, ml)
    fr, mr = _sorted_xy(fr, mr)
    if fl.size < 32 or fr.size < 32:
        return None

    try:
        lvl_min = float(data.get("lvl_min", 500.0) or 500.0)
        lvl_max = float(data.get("lvl_max", 2000.0) or 2000.0)
    except Exception:
        lvl_min, lvl_max = 500.0, 2000.0
    if not np.isfinite(lvl_min) or not np.isfinite(lvl_max) or lvl_min <= 0.0 or lvl_max <= lvl_min:
        lvl_min, lvl_max = 500.0, 2000.0

    try:
        mag_lo = float(data.get("mag_c_min", 20.0) or 20.0)
        mag_hi = float(data.get("mag_c_max", 250.0) or 250.0)
    except Exception:
        mag_lo, mag_hi = 20.0, 250.0
    if not np.isfinite(mag_lo) or not np.isfinite(mag_hi) or mag_lo <= 0.0 or mag_hi <= mag_lo:
        mag_lo, mag_hi = 20.0, 250.0

    scored = []
    for hc_name in AUTO_MODE_BUILTIN_TARGETS:
        try:
            hf, hm = get_house_curve_by_name(hc_name)
            hf = np.asarray(hf, dtype=float).reshape(-1)
            hm = np.asarray(hm, dtype=float).reshape(-1)
            if hf.size < 4 or hm.size != hf.size:
                continue
            hs = np.argsort(hf)
            hf = hf[hs]
            hm = hm[hs]
            m_h = np.isfinite(hf) & np.isfinite(hm) & (hf > 0.0)
            hf = hf[m_h]
            hm = hm[m_h]
            if hf.size < 4:
                continue

            f_lo = max(20.0, float(np.min(fl)), float(np.min(fr)), float(np.min(hf)))
            f_hi = min(20000.0, float(np.max(fl)), float(np.max(fr)), float(np.max(hf)))
            if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= (f_lo * 1.15):
                continue

            fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
            ml_g = np.interp(fg, fl, ml)
            mr_g = np.interp(fg, fr, mr)
            m_avg = 0.5 * (ml_g + mr_g)
            try:
                m_avg_sm, _ = dsp.apply_smoothing_std(
                    fg,
                    m_avg,
                    np.zeros_like(m_avg),
                    float(AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT),
                )
                m_avg = np.asarray(m_avg_sm, dtype=float)
            except Exception:
                pass
            t_g = np.interp(fg, hf, hm)

            lvl_mask = (fg >= lvl_min) & (fg <= lvl_max)
            if int(np.count_nonzero(lvl_mask)) < 16:
                lvl_mask = (fg >= 300.0) & (fg <= 3000.0)
            if int(np.count_nonzero(lvl_mask)) < 16:
                lvl_mask = np.ones_like(fg, dtype=bool)

            off = float(np.median(m_avg[lvl_mask] - t_g[lvl_mask]))
            err = m_avg - (t_g + off)

            corr_mask = (fg >= mag_lo) & (fg <= mag_hi)
            if int(np.count_nonzero(corr_mask)) < 16:
                corr_mask = np.ones_like(fg, dtype=bool)

            rms = float(np.sqrt(np.mean(np.square(err[corr_mask]))))
            scored.append({"hc_mode": str(hc_name), "fit_rms_db": float(rms), "offset_db": float(off)})
        except Exception:
            continue

    if not scored:
        return None

    scored = sorted(scored, key=lambda d: float(d.get("fit_rms_db", 1e9)))
    best = scored[0]
    return {
        "selected_hc_mode": str(best.get("hc_mode", "Harman6")),
        "fit_rms_db": float(best.get("fit_rms_db", 0.0)),
        "offset_db": float(best.get("offset_db", 0.0)),
        "candidates": list(scored[:5]),
    }


def _auto_select_target_curve_with_trials(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
) -> dict | None:
    # Strong cache key: measurement response only.
    cached_target_entry = None
    try:
        cached_target_entry = _auto_cache_get_target_for_measurements(measurements)
    except Exception:
        cached_target_entry = None
    if isinstance(cached_target_entry, dict):
        cached_hc = str(
            cached_target_entry.get(
                "best_target_curve",
                cached_target_entry.get("best_hc_mode", ""),
            )
            or ""
        ).strip()
        if cached_hc:
            try:
                c_f, c_m = get_house_curve_by_name(cached_hc)
                c_f = np.asarray(c_f, dtype=float).reshape(-1)
                c_m = np.asarray(c_m, dtype=float).reshape(-1)
            except Exception:
                c_f = np.asarray([], dtype=float)
                c_m = np.asarray([], dtype=float)
            if c_f.size >= 4 and c_m.size == c_f.size:
                cached_preset = dict(cached_target_entry.get("best_preset", {}) or {})
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: using cached target curve "
                        f"{cached_hc}"
                    )
                return {
                    "selected_hc_mode": str(cached_hc),
                    "fit_rms_db": float("nan"),
                    "offset_db": 0.0,
                    "selection_method": "cache_measurement",
                    "top_n": 0,
                    "trials_per_curve": 0,
                    "candidates": [],
                    "evaluated": [],
                    "best_preset": dict(cached_preset),
                }

    # Try cache first: same measurements + key settings -> reuse previously chosen target curve.
    if bool(AUTO_MODE_CACHE_ENABLED):
        try:
            sig_target = _auto_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            cached_hc = _auto_cache_get_best_target(sig_target)
            if cached_hc:
                try:
                    c_f, c_m = get_house_curve_by_name(cached_hc)
                    c_f = np.asarray(c_f, dtype=float).reshape(-1)
                    c_m = np.asarray(c_m, dtype=float).reshape(-1)
                except Exception:
                    c_f = np.asarray([], dtype=float)
                    c_m = np.asarray([], dtype=float)
                if c_f.size >= 4 and c_m.size == c_f.size:
                    cached_preset = _auto_cache_get_best(sig_target) or {}
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: using cached target curve "
                            f"{cached_hc}"
                        )
                    return {
                        "selected_hc_mode": str(cached_hc),
                        "fit_rms_db": float("nan"),
                        "offset_db": 0.0,
                        "selection_method": "cache",
                        "top_n": 0,
                        "trials_per_curve": 0,
                        "candidates": [],
                        "evaluated": [],
                        "best_preset": dict(cached_preset),
                    }
        except Exception:
            pass

    f6_hz = _auto_safe_float(
        base_data.get("_auto_mag_c_min_hz", base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f" (-6 dB {f6_hz:.1f} Hz)" if np.isfinite(f6_hz) else ""

    quick = _auto_select_builtin_target_curve(
        base_data,
        f_l=measurements.get("f_l"),
        m_l=measurements.get("m_l"),
        f_r=measurements.get("f_r"),
        m_r=measurements.get("m_r"),
    )
    if not isinstance(quick, dict):
        return None

    quick_candidates = list(quick.get("candidates", []) or [])
    if not quick_candidates:
        return None

    top_n_eff = max(1, int(top_n))
    trials_eff = max(1, int(trials_per_curve))
    shortlisted = quick_candidates[:top_n_eff]
    prefer_milder = _auto_safe_bool(
        base_data.get("auto_target_prefer_milder_step", AUTO_MODE_TARGET_PREFER_MILDER_STEP),
        AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    )
    if prefer_milder and shortlisted:
        lead_hc = str(shortlisted[0].get("hc_mode", "") or "").strip()
        lead_milder = _auto_target_one_step_milder(lead_hc)
        if lead_milder:
            milder_tc = None
            for tc in quick_candidates:
                if str(tc.get("hc_mode", "") or "").strip() == str(lead_milder):
                    milder_tc = dict(tc)
                    break
            if isinstance(milder_tc, dict):
                already = {
                    str(tc.get("hc_mode", "") or "").strip()
                    for tc in shortlisted
                    if isinstance(tc, dict)
                }
                if str(lead_milder) not in already:
                    shortlisted = list(shortlisted) + [milder_tc]
    if not shortlisted:
        return None

    evaluated = []
    for t_idx, tc in enumerate(shortlisted, start=1):
        hc_name = str(tc.get("hc_mode", "") or "").strip()
        if not hc_name:
            continue
        try:
            hc_f, hc_m = get_house_curve_by_name(hc_name)
            hc_f = np.asarray(hc_f, dtype=float)
            hc_m = np.asarray(hc_m, dtype=float)
        except Exception:
            continue
        if hc_f.size < 4 or hc_m.size != hc_f.size:
            continue

        seed_tc = int(20260302 + int(fs_v) * 31 + int(taps_v) * 7 + sum(ord(ch) for ch in hc_name) * 13)
        base_tc = dict(base_data or {})
        base_tc["hc_mode"] = str(hc_name)
        candidates = _build_auto_mode_candidates(base_tc, n_trials=trials_eff, seed=seed_tc)

        best_metrics = None
        best_preset = None
        ok_n = 0
        rank_sum = 0.0

        for c_idx, preset in enumerate(candidates, start=1):
            trial_data = dict(base_tc)
            trial_data.update(dict(preset or {}))
            trial_data["comparison_mode"] = True
            trial_measurements = dict(measurements or {})
            trial_measurements["ui_data"] = trial_data

            try:
                cfg = build_config(
                    trial_data,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    hc_f=hc_f,
                    hc_m=hc_m,
                    pin=pin_obj,
                    max_safe_boost=float(MAX_SAFE_BOOST),
                )
                try:
                    setattr(cfg, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
                    setattr(cfg, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
                except Exception:
                    pass

                res = run_pipeline(cfg, trial_measurements)
                res.metrics["summary"] = summarize_run(res)
                met = _auto_score_result(
                    res,
                    auto_exc_freq_hz=_auto_safe_float(trial_data.get("_auto_exc_freq_hz", float("nan")), float("nan")),
                )
                ok_n += 1
                rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                    best_metrics = dict(met)
                    best_preset = dict(preset or {})
            except Exception as exc:
                logger.warning(
                    f"Automatic mode target trial failed: target={hc_name} "
                    f"{c_idx}/{len(candidates)} ({type(exc).__name__}: {exc})"
                )

            if callable(status_cb):
                best_txt = "n/a"
                if isinstance(best_metrics, dict):
                    best_txt = f"{_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}"
                status_cb(
                    f"CamillaFIR automatic mode: target test {t_idx}/{len(shortlisted)} "
                    f"{hc_name} trial {c_idx}/{len(candidates)}{f6_txt} (best {best_txt}/100)"
                )

        if ok_n <= 0 or not isinstance(best_metrics, dict):
            continue
        evaluated.append(
            {
                "hc_mode": str(hc_name),
                "fit_rms_db": _auto_safe_float(tc.get("fit_rms_db"), 0.0),
                "offset_db": _auto_safe_float(tc.get("offset_db"), 0.0),
                "trials_total": int(len(candidates)),
                "trials_ok": int(ok_n),
                "avg_rank_score": float(rank_sum / max(1, ok_n)),
                "best_metrics": dict(best_metrics),
                "best_preset": dict(best_preset or {}),
            }
        )

    if not evaluated:
        return quick

    def _tc_key(item: dict) -> tuple:
        bm = dict(item.get("best_metrics", {}) or {})
        return (
            -_auto_safe_float(bm.get("rank_score"), 0.0),
            -_auto_safe_float(item.get("avg_rank_score"), 0.0),
            _auto_safe_float(item.get("fit_rms_db"), 1e9),
        )

    evaluated = sorted(evaluated, key=_tc_key)
    winner = evaluated[0]
    selection_method = "top3x10_trials"
    if prefer_milder:
        winner_hc = str(winner.get("hc_mode", "") or "").strip()
        milder_hc = _auto_target_one_step_milder(winner_hc)
        if milder_hc:
            milder_item = None
            for it in evaluated:
                if str(it.get("hc_mode", "") or "").strip() == str(milder_hc):
                    milder_item = dict(it)
                    break
            if isinstance(milder_item, dict):
                w_rank = _auto_safe_float(dict(winner.get("best_metrics", {}) or {}).get("rank_score"), 0.0)
                m_rank = _auto_safe_float(dict(milder_item.get("best_metrics", {}) or {}).get("rank_score"), 0.0)
                w_fit = _auto_safe_float(winner.get("fit_rms_db"), float("nan"))
                m_fit = _auto_safe_float(milder_item.get("fit_rms_db"), float("nan"))
                rank_drop = float(w_rank - m_rank)
                fit_add = float(m_fit - w_fit) if np.isfinite(w_fit) and np.isfinite(m_fit) else 0.0
                if (
                    rank_drop <= float(AUTO_MODE_TARGET_MILDER_MAX_RANK_DROP)
                    and fit_add <= float(AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB)
                ):
                    winner = dict(milder_item)
                    selection_method = "top3x10_trials_milder_step"
                    logger.info(
                        "Automatic mode target select: prefer milder step "
                        f"{winner_hc} -> {milder_hc} "
                        f"(rank_drop={rank_drop:.3f}, fit_add={fit_add:.3f} dB)"
                    )
    return {
        "selected_hc_mode": str(winner.get("hc_mode", quick.get("selected_hc_mode", "Harman6"))),
        "fit_rms_db": float(winner.get("fit_rms_db", quick.get("fit_rms_db", 0.0))),
        "offset_db": float(winner.get("offset_db", quick.get("offset_db", 0.0))),
        "selection_method": str(selection_method),
        "top_n": int(len(shortlisted)),
        "trials_per_curve": int(trials_eff),
        "candidates": list(shortlisted),
        "evaluated": list(evaluated),
        "best_preset": dict(winner.get("best_preset", {}) or {}),
    }


def _auto_dsp_quality_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    real_rms = _auto_pick_metric(
        st,
        (
            "real_mag_error_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)
    dbg["real_rms"] = real_rms

    ripple_rms = _auto_pick_metric(
        st,
        (
            "ripple_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)
    dbg["ripple_rms"] = ripple_rms

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        penalty += 0.60 * max(0.0, float(gd_grad_max) - 12.0)
    dbg["gd_grad_max"] = gd_grad_max

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ),
    )
    if pre_ringing_db is not None:
        penalty += 0.70 * max(0.0, float(pre_ringing_db) + 40.0)
    dbg["pre_ringing_db"] = pre_ringing_db

    pre_post_ratio = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ),
        nonneg=True,
    )
    if pre_post_ratio is not None:
        penalty += 30.0 * max(0.0, float(pre_post_ratio) - 0.015)
    dbg["ir_pre_post_ratio"] = pre_post_ratio

    phase_boundary_mdb = _auto_pick_metric(
        st,
        (
            "phase_boundary_peak_mdb",
            "phase_corr_boundary_peak_mdb",
        ),
        abs_value=True,
        nonneg=True,
    )
    if phase_boundary_mdb is not None:
        penalty += 0.015 * max(0.0, float(phase_boundary_mdb) - 120.0)
    dbg["phase_boundary_peak_mdb"] = phase_boundary_mdb

    return float(max(0.0, penalty)), dbg


def _auto_excursion_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    exc_raw = st.get("exc_prot", None)
    exc_known = (exc_raw is not None)
    exc_on = bool(exc_raw) if exc_known else None
    exc_freq = _auto_safe_float(st.get("exc_freq", 0.0), 0.0)

    try:
        exc_bins = int(float(st.get("boost_candidate_bins_excprot", 0) or 0))
    except Exception:
        exc_bins = 0
    lf_boost_max = _auto_safe_float(st.get("lf_boost_max_db", 0.0), 0.0)

    # Prefer presets that keep excursion protection enabled.
    if exc_known and (exc_on is False):
        penalty += 2.0

    # If protection is enabled but configured too low/invalid, add a small penalty.
    if exc_known and (exc_on is True) and (not np.isfinite(exc_freq) or exc_freq <= 0.0):
        penalty += 0.8

    # Penalize tendency to boost in excursion-protected region.
    if exc_bins > 0:
        penalty += min(2.5, 0.10 * float(exc_bins))

    # Penalize remaining LF boost inside guard region.
    penalty += min(12.0, 1.25 * max(0.0, float(lf_boost_max) - 1.5))

    # Prevent a single excursion metric from collapsing rank to zero.
    penalty = min(16.0, float(penalty))

    dbg["exc_known"] = bool(exc_known)
    dbg["exc_on"] = exc_on
    dbg["exc_freq"] = float(exc_freq)
    dbg["exc_bins"] = int(exc_bins)
    dbg["lf_boost_max_db"] = float(lf_boost_max)
    return float(max(0.0, penalty)), dbg


def _auto_score_result(result, *, auto_exc_freq_hz: float | None = None) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = plots.calc_ai_summary_from_stats(l_st)
    r_ai = plots.calc_ai_summary_from_stats(r_st)

    def _ai_score_with_fallback(st: dict, ai: dict) -> float:
        score = _auto_safe_float((ai or {}).get("score"), float("nan"))
        if np.isfinite(score):
            return float(score)
        try:
            conf = _auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", 0.0)),
                0.0,
            )
            rms_fb, match_fb = target_match_from_stats(
                st,
                include_filter=False,
                use_confidence=True,
                use_smart_scan_range=True,
            )
            if match_fb is None:
                return 0.0
            rt60 = st.get("rt60_val", None)
            rt_rel = st.get("rt60_reliability", None)
            return _auto_safe_float(
                plots.calc_acoustic_score(conf, float(match_fb), rt60_s=rt60, rt60_rel=rt_rel),
                0.0,
            )
        except Exception:
            return 0.0

    l_score = _ai_score_with_fallback(l_st, l_ai)
    r_score = _ai_score_with_fallback(r_st, r_ai)
    avg_score = (l_score + r_score) / 2.0
    lr_delta = abs(l_score - r_score)

    net_boost_max = max(
        _auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        _auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    l_refs = _auto_collect_reflections(l_st)
    r_refs = _auto_collect_reflections(r_st)
    events_total = int(len(l_refs) + len(r_refs))
    events_severity_l = _auto_event_severity(l_refs)
    events_severity_r = _auto_event_severity(r_refs)
    events_severity_raw = float(events_severity_l + events_severity_r)
    events_severity = float(math.log1p(max(0.0, events_severity_raw) / 6.0))
    dsp_pen_l, dsp_dbg_l = _auto_dsp_quality_penalty(l_st)
    dsp_pen_r, dsp_dbg_r = _auto_dsp_quality_penalty(r_st)
    dsp_penalty_raw = 0.5 * (float(dsp_pen_l) + float(dsp_pen_r))
    exc_pen_l, exc_dbg_l = _auto_excursion_penalty(l_st)
    exc_pen_r, exc_dbg_r = _auto_excursion_penalty(r_st)
    exc_penalty_raw = 0.5 * (float(exc_pen_l) + float(exc_pen_r))
    exc_penalty_waived = bool(np.isfinite(_auto_safe_float(auto_exc_freq_hz, float("nan"))))
    # Auto-excursion frequency is a good sign, but don't fully "waive" excursion risk.
    # Scale down instead of zeroing so auto-mode won't ignore clear LF-boost/excursion problems.
    exc_penalty = float(exc_penalty_raw) * (0.35 if exc_penalty_waived else 1.0)

    # Penalty budget normalization (avoid single-term dominance)
    # Smooth the 5 dB boost "knee" to avoid hard-threshold behavior in optimization.
    _BOOST_KNEE_DB = 1.0  # larger = softer knee (e.g. 0.7..1.5)
    _x = (float(net_boost_max) - 5.0) / float(_BOOST_KNEE_DB)
    _x = float(np.clip(_x, -60.0, 60.0))  # guard exp overflow
    _soft_hinge_db = float(_BOOST_KNEE_DB) * float(np.log1p(np.exp(_x)))
    boost_pen = min(12.0, 1.25 * _soft_hinge_db)
    dsp_penalty = min(12.0, 0.07 * float(dsp_penalty_raw))
    event_pen = min(4.0, 0.25 * float(events_severity))
    lr_pen = min(4.0, 0.03 * lr_delta)
    exc_penalty = min(12.0, float(exc_penalty))
    rank_raw = float(avg_score - boost_pen - event_pen - lr_pen - dsp_penalty - exc_penalty)
    rank_score = float(np.clip(rank_raw, 0.0, 100.0))

    return {
        "rank_score": float(rank_score),
        "avg_score": float(avg_score),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "events_total": int(events_total),
        "events_severity": float(events_severity),
        "events_severity_raw": float(events_severity_raw),
        "events_severity_l": float(events_severity_l),
        "events_severity_r": float(events_severity_r),
        "event_penalty": float(event_pen),
        "dsp_penalty": float(dsp_penalty),
        "dsp_penalty_raw": float(dsp_penalty_raw),
        "dsp_penalty_l": float(dsp_pen_l),
        "dsp_penalty_r": float(dsp_pen_r),
        "exc_penalty": float(exc_penalty),
        "exc_penalty_raw": float(exc_penalty_raw),
        "exc_penalty_waived": bool(exc_penalty_waived),
        "exc_penalty_l": float(exc_pen_l),
        "exc_penalty_r": float(exc_pen_r),
        "dsp_dbg_l": dict(dsp_dbg_l),
        "dsp_dbg_r": dict(dsp_dbg_r),
        "exc_dbg_l": dict(exc_dbg_l),
        "exc_dbg_r": dict(exc_dbg_r),
    }


def _auto_rank_key(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("rank_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


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
                float(AUTO_MODE_MAG_C_MIN_SMOOTH_OCT),
            )
            mm_use = np.asarray(mm_sm, dtype=float)
        except Exception:
            mm_use = np.asarray(mm, dtype=float)

        ref_mask = (ff >= float(AUTO_MODE_MAG_C_MIN_REF_MIN_HZ)) & (ff <= float(AUTO_MODE_MAG_C_MIN_REF_MAX_HZ))
        if int(np.count_nonzero(ref_mask)) < 8:
            ref_mask = (ff >= 63.0) & (ff <= 250.0)
        if int(np.count_nonzero(ref_mask)) < 8:
            return None

        # Robust reference level: prefer upper-quantile to avoid SBIR/nulls pulling ref too low.
        ref_slice = np.asarray(mm_use[ref_mask], dtype=float)
        ref_slice = ref_slice[np.isfinite(ref_slice)]
        if ref_slice.size < 6:
            return None
        ref_db = float(np.quantile(ref_slice, 0.75))
        thr_db = float(ref_db - 6.0)

        lf_hi = float(min(float(AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ), float(AUTO_MODE_MAG_C_MIN_REF_MIN_HZ)))
        lf_mask = (ff >= float(AUTO_MODE_MAG_C_MIN_MIN_HZ)) & (ff <= lf_hi)
        if int(np.count_nonzero(lf_mask)) < 8:
            return None
        f_lo = ff[lf_mask]
        m_lo = mm_use[lf_mask]
        # Enforce monotonic LF envelope (rising with frequency) to reject local dips.
        m_env = np.maximum.accumulate(np.asarray(m_lo, dtype=float))
        above = np.asarray(m_env >= thr_db, dtype=bool)
        if not np.any(above):
            # If we never cross -6 dB, prefer a conservative fallback instead of pinning to lf_hi.
            return float(_auto_safe_float(default_hz, 25.0))

        # Require the envelope to stay above threshold for N consecutive points.
        # This reduces false early crossings caused by sparse LF bins or residual noise.
        try:
            df = float(np.median(np.diff(f_lo))) if f_lo.size > 2 else float("nan")
        except Exception:
            df = float("nan")
        # Target ~3 Hz minimum "stay above" span, but at least 3 points.
        if np.isfinite(df) and df > 0.0:
            N = int(max(3, round(3.0 / df)))
        else:
            N = 3
        N = int(np.clip(N, 3, 12))

        i1 = None
        if above.size >= N:
            # Find first index where above[i:i+N] are all True.
            for i in range(0, int(above.size) - N + 1):
                if bool(np.all(above[i : i + N])):
                    i1 = int(i)
                    break
        if i1 is None:
            # No stable crossing found -> conservative fallback.
            return float(_auto_safe_float(default_hz, 25.0))
        if i1 <= 0:
            return float(AUTO_MODE_MAG_C_MIN_MIN_HZ)

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
        est = _auto_safe_float(default_hz, 25.0)
    elif f6_l is None:
        est = float(f6_r)
    elif f6_r is None:
        est = float(f6_l)
    else:
        # If channels agree reasonably well, average for stability; otherwise stay conservative.
        if abs(float(f6_l) - float(f6_r)) <= 8.0:
            est = 0.5 * (float(f6_l) + float(f6_r))
        else:
            est = max(float(f6_l), float(f6_r))

    est = float(np.clip(est, float(AUTO_MODE_MAG_C_MIN_MIN_HZ), float(AUTO_MODE_MAG_C_MIN_MAX_HZ)))
    return float(round(est, 1))


def _build_auto_mode_candidates(base_data: dict, *, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_eff = max(1, int(n_trials))

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_linear = ("linear" in ft) and (not is_mixed)
    mixed_center = _auto_safe_float(base_data.get("mixed_freq", 180.0), 180.0)
    if not np.isfinite(mixed_center) or mixed_center <= 0.0:
        mixed_center = 180.0
    phase_center = _auto_safe_float(base_data.get("phase_limit", 400.0), 400.0)
    if not np.isfinite(phase_center) or phase_center <= 0.0:
        phase_center = 400.0
    mag_c_min_fixed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low_bass_cut_fixed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )

    out: list[dict] = [{}]
    for _ in range(max(0, n_eff - 1)):
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(rng.uniform(8.0, 16.0)), 2),
            "tdc_strength": round(float(rng.uniform(15.0, 75.0)), 1),
            "tdc_max_reduction_db": round(float(rng.uniform(6.0, 36.0)), 1),
            "tdc_slope_db_per_oct": float(rng.choice(np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]))),
            "reg_strength": round(float(rng.uniform(15.0, 45.0)), 1),
            "max_slope_db_per_oct": float(rng.choice(np.array([8.0, 10.0, 12.0, 14.0, 16.0]))),
            "max_boost": round(float(rng.uniform(3.0, 8.0)), 2),
            "mag_c_min": round(float(mag_c_min_fixed), 1),
            "mag_c_max": round(float(rng.uniform(170.0, 300.0)), 1),
            "trans_width": round(float(rng.uniform(70.0, 150.0)), 1),
            "filter_smooth": int(rng.choice(np.array([96]))),
            "bass_first_mode_max_hz": round(float(rng.uniform(150.0, 220.0)), 1),
            "low_bass_cut_hz": round(float(low_bass_cut_fixed), 1),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(loc=mixed_center, scale=35.0), 80.0, 320.0)), 1)
        if is_linear:
            cand["phase_limit"] = round(float(np.clip(rng.normal(loc=phase_center, scale=140.0), 150.0, 1400.0)), 1)
        out.append(cand)
    return out


def _build_auto_mode_refine_candidates(
    base_data: dict,
    *,
    anchors: list[dict],
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_eff = max(0, int(n_trials))
    if n_eff <= 0:
        return []

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_linear = ("linear" in ft) and (not is_mixed)

    anchor_items = list(anchors or [])
    if not anchor_items:
        anchor_items = [{"preset": {}}]

    def _anchor_val(anchor: dict, key: str, default: float) -> float:
        p = dict(anchor.get("preset", {}) or {})
        if key in p:
            return _auto_safe_float(p.get(key), default)
        return _auto_safe_float(base_data.get(key), default)

    def _near_discrete(center: float, choices: list[float], sigma: float) -> float:
        if not choices:
            return float(center)
        x = float(rng.normal(loc=float(center), scale=float(max(0.01, sigma))))
        return float(min(choices, key=lambda c: abs(float(c) - x)))

    out: list[dict] = []
    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0]
    max_slope_choices = [8.0, 10.0, 12.0, 14.0, 16.0]
    smooth_choices = [96]
    mag_c_min_fixed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )
    low_bass_cut_fixed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )

    for _ in range(n_eff):
        a = anchor_items[int(rng.integers(0, len(anchor_items)))]
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(float(np.clip(rng.normal(_anchor_val(a, "fdw_cycles", 10.0), 1.2), 8.0, 16.0)), 2),
            "tdc_strength": round(float(np.clip(rng.normal(_anchor_val(a, "tdc_strength", 50.0), 5.0), 35.0, 75.0)), 1),
            "tdc_max_reduction_db": round(float(np.clip(rng.normal(_anchor_val(a, "tdc_max_reduction_db", 9.0), 1.0), 6.0, 12.0)), 1),
            "tdc_slope_db_per_oct": _near_discrete(_anchor_val(a, "tdc_slope_db_per_oct", 6.0), slope_choices, 0.8),
            "reg_strength": round(float(np.clip(rng.normal(_anchor_val(a, "reg_strength", 30.0), 4.0), 15.0, 45.0)), 1),
            "max_slope_db_per_oct": _near_discrete(_anchor_val(a, "max_slope_db_per_oct", 12.0), max_slope_choices, 1.5),
            "max_boost": round(float(np.clip(rng.normal(_anchor_val(a, "max_boost", 4.0), 0.45), 3.0, 8.0)), 2),
            "mag_c_min": round(float(mag_c_min_fixed), 1),
            "mag_c_max": round(float(np.clip(rng.normal(_anchor_val(a, "mag_c_max", 220.0), 15.0), 170.0, 300.0)), 1),
            "trans_width": round(float(np.clip(rng.normal(_anchor_val(a, "trans_width", 100.0), 10.0), 70.0, 150.0)), 1),
            "filter_smooth": int(_near_discrete(_anchor_val(a, "filter_smooth", 96.0), [float(x) for x in smooth_choices], 96.0)),
            "bass_first_mode_max_hz": round(float(np.clip(rng.normal(_anchor_val(a, "bass_first_mode_max_hz", 180.0), 10.0), 150.0, 220.0)), 1),
            "low_bass_cut_hz": round(float(low_bass_cut_fixed), 1),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(_anchor_val(a, "mixed_freq", 180.0), 12.0), 80.0, 320.0)), 1)
        if is_linear:
            cand["phase_limit"] = round(float(np.clip(rng.normal(_anchor_val(a, "phase_limit", 400.0), 45.0), 150.0, 1400.0)), 1)
        out.append(cand)
    return out


def _run_auto_mode_search(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int = AUTO_MODE_TRIALS,
) -> dict | None:
    search_base_data = dict(base_data or {})
    try:
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
    except Exception:
        seed_preset = {}
    if seed_preset:
        search_base_data.update(seed_preset)

    # --- Auto-mode cache: load best preset for this measurement+settings signature ---
    if bool(AUTO_MODE_CACHE_ENABLED) and not seed_preset:
        try:
            sig = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(search_base_data.get("hc_mode", "") or "").strip() or None,
                include_hc_mode=True,
            )
            cached = _auto_cache_get_best(sig)
            if isinstance(cached, dict) and cached:
                # Use as seed preset (your code already merges _auto_target_seed_preset at top)
                search_base_data["_auto_target_seed_preset"] = dict(cached)
                # Also apply immediately so phase-1 includes this "known good" point
                search_base_data.update(dict(cached))
                logger.info("Automatic mode: loaded cached best preset seed.")
        except Exception:
            pass

    seed = int(20260302 + int(fs_v) * 17 + int(taps_v))
    candidates = _build_auto_mode_candidates(search_base_data, n_trials=int(n_trials), seed=seed)
    try:
        target_label = str(search_base_data.get("hc_mode", "") or "").strip()
    except Exception:
        target_label = ""
    if not target_label:
        target_label = "n/a"
    f6_hz = _auto_safe_float(
        search_base_data.get("_auto_mag_c_min_hz", search_base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    low_bass_hz = _auto_safe_float(
        search_base_data.get("_auto_low_bass_cut_hz", search_base_data.get("low_bass_cut_hz", float("nan"))),
        float("nan"),
    )
    exc_hz = _auto_safe_float(
        search_base_data.get("_auto_exc_freq_hz", search_base_data.get("exc_freq", float("nan"))),
        float("nan"),
    )
    low_txt = f"low-cut {low_bass_hz:.1f} Hz" if np.isfinite(low_bass_hz) else "low-cut n/a"
    exc_txt = f"exc {exc_hz:.1f} Hz" if np.isfinite(exc_hz) else "exc n/a"
    if np.isfinite(f6_hz):
        status_prefix = (
            f"CamillaFIR automatic mode [{target_label}] "
            f"(-6 dB {f6_hz:.1f} Hz, {low_txt}, {exc_txt})"
        )
    else:
        status_prefix = f"CamillaFIR automatic mode [{target_label}] ({low_txt}, {exc_txt})"

    best_result = None
    best_metrics = None
    best_preset = None
    scored = []
    phase1_ok = 0
    phase2_ok = 0
    phase1_tried = 0
    phase2_tried = 0
    phase1_plateau_hit = False
    phase2_plateau_hit = False

    def _eval_candidates(
        cands: list[dict],
        *,
        phase_label: str,
        plateau_after_no_improve: int = 0,
    ) -> dict:
        nonlocal best_result, best_metrics, best_preset, scored
        ok_n = 0
        tried_n = 0
        plateau_hit = False
        no_improve_streak = 0
        for idx, preset in enumerate(cands, start=1):
            tried_n = int(idx)
            improved = False
            trial_data = dict(search_base_data or {})
            trial_data.update(preset or {})
            trial_data["comparison_mode"] = True
            trial_measurements = dict(measurements or {})
            trial_measurements["ui_data"] = trial_data

            try:
                cfg = build_config(
                    trial_data,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    hc_f=hc_f,
                    hc_m=hc_m,
                    pin=pin_obj,
                    max_safe_boost=float(MAX_SAFE_BOOST),
                )
                try:
                    setattr(cfg, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
                    setattr(cfg, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
                except Exception:
                    pass

                result = run_pipeline(cfg, trial_measurements)
                result.metrics["summary"] = summarize_run(result)
                metrics = _auto_score_result(
                    result,
                    auto_exc_freq_hz=_auto_safe_float(trial_data.get("_auto_exc_freq_hz", float("nan")), float("nan")),
                )
                metrics["trial"] = int(len(scored) + 1)
                metrics["phase"] = str(phase_label)
                scored.append({"metrics": metrics, "preset": dict(preset or {})})
                ok_n += 1

                if best_metrics is None or _auto_rank_key(metrics) < _auto_rank_key(best_metrics):
                    best_result = result
                    best_metrics = metrics
                    best_preset = dict(preset or {})
                    improved = True
            except Exception as exc:
                logger.warning(
                    f"Automatic mode trial {idx}/{len(cands)} failed "
                    f"({phase_label}): {type(exc).__name__}: {exc}"
                )

            if callable(status_cb):
                best_txt = "n/a" if not best_metrics else f"{_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}"
                status_cb(
                    f"{status_prefix}: {phase_label} {idx}/{len(cands)} "
                    f"(best {best_txt}/100)"
                )
            if int(plateau_after_no_improve) > 0:
                if improved:
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                if no_improve_streak >= int(plateau_after_no_improve):
                    plateau_hit = True
                    best_now = round(_auto_safe_float((best_metrics or {}).get("rank_score", 0.0), 0.0), 3)
                    move_txt = "plateau -> phase 2" if "1/2" in str(phase_label) else "plateau -> stop"
                    logger.info(
                        f"Automatic mode {phase_label}: no-improve plateau detected "
                        f"({int(plateau_after_no_improve)} rounds), {move_txt}."
                    )
                    if callable(status_cb):
                        status_cb(
                            f"{status_prefix}: {phase_label} {idx}/{len(cands)} "
                            f"(best {best_now:.3f}/100, {move_txt})"
                        )
                    break
        return {"ok": int(ok_n), "tried": int(tried_n), "plateau_hit": bool(plateau_hit)}

    phase1_stats = _eval_candidates(
        candidates,
        phase_label="phase 1/2",
        plateau_after_no_improve=int(AUTO_MODE_PHASE1_PLATEAU_ROUNDS),
    )
    phase1_ok = int(phase1_stats.get("ok", 0) or 0)
    phase1_tried = int(phase1_stats.get("tried", 0) or 0)
    phase1_plateau_hit = bool(phase1_stats.get("plateau_hit", False))

    phase1_top = sorted(scored, key=lambda x: _auto_rank_key(x.get("metrics", {})))[:5]
    refine_candidates = _build_auto_mode_refine_candidates(
        search_base_data,
        anchors=phase1_top,
        n_trials=int(AUTO_MODE_REFINE_TRIALS),
        seed=int(seed + 7919),
    )
    if refine_candidates:
        phase2_stats = _eval_candidates(
            refine_candidates,
            phase_label="phase 2/2",
            plateau_after_no_improve=int(AUTO_MODE_PHASE2_PLATEAU_ROUNDS),
        )
        phase2_ok = int(phase2_stats.get("ok", 0) or 0)
        phase2_tried = int(phase2_stats.get("tried", 0) or 0)
        phase2_plateau_hit = bool(phase2_stats.get("plateau_hit", False))

    if best_result is None or best_metrics is None:
        return None

    top = sorted(scored, key=lambda x: _auto_rank_key(x.get("metrics", {})))[:5]

    # --- Auto-mode cache: save best preset for this signature ---
    if bool(AUTO_MODE_CACHE_ENABLED):
        try:
            best_hc_mode = str(search_base_data.get("hc_mode", "") or "").strip() or None
            measurement_sig = _auto_measurement_signature(measurements)
            sig = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=best_hc_mode,
                include_hc_mode=True,
            )
            sig_target = _auto_signature(
                base_data=search_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            _auto_cache_put_best(
                sig,
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                best_hc_mode=best_hc_mode,
                measurement_sig=measurement_sig,
            )
            _auto_cache_put_best(
                sig_target,
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                best_hc_mode=best_hc_mode,
                measurement_sig=measurement_sig,
            )
            _auto_cache_put_target_for_measurements(
                measurements=measurements,
                best_hc_mode=best_hc_mode,
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
            )
            logger.info("Automatic mode: saved best preset to cache.")
        except Exception:
            pass

    return {
        "best_result": best_result,
        "best_metrics": dict(best_metrics),
        "best_preset": dict(best_preset or {}),
        "top": top,
        "trials_total": int(phase1_tried + phase2_tried),
        "trials_ok": int(len(scored)),
        "trials_phase1_total": int(phase1_tried),
        "trials_phase1_ok": int(phase1_ok),
        "trials_phase2_total": int(phase2_tried),
        "trials_phase2_ok": int(phase2_ok),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }



