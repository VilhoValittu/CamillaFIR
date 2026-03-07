import hashlib
import json
import os
import random
import time

import numpy as np

from ...app_paths import camillafir_data_dir
from .shared import (
    AUTO_MODE_CACHE_ENABLED,
    AUTO_MODE_CACHE_FILTER_KEYS,
    AUTO_MODE_CACHE_FILENAME,
    AUTO_MODE_CACHE_MAX_ITEMS,
    AUTO_MODE_GOAL_DEFAULT,
    _auto_builtin_target_name,
    _auto_filter_cache_key,
    _auto_goal,
    _auto_goal_norm,
    _auto_hash_array,
    _auto_safe_float,
    logger,
)

_AUTO_CACHE_VERSION_MISMATCH_LOGGED = False


def _auto_cache_path() -> str:
    preferred_base = os.fspath(camillafir_data_dir())
    preferred_path = os.path.join(preferred_base, AUTO_MODE_CACHE_FILENAME)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, AUTO_MODE_CACHE_FILENAME)

    try:
        os.makedirs(preferred_base, exist_ok=True)
    except Exception:
        try:
            os.makedirs(legacy_base, exist_ok=True)
        except Exception:
            pass
        return legacy_path

    try:
        if (not os.path.isfile(preferred_path)) and os.path.isfile(legacy_path):
            with open(legacy_path, "rb") as src_f:
                payload = src_f.read()
            with open(preferred_path, "wb") as dst_f:
                dst_f.write(payload)
            logger.info(f"Automatic mode cache migrated to: {preferred_path}")
    except Exception:
        return legacy_path

    return preferred_path


def get_auto_mode_cache_path() -> str:
    return _auto_cache_path()


def _auto_program_version(base_data: dict | None) -> str:
    try:
        return str((base_data or {}).get("program_version", "") or "").strip()
    except Exception:
        return ""


def _auto_cache_bucket_template() -> dict:
    return {
        "items": {},
        "target_by_measurement": {},
        "last_used_best": {},
    }


def _auto_cache_empty(*, program_version: str | None = None) -> dict:
    out = {
        "v": 3,
        "items": {},
        "target_by_measurement": {},
        "by_filter": {},
    }
    ver = str(program_version or "").strip()
    if ver:
        out["program_version"] = str(ver)
    for k in AUTO_MODE_CACHE_FILTER_KEYS:
        out["by_filter"][str(k)] = _auto_cache_bucket_template()
    return out


def _auto_cache_bucket(
    cache: dict,
    *,
    filter_key: str | None,
    create: bool = False,
) -> dict | None:
    if not isinstance(cache, dict):
        return None
    by_filter = cache.get("by_filter", {})
    if not isinstance(by_filter, dict):
        if not bool(create):
            return None
        by_filter = {}
        cache["by_filter"] = by_filter
    fk = _auto_filter_cache_key(filter_type=str(filter_key or ""))
    bucket = by_filter.get(fk)
    if not isinstance(bucket, dict):
        if not bool(create):
            return None
        bucket = _auto_cache_bucket_template()
        by_filter[fk] = bucket
    if bool(create):
        if not isinstance(bucket.get("items", {}), dict):
            bucket["items"] = {}
        if not isinstance(bucket.get("target_by_measurement", {}), dict):
            bucket["target_by_measurement"] = {}
        if not isinstance(bucket.get("last_used_best", {}), dict):
            bucket["last_used_best"] = {}
    return bucket


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
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    h = hashlib.sha256()
    h.update(_auto_measurement_signature(measurements).encode("ascii", "ignore"))
    keys = {
        "fs": int(fs_v),
        "taps": int(taps_v),
        "filter_type": ft,
        "auto_goal": str(_auto_goal(base_data)),
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


def _auto_seed_from_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> int:
    try:
        sig = _auto_signature(
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=hc_mode,
            include_hc_mode=bool(include_hc_mode),
        )
        if not sig:
            raise ValueError("empty signature")
        return int(str(sig)[:8], 16) & 0xFFFFFFFF
    except Exception:
        try:
            msig = _auto_measurement_signature(measurements or {})
            return int(str(msig)[:8], 16) & 0xFFFFFFFF if msig else 0
        except Exception:
            return 0


def _auto_apply_seed(seed: int) -> None:
    try:
        s = int(seed) & 0xFFFFFFFF
    except Exception:
        s = 0
    try:
        random.seed(s)
    except Exception:
        pass
    try:
        np.random.seed(s)
    except Exception:
        pass


def _auto_cache_load(*, program_version: str | None = None) -> dict:
    global _AUTO_CACHE_VERSION_MISMATCH_LOGGED
    path = _auto_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            obj = {}
        expected_ver = str(program_version or "").strip()
        if not expected_ver:
            return obj
        cached_ver = str(obj.get("program_version", "") or "").strip()
        if cached_ver == expected_ver:
            return obj
        if not bool(_AUTO_CACHE_VERSION_MISMATCH_LOGGED):
            logger.info(
                "Automatic mode cache version mismatch: "
                f"cached='{cached_ver or 'n/a'}', current='{expected_ver}'. "
                "Ignoring cache and running fresh trials."
            )
            _AUTO_CACHE_VERSION_MISMATCH_LOGGED = True
        return _auto_cache_empty(program_version=expected_ver)
    except Exception:
        expected_ver = str(program_version or "").strip()
        if expected_ver:
            return _auto_cache_empty(program_version=expected_ver)
        return {}


def _auto_cache_save(cache: dict, *, program_version: str | None = None) -> None:
    path = _auto_cache_path()
    try:
        cache_obj = dict(cache or {})
        try:
            cache_obj["v"] = int(max(3, int(cache_obj.get("v", 0) or 0)))
        except Exception:
            cache_obj["v"] = 3
        by_filter = cache_obj.get("by_filter", {})
        if not isinstance(by_filter, dict):
            by_filter = {}
        for k in AUTO_MODE_CACHE_FILTER_KEYS:
            if not isinstance(by_filter.get(k), dict):
                by_filter[k] = _auto_cache_bucket_template()
            else:
                if not isinstance(by_filter[k].get("items", {}), dict):
                    by_filter[k]["items"] = {}
                if not isinstance(by_filter[k].get("target_by_measurement", {}), dict):
                    by_filter[k]["target_by_measurement"] = {}
                if not isinstance(by_filter[k].get("last_used_best", {}), dict):
                    by_filter[k]["last_used_best"] = {}
        cache_obj["by_filter"] = by_filter
        ver = str(program_version or cache_obj.get("program_version", "") or "").strip()
        if ver:
            cache_obj["program_version"] = str(ver)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache_obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        return


def _auto_cache_get_entry(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    if not sig:
        return None
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    bucket_items = {}
    if isinstance(bucket, dict):
        raw_bucket_items = bucket.get("items", {})
        if isinstance(raw_bucket_items, dict):
            bucket_items = raw_bucket_items
    if isinstance(bucket_items, dict):
        entry = bucket_items.get(sig)
        if isinstance(entry, dict):
            return dict(entry)
        if len(bucket_items) > 0:
            return None
    items = cache.get("items", {})
    if isinstance(items, dict):
        entry = items.get(sig)
        return dict(entry) if isinstance(entry, dict) else None
    return None


def _auto_cache_get_best(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    entry = _auto_cache_get_entry(sig, filter_key=filter_key, program_version=program_version)
    if not isinstance(entry, dict):
        return None
    best = entry.get("best_preset", {})
    return dict(best) if isinstance(best, dict) else None


def _auto_cache_get_best_target(
    sig: str,
    *,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> str | None:
    entry = _auto_cache_get_entry(sig, filter_key=filter_key, program_version=program_version)
    if not isinstance(entry, dict):
        return None
    hc = str(entry.get("best_target_curve", entry.get("best_hc_mode", "")) or "").strip()
    return _auto_builtin_target_name(hc)


def _auto_cache_get_target_for_measurements(
    measurements: dict,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return None
    cache = _auto_cache_load(program_version=program_version)

    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    target_map = {}
    if isinstance(bucket, dict):
        raw_target_map = bucket.get("target_by_measurement", {})
        if isinstance(raw_target_map, dict):
            target_map = raw_target_map
    if isinstance(target_map, dict):
        direct = target_map.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)
        if len(target_map) > 0:
            return None

    target_map_legacy = cache.get("target_by_measurement", {})
    if isinstance(target_map_legacy, dict):
        direct = target_map_legacy.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map_legacy.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)

    items = {}
    if isinstance(bucket, dict):
        raw_items = bucket.get("items", {})
        if isinstance(raw_items, dict):
            items = raw_items
    if not items:
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
        entry_goal = _auto_goal_norm(str(entry.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
        if entry_goal != goal_norm:
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
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    hc_val = str(best_hc_mode or "").strip()
    if not hc_val:
        return
    msig = _auto_measurement_signature(measurements or {})
    if not msig:
        return
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    target_map = bucket.get("target_by_measurement", {})
    if not isinstance(target_map, dict):
        target_map = {}
    goal_norm = _auto_goal_norm(goal)
    scoped_key = f"{msig}|{goal_norm}"
    target_map[str(scoped_key)] = {
        "t": int(time.time()),
        "measurement_sig": str(msig),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
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
    bucket["target_by_measurement"] = target_map
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)


def _auto_cache_put_best(
    sig: str,
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    if not isinstance(best_preset, dict) or not best_preset:
        return
    sig = str(sig or "").strip()
    if not sig:
        return
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    items = bucket.get("items", {})
    if not isinstance(items, dict):
        items = {}
    entry = {
        "t": int(time.time()),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_preset": dict(best_preset or {}),
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
    bucket["items"] = items
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)


def _auto_cache_get_last_used_best(
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    last_map = {}
    if isinstance(bucket, dict):
        raw_last_map = bucket.get("last_used_best", {})
        if isinstance(raw_last_map, dict):
            last_map = raw_last_map
    if isinstance(last_map, dict):
        direct = last_map.get(str(goal_norm))
        if isinstance(direct, dict):
            return dict(direct)
        if len(last_map) > 0:
            return None
    legacy_map = cache.get("last_used_best", {})
    if not isinstance(legacy_map, dict):
        return None
    direct = legacy_map.get(str(goal_norm))
    return dict(direct) if isinstance(direct, dict) else None


def _auto_cache_put_last_used_best(
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    program_version: str | None = None,
) -> None:
    if not isinstance(best_preset, dict) or not best_preset:
        return
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(program_version=program_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    last_map = bucket.get("last_used_best", {})
    if not isinstance(last_map, dict):
        last_map = {}
    entry = {
        "t": int(time.time()),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_preset": dict(best_preset or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
    last_map[str(goal_norm)] = entry
    bucket["last_used_best"] = last_map
    cache["v"] = 3
    _auto_cache_save(cache, program_version=program_version)
