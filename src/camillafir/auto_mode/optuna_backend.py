"""Optuna backend helpers for automatic mode."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from .api import (
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
    AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
    AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS,
    AUTO_MODE_OPTUNA_PRUNING_ENABLED,
    AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
    AUTO_MODE_OPTUNA_STORAGE_FILENAME,
    AUTO_MODE_OPTUNA_TELEMETRY,
    AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    AutoModeConfig,
    _auto_compat_version,
    _auto_goal,
    _auto_goal_norm,
    _auto_optuna_sampler_kwargs,
    _auto_ripple_metric_for_gate,
    _auto_safe_bool,
    _auto_safe_float,
    _auto_safe_int,
    _auto_trial_chunk_size,
    _clear_pruning_hook,
    _set_pruning_hook,
    camillafir_data_dir,
    program_version_token,
)

from .optuna_telemetry import (
    _auto_metric_summary,
    _auto_metric_summary_text,
    _auto_optuna_log_run_telemetry,
    _auto_optuna_fmt_value,
    _auto_optuna_telemetry_text,
    _auto_optuna_telemetry_text_ex,
    _auto_optuna_events_debug_text,
    _auto_optuna_fallback_summary_text,
    _auto_optuna_telemetry_rollup,
)

logger = logging.getLogger("CamillaFIR")
def _auto_import_optuna():
    try:
        import optuna  # type: ignore
    except Exception:
        # Optional dependency: fall back to builtin search if Optuna is absent or unusable.
        logger.debug("Optuna not available; automatic mode will use builtin backend", exc_info=True)
        return None
    return optuna

def _auto_optuna_module_ready(optuna_mod) -> bool:
    if optuna_mod is None:
        return False
    try:
        sampler_cls = getattr(getattr(optuna_mod, "samplers", None), "TPESampler", None)
        create_study = getattr(optuna_mod, "create_study", None)
        trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    except (AttributeError, TypeError):
        return False
    return bool(
        callable(sampler_cls)
        and callable(create_study)
        and trial_state is not None
        and hasattr(trial_state, "FAIL")
    )

def _auto_optuna_storage_filename(*, compat_version: str | None = None) -> str:
    token = str(program_version_token(compat_version, default="") or "").strip()
    if not token:
        return str(AUTO_MODE_OPTUNA_STORAGE_FILENAME)
    stem, ext = os.path.splitext(str(AUTO_MODE_OPTUNA_STORAGE_FILENAME))
    return f"{stem}_{token}{ext or '.log'}"

def _auto_optuna_storage_path(*, compat_version: str | None = None) -> str:
    filename = _auto_optuna_storage_filename(compat_version=compat_version)
    preferred_base = os.fspath(camillafir_data_dir())
    preferred_path = os.path.join(preferred_base, filename)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, filename)

    try:
        os.makedirs(preferred_base, exist_ok=True)
    except OSError:
        try:
            os.makedirs(legacy_base, exist_ok=True)
        except OSError:
            logger.debug("Failed to create legacy Optuna storage directory", exc_info=True)
            pass
        logger.debug("Falling back to legacy Optuna storage directory", exc_info=True)
        return legacy_path
    try:
        source_candidates = [legacy_path]
        if str(filename) != str(AUTO_MODE_OPTUNA_STORAGE_FILENAME):
            source_candidates.extend(
                (
                    os.path.join(preferred_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME),
                    os.path.join(legacy_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME),
                )
            )
        source_path = next(
            (
                path
                for path in source_candidates
                if path != preferred_path and os.path.isfile(path)
            ),
            None,
        )
        if (not os.path.isfile(preferred_path)) and source_path:
            try:
                os.replace(source_path, preferred_path)
            except OSError:
                with open(source_path, "rb") as src_f:
                    payload = src_f.read()
                with open(preferred_path, "wb") as dst_f:
                    dst_f.write(payload)
                try:
                    os.remove(source_path)
                except OSError:
                    logger.debug("Failed to remove migrated Optuna storage source file", exc_info=True)
                    pass
            logger.info(f"Automatic mode Optuna storage migrated to: {preferred_path}")
    except OSError:
        logger.debug("Optuna storage setup failed; falling back to legacy path", exc_info=True)
        return legacy_path
    return preferred_path

def _auto_optuna_study_name(*, study_sig: str | None, scope: str | None) -> str:
    sig_txt = str(study_sig or "").strip().lower()
    scope_txt = str(scope or "study").strip().lower()
    scope_tok = re.sub(r"[^a-z0-9._-]+", "-", scope_txt).strip("-") or "study"
    scope_hash = hashlib.sha1(scope_txt.encode("utf-8", "ignore")).hexdigest()[:12]
    sig_tok = sig_txt[:32] if sig_txt else "nosig"
    return f"camillafir-{scope_tok[:48]}-{scope_hash}-{sig_tok}"

def _auto_optuna_create_storage(optuna_mod, *, base_data: dict | None):
    if not _auto_safe_bool((base_data or {}).get("auto_mode_optuna_persistent_study", True), True):
        return None
    storages_mod = getattr(optuna_mod, "storages", None)
    if storages_mod is None:
        return None
    path = _auto_optuna_storage_path(
        compat_version=_auto_compat_version(base_data),
    )
    candidates = [
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileOpenLock", None),
        ),
        (
            getattr(storages_mod, "JournalStorage", None),
            getattr(storages_mod, "JournalFileStorage", None),
            getattr(storages_mod, "JournalFileOpenLock", None),
        ),
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            None,
        ),
    ]
    for storage_cls, backend_cls, open_lock_cls in candidates:
        if not callable(storage_cls) or not callable(backend_cls):
            continue
        try:
            if callable(open_lock_cls):
                return storage_cls(backend_cls(path, lock_obj=open_lock_cls(path)))
            return storage_cls(backend_cls(path))
        except Exception:
            # Third-party Optuna journal backends differ across versions; try the next compatible variant.
            logger.debug("Optuna journal storage candidate initialization failed", exc_info=True)
            continue
    return None

def _auto_optuna_create_study(
    optuna_mod,
    *,
    sampler,
    pruner=None,
    base_data: dict | None,
    study_name: str | None,
):
    storage = _auto_optuna_create_storage(optuna_mod, base_data=base_data)
    create_kwargs = {"direction": "maximize", "sampler": sampler}
    if pruner is not None:
        create_kwargs["pruner"] = pruner
    if storage is not None and study_name:
        try:
            return optuna_mod.create_study(
                **create_kwargs,
                storage=storage,
                study_name=str(study_name),
                load_if_exists=True,
            )
        except TypeError:
            pass
        except Exception as exc:
            logger.warning(
                "Automatic mode Optuna storage unavailable for study %s (%s: %s). "
                "Falling back to in-memory study.",
                str(study_name),
                type(exc).__name__,
                exc,
            )
    return optuna_mod.create_study(**create_kwargs)

def _auto_optuna_cross_study_best_params(
    optuna_mod,
    *,
    base_data: dict | None,
    scope: str,
    current_study_name: str,
    top_n: int = 8,
) -> list[dict]:
    """Return top param dicts from sibling studies (same scope, different sig)."""
    storage = _auto_optuna_create_storage(optuna_mod, base_data=base_data)
    if storage is None:
        return []
    get_summaries = getattr(optuna_mod, "get_all_study_summaries", None)
    if not callable(get_summaries):
        return []
    try:
        summaries = get_summaries(storage=storage)
    except Exception:
        return []
    seen_sigs: set[str] = set()
    results: list[tuple[float, dict]] = []
    for summary in list(summaries or []):
        sname = str(getattr(summary, "study_name", "") or "")
        if sname == current_study_name:
            continue
        if not scope or scope not in sname:
            continue
        try:
            s = optuna_mod.load_study(study_name=sname, storage=storage)
            trials = s.get_trials(deepcopy=False)
        except Exception:
            continue
        for tr in list(trials or []):
            val = getattr(tr, "value", None)
            try:
                vf = float(val)
            except Exception:
                vf = float("nan")
            if not np.isfinite(vf):
                continue
            try:
                params = dict(getattr(tr, "params", {}) or {})
            except Exception:
                params = {}
            if not params:
                continue
            sig = _auto_optuna_param_signature(params)
            if not sig or sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            results.append((vf, params))
    results.sort(key=lambda kv: kv[0], reverse=True)
    return [p for _, p in results[: int(top_n)]]

def _auto_optuna_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _auto_optuna_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_auto_optuna_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v):
            return "nan"
        if np.isposinf(v):
            return "inf"
        if np.isneginf(v):
            return "-inf"
        return round(v, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)

def _auto_optuna_scope_context_hash(
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    payload = {
        "center": _auto_optuna_jsonable(dict(center or {})),
        "shrink": None if shrink is None else round(float(shrink), 6),
        "extra": _auto_optuna_jsonable(dict(extra or {})),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]

def _auto_optuna_scope_with_context(
    scope_base: str,
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    ctx = _auto_optuna_scope_context_hash(center=center, shrink=shrink, extra=extra)
    return f"{str(scope_base)}-{ctx}"

def _auto_optuna_param_signature(params: dict | None) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    try:
        payload = json.dumps(_auto_optuna_jsonable(params), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = str(params)
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()

def _auto_optuna_trial_params(
    *,
    trial_obj,
    preset: dict | None,
    seed_to_params=None,
) -> dict:
    if callable(seed_to_params):
        try:
            params = dict(seed_to_params(dict(preset or {})) or {})
        except Exception:
            # Seed adapters are pluggable; keep the study alive even if one returns malformed data.
            logger.debug("Optuna seed_to_params adapter failed", exc_info=True)
            params = {}
        if params:
            return params
    try:
        params = dict(getattr(trial_obj, "params", {}) or {})
    except (TypeError, ValueError, AttributeError):
        params = {}
    if params:
        return params
    return dict(preset or {})

def _auto_optuna_trial_payload_preset(user_attrs: dict | None) -> dict:
    payload = dict((user_attrs or {}).get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
    preset = payload.get("trial_preset", None)
    if not isinstance(preset, dict) or not preset:
        preset = payload.get("preset", None)
    return dict(preset or {})

def _auto_optuna_tdc_min(base_data: dict | None) -> float:
    if (_auto_goal_norm(_auto_goal(base_data)) == AUTO_MODE_GOAL_LOW_RIPPLE) and bool((base_data or {}).get("enable_tdc", True)):
        return 55.0
    return 15.0

def _auto_optuna_trial_distributions(optuna_mod, *, params: dict | None, base_data: dict | None) -> dict | None:
    params_in = dict(params or {})
    if not params_in:
        return None
    dist_mod = getattr(optuna_mod, "distributions", None)
    float_dist = getattr(dist_mod, "FloatDistribution", None)
    cat_dist = getattr(dist_mod, "CategoricalDistribution", None)
    if not callable(float_dist) or not callable(cat_dist):
        return None

    tdc_min = float(_auto_optuna_tdc_min(base_data))
    categorical_choices = {
        "tdc_slope_db_per_oct": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0],
        "max_slope_db_per_oct": [8.0, 10.0, 12.0, 14.0, 16.0],
        "max_slope_boost_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "max_slope_cut_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "filter_smooth": [12, 24, 48, 96],
    }
    float_ranges = {
        "fdw_cycles": (5.0, 16.0, 0.01),
        "tdc_strength": (tdc_min, 75.0, 0.1),
        "tdc_max_reduction_db": (6.0, 36.0, 0.1),
        "reg_strength": (15.0, 45.0, 0.1),
        "max_boost": (3.0, 8.0, 0.01),
        "mag_c_min": (float(AUTO_MODE_MAG_C_MIN_MIN_HZ), float(AUTO_MODE_MAG_C_MIN_MAX_HZ), 0.1),
        "mag_c_max": (170.0, 300.0, 0.1),
        "trans_width": (70.0, 150.0, 0.1),
        "bass_first_mode_max_hz": (150.0, 220.0, 0.1),
        "conf_pull_max_hz": (120.0, 260.0, 5.0),
        "low_bass_cut_hz": (float(AUTO_MODE_LOW_BASS_MIN_HZ), float(AUTO_MODE_LOW_BASS_MAX_HZ), 0.1),
        "mixed_freq": (80.0, 320.0, 0.1),
        "phase_limit": (float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ), 0.1),
    }
    # Parameters sampled on a log scale in _suggest_auto_mode_candidate_optuna;
    # distributions must match or enqueue_trial / add_trial will fail.
    log_params = {"mag_c_min", "low_bass_cut_hz", "phase_limit"}

    out: dict[str, object] = {}
    for name in list(params_in.keys()):
        key = str(name)
        if key.endswith("_u"):
            try:
                out[key] = float_dist(0.0, 1.0)
            except Exception:
                return None
            continue
        if key in categorical_choices:
            try:
                out[key] = cat_dist(list(categorical_choices[key]))
            except Exception:
                return None
            continue
        if key in float_ranges:
            lo, hi, _step = float_ranges[key]
            try:
                if key in log_params:
                    out[key] = float_dist(float(lo), float(hi), log=True)
                else:
                    out[key] = float_dist(float(lo), float(hi), step=float(_step))
            except Exception:
                return None
            continue
        return None
    return dict(out)

def _auto_optuna_build_completed_trial(
    optuna_mod,
    *,
    params: dict | None,
    value: float,
    user_attrs: dict | None,
    base_data: dict | None,
    system_attrs: dict | None = None,
):
    create_trial = getattr(getattr(optuna_mod, "trial", None), "create_trial", None)
    if not callable(create_trial):
        create_trial = getattr(optuna_mod, "create_trial", None)
    if not callable(create_trial):
        return None
    distributions = _auto_optuna_trial_distributions(optuna_mod, params=params, base_data=base_data)
    if not distributions:
        return None

    trial_kwargs = {
        "params": dict(params or {}),
        "distributions": dict(distributions),
        "value": float(value),
        "user_attrs": dict(user_attrs or {}),
    }
    if system_attrs:
        trial_kwargs["system_attrs"] = dict(system_attrs)
    try:
        return create_trial(**trial_kwargs)
    except TypeError:
        trial_kwargs.pop("system_attrs", None)
    except Exception:
        return None

    try:
        return create_trial(**trial_kwargs)
    except TypeError:
        pass
    except Exception:
        return None

    trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    complete_state = getattr(trial_state, "COMPLETE", None) if trial_state is not None else None
    if complete_state is None:
        return None
    try:
        return create_trial(state=complete_state, **trial_kwargs)
    except Exception:
        return None

def _auto_optuna_startup_for_phase_kind(cfg, *, phase_kind: str | None, total: int) -> int:
    kind = str(phase_kind or "").strip().lower()
    total_i = int(max(1, total))

    if kind == "phase1":
        base = int(getattr(cfg, "optuna_startup_phase1", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "target":
        base = int(getattr(cfg, "optuna_startup_target", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "local":
        base = int(getattr(cfg, "optuna_startup_local", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "micro":
        base = int(getattr(cfg, "optuna_startup_micro", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    else:
        base = int(getattr(cfg, "optuna_pilot_startup_trials", 12))

    return int(max(1, min(base, total_i)))

def _auto_optuna_is_refine_phase_kind(phase_kind: str | None) -> bool:
    kind = str(phase_kind or "").strip().lower()
    return kind in {"local", "micro"}

def _auto_optuna_constraint_scope_kind(scope: str | None) -> str:
    scope_txt = str(scope or "").strip().lower()
    if not scope_txt:
        return ""
    if "phase2-local" in scope_txt or "local_center_" in scope_txt:
        return "local"
    if "phase3-micro" in scope_txt or "cache-micro" in scope_txt or "cache_micro" in scope_txt:
        return "micro"
    return ""

def _auto_optuna_constraints_enabled_for_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> bool:
    data = dict(base_data or {})
    enabled = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints", AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    )
    if not enabled:
        return False
    refine_only = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints_refine_only", AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY),
        AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    )
    if not refine_only:
        return True
    if str(phase_kind or "").strip():
        return bool(_auto_optuna_is_refine_phase_kind(phase_kind))
    return bool(_auto_optuna_constraint_scope_kind(scope))

def _auto_optuna_effective_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> str:
    scope_txt = str(scope or "study").strip() or "study"
    if _auto_optuna_is_refine_phase_kind(phase_kind) and str(phase_kind or "").strip().lower() == "local":
        if not str(scope_txt).lower().endswith("-locv2") and "-locv2-" not in str(scope_txt).lower():
            scope_txt = f"{scope_txt}-locv2"
    if str(scope_txt).lower().endswith("-c1"):
        return str(scope_txt)
    if _auto_optuna_constraints_enabled_for_scope(base_data, scope_txt, phase_kind=phase_kind):
        return f"{scope_txt}-c1"
    return str(scope_txt)

def _auto_optuna_constraint_thresholds(base_data: dict | None, scope: str | None) -> dict:
    data = dict(base_data or {})
    kind = _auto_optuna_constraint_scope_kind(scope)

    max_mode_ripple = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_mode_ripple_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
        ),
    )
    max_events = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_events_severity",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
        ),
    )
    max_boost = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_net_boost_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
        ),
    )

    return {
        "kind": str(kind),
        "max_mode_ripple_db": float(max_mode_ripple),
        "max_events_severity": float(max_events),
        "max_net_boost_db": float(max_boost),
    }

def _auto_optuna_trial_out_payload(trial) -> dict:
    try:
        user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
    except Exception:
        user_attrs = {}
    out = user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {})
    if isinstance(out, dict):
        return dict(out or {})
    return {}

def _auto_optuna_constraint_vector_from_metrics(
    metrics: dict | None,
    *,
    max_mode_ripple_db: float,
    max_events_severity: float,
    max_net_boost_db: float,
    use_events: bool = True,
) -> tuple[float, float, float]:
    met = dict(metrics or {})

    ripple = _auto_ripple_metric_for_gate(met)
    events = _auto_safe_float(met.get("events_severity", float("nan")), float("nan"))
    boost = _auto_safe_float(met.get("max_net_boost_db", float("nan")), float("nan"))

    ripple_violation = 0.0
    event_violation = 0.0
    boost_violation = 0.0

    if np.isfinite(ripple):
        ripple_violation = float(max(0.0, float(ripple) - float(max_mode_ripple_db)))
    if bool(use_events) and np.isfinite(events):
        event_violation = float(max(0.0, float(events) - float(max_events_severity)))
    if np.isfinite(boost):
        boost_violation = float(max(0.0, float(boost) - float(max_net_boost_db)))

    return (
        float(ripple_violation),
        float(event_violation),
        float(boost_violation),
    )

def _auto_optuna_use_events_constraint(
    base_data: dict | None,
    *,
    phase_kind: str | None,
) -> bool:
    data = dict(base_data or {})
    kind = str(phase_kind or "").strip().lower()

    if kind in {"local", "micro"}:
        return _auto_safe_bool(
            data.get(
                "auto_mode_optuna_constraints_use_events_in_refine",
                AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
        )

    return True

def _auto_optuna_constraints_func(
    *,
    base_data: dict | None,
    scope: str | None,
    phase_kind: str | None = None,
):
    if not _auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind):
        return None

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    logger.info(
        "Automatic mode Optuna constraints: phase_kind=%s use_events=%s scope=%s",
        str(phase_kind or ""),
        str(bool(use_events)),
        str(scope or ""),
    )

    def _constraints(trial):
        out = _auto_optuna_trial_out_payload(trial)
        metrics = dict(out.get("metrics", {}) or {})
        return _auto_optuna_constraint_vector_from_metrics(
            metrics,
            max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
            max_events_severity=float(thr["max_events_severity"]),
            max_net_boost_db=float(thr["max_net_boost_db"]),
            use_events=bool(use_events),
        )

    return _constraints

def _auto_optuna_run_token(
    *,
    study_name: str | None,
    study_scope: str | None,
    seed: int,
    total: int,
    startup_trials: int,
) -> str:
    payload = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "seed": int(seed),
        "total": int(total),
        "startup_trials": int(startup_trials),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]

def _auto_optuna_constraint_info_for_metrics(
    *,
    base_data: dict | None,
    scope: str | None,
    metrics: dict | None,
    phase_kind: str | None = None,
) -> dict:
    enabled = bool(_auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind))
    if not enabled:
        return {
            "constraints_active": False,
            "feasible": None,
            "violations": {},
            "constraint_flags": {},
        }

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    vec = _auto_optuna_constraint_vector_from_metrics(
        dict(metrics or {}),
        max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
        max_events_severity=float(thr["max_events_severity"]),
        max_net_boost_db=float(thr["max_net_boost_db"]),
        use_events=bool(use_events),
    )
    ripple_v, events_v, boost_v = vec
    feasible = bool(
        float(ripple_v) <= 0.0
        and float(events_v) <= 0.0
        and float(boost_v) <= 0.0
    )
    return {
        "constraints_active": True,
        "feasible": bool(feasible),
        "violations": {
            "ripple": float(ripple_v),
            "events": float(events_v),
            "boost": float(boost_v),
        },
        "constraint_flags": {
            "use_events": bool(use_events),
        },
    }

def _auto_optuna_trial_objective_value(trial, out_payload: dict | None = None) -> float | None:
    out = dict(out_payload or {})
    opt_meta = dict(out.get("optuna", {}) or {})
    val = opt_meta.get("objective_value", None)
    try:
        if val is not None and np.isfinite(float(val)):
            return float(val)
    except (TypeError, ValueError):
        pass

    direct_val = getattr(trial, "value", None)
    try:
        if direct_val is not None and np.isfinite(float(direct_val)):
            return float(direct_val)
    except (TypeError, ValueError):
        pass

    vals = getattr(trial, "values", None)
    try:
        if vals and np.isfinite(float(vals[0])):
            return float(vals[0])
    except (TypeError, ValueError, IndexError):
        pass

    return None

def _auto_optuna_attach_out_telemetry(
    out: dict | None,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None = None,
    run_token: str,
    source: str,
    objective_value_num: float | None,
) -> dict:
    out2 = dict(out or {})
    metrics = dict(out2.get("metrics", {}) or {})
    cinfo = _auto_optuna_constraint_info_for_metrics(
        base_data=base_data,
        scope=study_scope,
        metrics=metrics,
        phase_kind=phase_kind,
    )
    out2["optuna"] = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(phase_kind or ""),
        "run_token": str(run_token),
        "source": str(source or ""),
        "objective_value": (
            None
            if objective_value_num is None or not np.isfinite(float(objective_value_num))
            else float(objective_value_num)
        ),
        "constraints_active": bool(cinfo.get("constraints_active", False)),
        "feasible": cinfo.get("feasible", None),
        "violations": dict(cinfo.get("violations", {}) or {}),
        "constraint_flags": dict(cinfo.get("constraint_flags", {}) or {}),
    }
    return out2

def _auto_optuna_base_data_without_constraints(base_data: dict | None) -> dict:
    data = dict(base_data or {})
    data["auto_mode_optuna_constraints"] = False
    return data

def _auto_optuna_needs_zero_feasible_rescue(
    *,
    base_data: dict | None,
    phase_kind: str | None,
    telemetry: dict | None,
) -> bool:
    tel = dict(telemetry or {})
    enabled = _auto_safe_bool(
        (base_data or {}).get(
            "auto_mode_optuna_constraints_zero_feasible_fallback",
            AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
        ),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    )
    if not enabled:
        return False

    kind = str(phase_kind or "").strip().lower()
    if kind not in {"local", "micro"}:
        return False
    if not bool(tel.get("constraints_active", False)):
        return False
    if _auto_optuna_is_refine_phase_kind(phase_kind):
        cflags = dict(tel.get("constraint_flags", {}) or {})
        if not bool(cflags.get("use_events", True)):
            vc = dict(tel.get("violation_counts", {}) or {})
            ripple_bad = int(vc.get("ripple", 0) or 0)
            boost_bad = int(vc.get("boost", 0) or 0)
            if ripple_bad == 0 and boost_bad == 0:
                return False

    complete_n = int(tel.get("complete_trials", 0) or 0)
    feasible_n = int(tel.get("feasible_trials", 0) or 0)
    infeasible_n = int(tel.get("infeasible_trials", 0) or 0)
    return bool(complete_n > 0 and feasible_n == 0 and infeasible_n > 0)

def _auto_optuna_build_run_telemetry(
    study,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None,
    run_token: str,
    requested_total: int,
    startup_trials: int,
    duplicate_skips: int,
    duplicate_replays: int,
    duplicate_reserved: int,
) -> dict:
    try:
        trials = list(
            study.get_trials(deepcopy=False)
            if hasattr(study, "get_trials")
            else getattr(study, "trials", [])
        )
    except Exception:
        trials = []

    run_trials = []
    for tr in list(trials or []):
        try:
            user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
        except Exception:
            user_attrs = {}
        out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
        opt_meta = dict(out.get("optuna", {}) or {})
        if str(opt_meta.get("run_token", "")) == str(run_token):
            run_trials.append((tr, out, opt_meta))

    state_counts = {}
    complete_n = 0
    fail_n = 0
    feasible_n = 0
    infeasible_n = 0
    best_raw_value = None
    best_raw_trial = None
    best_feasible_value = None
    best_feasible_trial = None

    violation_counts = {"ripple": 0, "events": 0, "boost": 0}
    violation_max = {"ripple": 0.0, "events": 0.0, "boost": 0.0}
    source_counts = {}
    events_all = []
    events_feasible = []
    events_infeasible = []
    constraint_flags = {}
    run_phase_kind = ""

    for tr, out, opt_meta in list(run_trials):
        state_obj = getattr(tr, "state", None)
        state_name = str(getattr(state_obj, "name", state_obj or "UNKNOWN"))
        state_counts[state_name] = int(state_counts.get(state_name, 0) or 0) + 1

        source = str(opt_meta.get("source", "") or "")
        if source:
            source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
        if not run_phase_kind:
            run_phase_kind = str(opt_meta.get("phase_kind", "") or "")

        if state_name == "COMPLETE":
            complete_n += 1
        elif state_name == "FAIL":
            fail_n += 1

        constraints_active = bool(opt_meta.get("constraints_active", False))
        feasible = opt_meta.get("feasible", None)
        violations = dict(opt_meta.get("violations", {}) or {})
        trial_constraint_flags = dict(opt_meta.get("constraint_flags", {}) or {})
        if trial_constraint_flags and not constraint_flags:
            constraint_flags = dict(trial_constraint_flags)

        if constraints_active and feasible is True:
            feasible_n += 1
        elif constraints_active and feasible is False:
            infeasible_n += 1

        for key in ("ripple", "events", "boost"):
            v = _auto_safe_float(violations.get(key, 0.0), 0.0)
            if float(v) > 0.0:
                violation_counts[key] = int(violation_counts.get(key, 0) or 0) + 1
                violation_max[key] = float(max(float(violation_max.get(key, 0.0) or 0.0), float(v)))

        if state_name == "COMPLETE":
            metrics = dict(out.get("metrics", {}) or {})
            events_val = _auto_safe_float(metrics.get("events_severity", float("nan")), float("nan"))
            if np.isfinite(events_val):
                events_all.append(float(events_val))
                if constraints_active and feasible is True:
                    events_feasible.append(float(events_val))
                elif constraints_active and feasible is False:
                    events_infeasible.append(float(events_val))
            obj_val = _auto_optuna_trial_objective_value(tr, out)
            if obj_val is not None:
                if best_raw_value is None or float(obj_val) > float(best_raw_value):
                    best_raw_value = float(obj_val)
                    best_raw_trial = int(getattr(tr, "number", -1))

                feasible_ok = opt_meta.get("feasible", None)
                if feasible_ok is True:
                    if best_feasible_value is None or float(obj_val) > float(best_feasible_value):
                        best_feasible_value = float(obj_val)
                        best_feasible_trial = int(getattr(tr, "number", -1))

    startup_complete = int(min(max(1, int(startup_trials)), int(complete_n))) if complete_n > 0 else 0
    model_complete = int(max(0, int(complete_n) - int(startup_complete)))
    constraints_active_any = bool(_auto_optuna_constraints_enabled_for_scope(base_data, study_scope, phase_kind=phase_kind or run_phase_kind))
    constraint_thresholds = {}
    try:
        thr = _auto_optuna_constraint_thresholds(base_data, study_scope)
        constraint_thresholds = {
            "max_events_severity": float(thr["max_events_severity"]),
            "max_mode_ripple_db": float(thr["max_mode_ripple_db"]),
            "max_net_boost_db": float(thr["max_net_boost_db"]),
        }
    except Exception:
        constraint_thresholds = {}

    return {
        "study_name": str(study_name or "in-memory"),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(run_phase_kind or phase_kind or ""),
        "run_token": str(run_token),
        "requested_total": int(requested_total),
        "run_trials": int(len(run_trials)),
        "complete_trials": int(complete_n),
        "failed_trials": int(fail_n),
        "state_counts": dict(state_counts or {}),
        "startup_trials": int(startup_trials),
        "startup_complete": int(startup_complete),
        "model_complete": int(model_complete),
        "duplicate_skips": int(duplicate_skips),
        "duplicate_replays": int(duplicate_replays),
        "duplicate_reserved": int(duplicate_reserved),
        "constraints_active": bool(constraints_active_any),
        "feasible_trials": int(feasible_n) if bool(constraints_active_any) else 0,
        "infeasible_trials": int(infeasible_n) if bool(constraints_active_any) else 0,
        "best_raw_value": best_raw_value,
        "best_raw_trial": best_raw_trial,
        "best_feasible_value": best_feasible_value if bool(constraints_active_any) else None,
        "best_feasible_trial": best_feasible_trial if bool(constraints_active_any) else None,
        "violation_counts": dict(violation_counts or {}),
        "violation_max": dict(violation_max or {}),
        "events_summary": _auto_metric_summary(events_all),
        "events_feasible_summary": _auto_metric_summary(events_feasible),
        "events_infeasible_summary": _auto_metric_summary(events_infeasible),
        "constraint_thresholds": dict(constraint_thresholds or {}),
        "constraint_flags": dict(constraint_flags or {}),
        "source_counts": dict(source_counts or {}),
    }

def _auto_optuna_study_records(study, *, seed_to_params=None) -> dict[str, dict]:
    try:
        trials = study.get_trials(deepcopy=False)
    except TypeError:
        trials = study.get_trials()
    except Exception:
        trials = getattr(study, "trials", [])
    out: dict[str, dict] = {}
    for tr in list(trials or []):
        try:
            user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
        except Exception:
            user_attrs = {}
        payload_preset = _auto_optuna_trial_payload_preset(user_attrs)
        try:
            params = dict(getattr(tr, "params", {}) or {})
        except Exception:
            params = {}
        if callable(seed_to_params) and payload_preset:
            try:
                canonical_params = dict(seed_to_params(dict(payload_preset)) or {})
            except Exception:
                canonical_params = {}
            if canonical_params:
                params = dict(canonical_params)
        sig = _auto_optuna_param_signature(params)
        if not sig:
            continue
        rec = {"params": dict(params)}
        val = getattr(tr, "value", None)
        if val is None:
            vals = getattr(tr, "values", None)
            if isinstance(vals, (list, tuple)) and vals:
                val = vals[0]
        try:
            val_f = float(val)
        except Exception:
            val_f = float("nan")
        if np.isfinite(val_f):
            rec["value"] = float(val_f)
        state = getattr(tr, "state", None)
        if state is not None:
            rec["state"] = state
        cached_out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
        if cached_out:
            rec["out"] = cached_out
        out[sig] = rec
    return out

def _auto_optuna_remember_result(
    optuna_mod,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None = None,
    phase_kind: str | None = None,
    seed: int,
    preset: dict | None,
    metrics: dict | None,
    seed_to_params=None,
    use_refine_tiebreak: bool = False,
    out_payload: dict | None = None,
) -> bool:
    if (not _auto_optuna_module_ready(optuna_mod)) or not study_name or not callable(seed_to_params):
        return False
    params = {}
    try:
        params = dict(seed_to_params(dict(preset or {})) or {})
    except Exception:
        params = {}
    params_sig = _auto_optuna_param_signature(params)
    if not params_sig:
        return False
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=1,
        startup_trials=1,
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=1) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=1,
        **sampler_kwargs,
    )
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        base_data=base_data,
        study_name=study_name,
    )
    if params_sig in _auto_optuna_study_records(study, seed_to_params=seed_to_params):
        return False
    value = _auto_optuna_objective_value(
        dict(metrics or {}),
        use_refine_tiebreak=bool(use_refine_tiebreak),
    )
    payload = _auto_optuna_attach_out_telemetry(
        out_payload
        or {
            "ok": True,
            "metrics": dict(metrics or {}),
            "trial_preset": dict(preset or {}),
            "replayed_from_cache": True,
        },
        base_data=base_data,
        study_name=study_name,
        study_scope=scope_eff,
        phase_kind=phase_kind,
        run_token=run_token,
        source="remembered",
        objective_value_num=float(value) if np.isfinite(value) else None,
    )
    payload_json = _auto_optuna_jsonable(dict(payload or {}))
    if hasattr(study, "add_trial"):
        trial_system_attrs: dict | None = None
        if callable(constraint_fn):
            try:
                thr = _auto_optuna_constraint_thresholds(base_data, scope_eff)
                use_events = _auto_optuna_use_events_constraint(base_data, phase_kind=phase_kind)
                cv = _auto_optuna_constraint_vector_from_metrics(
                    dict(metrics or {}),
                    max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
                    max_events_severity=float(thr["max_events_severity"]),
                    max_net_boost_db=float(thr["max_net_boost_db"]),
                    use_events=bool(use_events),
                )
                trial_system_attrs = {"constraints": list(cv)}
            except Exception:
                trial_system_attrs = None
        add_trial_obj = _auto_optuna_build_completed_trial(
            optuna_mod,
            params=params,
            value=float(value),
            user_attrs={AUTO_MODE_OPTUNA_USER_ATTR_OUT: payload_json},
            base_data=base_data,
            system_attrs=trial_system_attrs,
        )
        if add_trial_obj is not None:
            try:
                study.add_trial(add_trial_obj)
                return True
            except Exception:
                pass
    if not hasattr(study, "enqueue_trial") or not hasattr(study, "ask") or not hasattr(study, "tell"):
        return False
    try:
        study.enqueue_trial(dict(params))
        trial_obj = study.ask()
    except Exception:
        return False
    try:
        if hasattr(trial_obj, "set_user_attr"):
            trial_obj.set_user_attr(
                AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                payload_json,
            )
    except Exception:
        pass
    try:
        study.tell(trial_obj, float(value))
    except Exception:
        return False
    return True

def _auto_optuna_objective_value(metrics: dict | None, *, use_refine_tiebreak: bool = False) -> float:
    met = dict(metrics or {})
    key = "rank_score_refine" if bool(use_refine_tiebreak) else "rank_score"
    value = _auto_safe_float(met.get(key, float("nan")), float("nan"))
    if (not np.isfinite(value)) and bool(use_refine_tiebreak):
        value = _auto_safe_float(met.get("rank_score", float("nan")), float("nan"))
    if np.isfinite(value):
        return float(value)
    return 0.0


@dataclass(slots=True)
class _OptunaEvalContext:
    params: dict
    total: int
    workers: int
    phase_label: str


@dataclass(slots=True)
class _OptunaEvalState:
    context: _OptunaEvalContext
    telemetry: dict = field(default_factory=dict)


def _prepare_optuna_eval_context(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None = None,
    n_total: int,
    seed: int,
    startup_trials: int | None = None,
    base_data: dict | None,
    seed_presets: list[dict] | None,
    build_preset,
    eval_one,
    consume_one,
    objective_value,
    workers: int,
    seed_to_params=None,
    study_name: str | None = None,
    study_scope: str | None = None,
    phase_label: str | None = None,
    phase_kind: str | None = None,
) -> _OptunaEvalContext:
    return _OptunaEvalContext(
        params={
            "optuna_mod": optuna_mod,
            "cfg": cfg,
            "n_total": int(n_total),
            "seed": int(seed),
            "startup_trials": startup_trials,
            "base_data": dict(base_data or {}) if isinstance(base_data, dict) else base_data,
            "seed_presets": list(seed_presets or []) if seed_presets is not None else None,
            "build_preset": build_preset,
            "eval_one": eval_one,
            "consume_one": consume_one,
            "objective_value": objective_value,
            "workers": int(workers),
            "seed_to_params": seed_to_params,
            "study_name": study_name,
            "study_scope": study_scope,
            "phase_label": phase_label,
            "phase_kind": phase_kind,
        },
        total=int(max(0, n_total)),
        workers=int(workers),
        phase_label=str(phase_label or study_scope or study_name or "optuna"),
    )


def _submit_or_schedule_trials(
    *,
    context: _OptunaEvalContext,
) -> _OptunaEvalState:
    telemetry = _auto_run_optuna_eval_loop_core(**dict(context.params or {}))
    return _OptunaEvalState(
        context=context,
        telemetry=dict(telemetry or {}),
    )


def _consume_completed_trial(
    *,
    state: _OptunaEvalState,
) -> _OptunaEvalState:
    return state


def _update_best_and_telemetry(
    *,
    state: _OptunaEvalState,
) -> _OptunaEvalState:
    return _OptunaEvalState(
        context=state.context,
        telemetry=dict(state.telemetry or {}),
    )


def _finalize_optuna_eval_loop(
    *,
    state: _OptunaEvalState,
) -> dict:
    return dict(state.telemetry or {})


def _log_optuna_duplicate_summary(*, duplicate_skips: int, study_name: str | None) -> None:
    if int(duplicate_skips) <= 0:
        return
    logger.info(
        "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
        int(duplicate_skips),
        str(study_name or "in-memory"),
    )


def _run_optuna_seed_trials(
    *,
    total: int,
    seed_items: list[dict],
    seed_to_params,
    ask_new_trial,
    eval_one,
    consume_one,
    tell_trial,
    finalize_telemetry,
) -> tuple[int, dict | None]:
    idx_next = 1
    for preset in list(seed_items):
        if idx_next > int(total):
            return int(idx_next), dict(finalize_telemetry() or {})
        trial_obj = None
        params_sig = ""
        preset_eval = dict(preset or {})
        if callable(seed_to_params):
            trial_obj, preset_eval, params_sig, ask_error = ask_new_trial()
            if trial_obj is None:
                out = {
                    "idx": int(idx_next),
                    "ok": False,
                    "error": str(ask_error or "no unique optuna candidate available"),
                }
                if consume_one(int(idx_next), dict(out or {})):
                    return int(idx_next), dict(finalize_telemetry() or {})
                idx_next += 1
                continue
        try:
            out = eval_one(int(idx_next), dict(preset_eval or {}))
        except Exception as exc:
            out = {
                "idx": int(idx_next),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            _clear_pruning_hook()
        if trial_obj is not None:
            tell_trial(trial_obj, out, params_sig=params_sig, source="seed")
        if consume_one(int(idx_next), dict(out or {})):
            return int(idx_next), dict(finalize_telemetry() or {})
        idx_next += 1
    return int(idx_next), None


def _run_optuna_serial_trials(
    *,
    idx_next: int,
    total: int,
    ask_new_trial,
    eval_one,
    consume_one,
    tell_trial,
    finalize_telemetry,
    pruner,
    make_pruning_hook,
    trial_pruned_cls,
    pruned_state,
    study,
    reserved_signatures: set[str],
) -> dict | None:
    for idx in range(int(idx_next), int(total) + 1):
        trial_obj, preset, params_sig, ask_error = ask_new_trial()
        if trial_obj is None:
            out = {
                "idx": int(idx),
                "ok": False,
                "error": str(ask_error or "no unique optuna candidate available"),
            }
            if consume_one(int(idx), dict(out or {})):
                return dict(finalize_telemetry() or {})
            continue
        if pruner is not None:
            _set_pruning_hook(make_pruning_hook(trial_obj))
        try:
            out = eval_one(int(idx), dict(preset))
        except Exception as exc:
            if trial_pruned_cls is not None and isinstance(exc, trial_pruned_cls):
                if pruned_state is not None:
                    try:
                        study.tell(trial_obj, state=pruned_state)
                    except Exception:
                        pass
                if params_sig:
                    reserved_signatures.discard(str(params_sig))
                _clear_pruning_hook()
                continue
            out = {
                "idx": int(idx),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            _clear_pruning_hook()
        tell_trial(trial_obj, out, params_sig=params_sig, source="optuna")
        if consume_one(int(idx), dict(out or {})):
            return dict(finalize_telemetry() or {})
    return None


def _run_optuna_parallel_trials(
    *,
    idx_next: int,
    total: int,
    workers: int,
    chunk_size: int,
    ask_new_trial,
    eval_with_hook,
    consume_one,
    tell_trial,
    finalize_telemetry,
    trial_pruned_cls,
    pruned_state,
    study,
    reserved_signatures: set[str],
) -> dict | None:
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        idx_cursor = int(idx_next)
        while idx_cursor <= int(total):
            chunk_items = []
            while idx_cursor <= int(total) and len(chunk_items) < int(chunk_size):
                trial_obj, preset, params_sig, ask_error = ask_new_trial()
                if trial_obj is None:
                    chunk_items.append(
                        (
                            int(idx_cursor),
                            None,
                            {},
                            "",
                            {
                                "idx": int(idx_cursor),
                                "ok": False,
                                "error": str(ask_error or "no unique optuna candidate available"),
                            },
                        )
                    )
                    idx_cursor += 1
                    continue
                chunk_items.append((int(idx_cursor), trial_obj, dict(preset), str(params_sig), None))
                idx_cursor += 1
            if not chunk_items:
                break

            fut_map = {
                ex.submit(eval_with_hook, int(idx), dict(preset), trial_obj): (int(idx), trial_obj, str(params_sig))
                for idx, trial_obj, preset, params_sig, pre_out in chunk_items
                if trial_obj is not None and pre_out is None
            }
            chunk_out: dict[int, dict] = {}
            for fut in as_completed(list(fut_map.keys())):
                idx, trial_obj, params_sig = fut_map.get(fut, (0, None, ""))
                try:
                    out = fut.result()
                    if not isinstance(out, dict):
                        out = {"idx": int(idx), "ok": False, "error": "invalid worker result"}
                except Exception as exc:
                    if trial_pruned_cls is not None and isinstance(exc, trial_pruned_cls):
                        if trial_obj is not None and pruned_state is not None:
                            try:
                                study.tell(trial_obj, state=pruned_state)
                            except Exception:
                                pass
                        if params_sig:
                            reserved_signatures.discard(str(params_sig))
                        continue
                    out = {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                tell_trial(trial_obj, out, params_sig=params_sig, source="optuna")
                chunk_out[int(idx)] = dict(out or {})

            for idx, _trial_obj, _preset, _params_sig, pre_out in chunk_items:
                if isinstance(pre_out, dict):
                    out = dict(pre_out or {})
                else:
                    out = dict(
                        chunk_out.get(
                            int(idx),
                            {"idx": int(idx), "ok": False, "error": "missing worker result"},
                        )
                        or {}
                    )
                if consume_one(int(idx), out):
                    return dict(finalize_telemetry() or {})
    return None


def _auto_run_optuna_eval_loop(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None = None,
    n_total: int,
    seed: int,
    startup_trials: int | None = None,
    base_data: dict | None,
    seed_presets: list[dict] | None,
    build_preset,
    eval_one,
    consume_one,
    objective_value,
    workers: int,
    seed_to_params=None,
    study_name: str | None = None,
    study_scope: str | None = None,
    phase_label: str | None = None,
    phase_kind: str | None = None,
) -> dict:
    context = _prepare_optuna_eval_context(
        optuna_mod=optuna_mod,
        cfg=cfg,
        n_total=int(n_total),
        seed=int(seed),
        startup_trials=startup_trials,
        base_data=base_data,
        seed_presets=seed_presets,
        build_preset=build_preset,
        eval_one=eval_one,
        consume_one=consume_one,
        objective_value=objective_value,
        workers=int(workers),
        seed_to_params=seed_to_params,
        study_name=study_name,
        study_scope=study_scope,
        phase_label=phase_label,
        phase_kind=phase_kind,
    )
    scheduled = _submit_or_schedule_trials(context=context)
    consumed = _consume_completed_trial(state=scheduled)
    telemetry = _update_best_and_telemetry(state=consumed)
    return _finalize_optuna_eval_loop(state=telemetry)


def _auto_run_optuna_eval_loop_core(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None = None,
    n_total: int,
    seed: int,
    startup_trials: int | None = None,
    base_data: dict | None,
    seed_presets: list[dict] | None,
    build_preset,
    eval_one,
    consume_one,
    objective_value,
    workers: int,
    seed_to_params=None,
    study_name: str | None = None,
    study_scope: str | None = None,
    phase_label: str | None = None,
    phase_kind: str | None = None,
) -> dict:
    if not _auto_optuna_module_ready(optuna_mod):
        return {}
    total = int(max(0, n_total))
    if total <= 0:
        return {}
    cfg_optuna = cfg if isinstance(cfg, AutoModeConfig) else AutoModeConfig.from_base_data(base_data)
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    startup_effective = _auto_optuna_startup_for_phase_kind(
        cfg_optuna,
        phase_kind=phase_kind,
        total=int(total),
    )
    if startup_trials is not None and not str(phase_kind or "").strip():
        startup_effective = int(max(1, min(int(total), _auto_safe_int(startup_trials, startup_effective))))
    logger.info(
        "Automatic mode Optuna startup policy: phase_kind=%s scope=%s total=%d startup=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(total),
        int(startup_effective),
    )
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=int(total),
        startup_trials=int(startup_effective),
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=int(workers)) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(startup_effective),
        **sampler_kwargs,
    )
    pruning_enabled = bool(
        _auto_safe_bool(
            (base_data or {}).get("auto_mode_optuna_pruning_enabled", AUTO_MODE_OPTUNA_PRUNING_ENABLED),
            AUTO_MODE_OPTUNA_PRUNING_ENABLED,
        )
    )
    pruner = None
    if bool(pruning_enabled):
        pruning_n_startup = max(
            1,
            _auto_safe_int(
                (base_data or {}).get("auto_mode_optuna_pruning_n_startup", AUTO_MODE_OPTUNA_PRUNING_N_STARTUP),
                AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
            ),
        )
        pruners_mod = getattr(optuna_mod, "pruners", None)
        median_pruner_cls = getattr(pruners_mod, "MedianPruner", None) if pruners_mod is not None else None
        if callable(median_pruner_cls):
            try:
                pruner = median_pruner_cls(
                    n_startup_trials=int(pruning_n_startup),
                    n_warmup_steps=0,
                    interval_steps=1,
                )
            except Exception:
                pruner = None
    logger.info(
        "Automatic mode Optuna study %s: startup=%d total=%d pruning=%s",
        str(study_name or "in-memory"),
        int(startup_effective),
        int(total),
        "on" if pruner is not None else "off",
    )
    logger.info(
        "Automatic mode Optuna phase=%s scope=%s startup=%d total=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(startup_effective),
        int(total),
    )
    if callable(constraint_fn):
        thr = _auto_optuna_constraint_thresholds(base_data, scope_eff)
        use_events = _auto_optuna_use_events_constraint(
            base_data,
            phase_kind=phase_kind,
        )
        logger.info(
            "Automatic mode Optuna constraints enabled: scope=%s ripple<=%.3f events=%s boost<=%.3f",
            str(scope_eff),
            float(thr["max_mode_ripple_db"]),
            "off" if not bool(use_events) else f"{float(thr['max_events_severity']):.3f}",
            float(thr["max_net_boost_db"]),
        )
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        pruner=pruner,
        base_data=base_data,
        study_name=study_name,
    )
    if (
        _auto_safe_bool(
            (base_data or {}).get("auto_mode_optuna_cross_study_seeds", AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS),
            AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
        )
        and study_name
        and hasattr(study, "enqueue_trial")
    ):
        try:
            _existing_complete = [
                tr for tr in study.get_trials(deepcopy=False)
                if getattr(tr, "value", None) is not None
            ]
        except Exception:
            _existing_complete = []
        if not _existing_complete:
            _top_n_cross = max(
                1,
                _auto_safe_int(
                    (base_data or {}).get(
                        "auto_mode_optuna_cross_study_seeds_top_n",
                        AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                    ),
                    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                ),
            )
            _cross_params = _auto_optuna_cross_study_best_params(
                optuna_mod,
                base_data=base_data,
                scope=str(scope_eff or ""),
                current_study_name=str(study_name),
                top_n=int(_top_n_cross),
            )
            _cross_enqueued = 0
            for _cp in _cross_params:
                try:
                    study.enqueue_trial(dict(_cp))
                    _cross_enqueued += 1
                except Exception:
                    pass
            if _cross_enqueued:
                logger.info(
                    "Automatic mode cross-study seeds: enqueued %d trials from sibling studies (scope=%s)",
                    _cross_enqueued,
                    str(scope_eff or ""),
                )
    _trial_pruned_cls = getattr(optuna_mod, "TrialPruned", None)
    _trial_pruned_state = getattr(
        getattr(optuna_mod, "trial", None),
        "TrialState",
        None,
    )
    _pruned_state = getattr(_trial_pruned_state, "PRUNED", None) if _trial_pruned_state is not None else None
    fail_state = optuna_mod.trial.TrialState.FAIL
    duplicate_guard = bool(
        _auto_safe_bool((base_data or {}).get("auto_mode_optuna_avoid_duplicates", True), True)
    )
    known_records = _auto_optuna_study_records(study, seed_to_params=seed_to_params) if bool(duplicate_guard) else {}
    reserved_signatures: set[str] = set()
    duplicate_skips = 0
    duplicate_replays = 0
    duplicate_reserved = 0

    def _make_pruning_hook(trial_obj_ref):
        """Return a hook that reports a partial score and raises TrialPruned if warranted."""
        step_counter = [0]
        def _hook(partial_score: float) -> None:
            try:
                trial_obj_ref.report(float(partial_score), step=step_counter[0])
                step_counter[0] += 1
                should = trial_obj_ref.should_prune()
            except Exception:
                return
            if bool(should) and _trial_pruned_cls is not None:
                raise _trial_pruned_cls()
        return _hook

    def _finalize_telemetry() -> dict:
        if not bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry", AUTO_MODE_OPTUNA_TELEMETRY),
                AUTO_MODE_OPTUNA_TELEMETRY,
            )
        ):
            return {}
        telemetry = _auto_optuna_build_run_telemetry(
            study,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            requested_total=int(total),
            startup_trials=int(startup_effective),
            duplicate_skips=int(duplicate_skips),
            duplicate_replays=int(duplicate_replays),
            duplicate_reserved=int(duplicate_reserved),
        )
        if bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry_log_summary", AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY),
                AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
            )
        ):
            _auto_optuna_log_run_telemetry(
                logger,
                phase_label=str(phase_label or scope_eff or "optuna"),
                tel=telemetry,
            )
        return dict(telemetry or {})

    def _tell(trial_obj, out: dict, *, params_sig: str = "", source: str = "optuna") -> None:
        value = None
        out_payload = dict(out or {})
        if bool(out_payload.get("ok", False)):
            try:
                value = float(objective_value(dict(out_payload or {})))
                if not np.isfinite(value):
                    value = 0.0
            except Exception:
                value = 0.0
        out_payload = _auto_optuna_attach_out_telemetry(
            out_payload,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            source=str(source or "optuna"),
            objective_value_num=value,
        )
        try:
            if hasattr(trial_obj, "set_user_attr"):
                trial_obj.set_user_attr(
                    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                    _auto_optuna_jsonable(dict(out_payload or {})),
                )
        except Exception:
            pass
        try:
            if bool(dict(out_payload or {}).get("ok", False)):
                study.tell(trial_obj, float(value))
            else:
                study.tell(trial_obj, state=fail_state)
        except Exception:
            pass
        if params_sig:
            reserved_signatures.discard(str(params_sig))
            rec = {"params_sig": str(params_sig)}
            if bool(dict(out_payload or {}).get("ok", False)) and value is not None and np.isfinite(value):
                rec["value"] = float(value)
            else:
                rec["state"] = fail_state
            if isinstance(out_payload, dict) and out_payload:
                rec["out"] = dict(out_payload or {})
            known_records[str(params_sig)] = rec

    def _reuse_duplicate_trial(trial_obj, params_sig: str) -> None:
        rec = dict(known_records.get(str(params_sig), {}) or {})
        out_prev = dict(rec.get("out", {}) or {})
        val = rec.get("value", None)
        out_payload = _auto_optuna_attach_out_telemetry(
            out_prev,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            source="replayed",
            objective_value_num=(
                float(val)
                if val is not None and np.isfinite(_auto_safe_float(val, float("nan")))
                else None
            ),
        )
        if out_payload and hasattr(trial_obj, "set_user_attr"):
            try:
                trial_obj.set_user_attr(
                    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                    _auto_optuna_jsonable(out_payload),
                )
            except Exception:
                pass
        try:
            if val is not None and np.isfinite(float(val)):
                study.tell(trial_obj, float(val))
            else:
                study.tell(trial_obj, state=fail_state)
        except Exception:
            pass

    def _ask_new_trial():
        nonlocal duplicate_reserved, duplicate_replays, duplicate_skips
        attempts = int(max(1, AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS))
        last_error = None
        for _ in range(attempts):
            try:
                trial_obj = study.ask()
                preset = dict(build_preset(trial_obj) or {})
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            params = _auto_optuna_trial_params(
                trial_obj=trial_obj,
                preset=preset,
                seed_to_params=seed_to_params,
            )
            params_sig = _auto_optuna_param_signature(params)
            if (not bool(duplicate_guard)) or (not params_sig):
                if params_sig:
                    reserved_signatures.add(str(params_sig))
                return trial_obj, preset, str(params_sig), None
            if params_sig in reserved_signatures:
                duplicate_skips += 1
                duplicate_reserved += 1
                reserved_out = _auto_optuna_attach_out_telemetry(
                    {
                        "ok": False,
                        "error": "duplicate suggestion reserved in current batch",
                    },
                    base_data=base_data,
                    study_name=study_name,
                    study_scope=scope_eff,
                    phase_kind=phase_kind,
                    run_token=run_token,
                    source="reserved",
                    objective_value_num=None,
                )
                try:
                    if hasattr(trial_obj, "set_user_attr"):
                        trial_obj.set_user_attr(
                            AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                            _auto_optuna_jsonable(dict(reserved_out or {})),
                        )
                    study.tell(trial_obj, state=fail_state)
                except Exception:
                    pass
                continue
            if params_sig in known_records:
                duplicate_skips += 1
                duplicate_replays += 1
                _reuse_duplicate_trial(trial_obj, str(params_sig))
                continue
            reserved_signatures.add(str(params_sig))
            return trial_obj, preset, str(params_sig), None
        return None, {}, "", str(last_error or "no unique optuna candidate available")

    seed_items = list(seed_presets or [])[: int(total)]
    if callable(seed_to_params) and hasattr(study, "enqueue_trial"):
        seed_items_filtered = []
        enqueued_signatures: set[str] = set()
        for preset in list(seed_items):
            try:
                params = dict(seed_to_params(dict(preset or {})) or {})
            except Exception:
                params = {}
            params_sig = _auto_optuna_param_signature(params)
            if bool(duplicate_guard) and params_sig and (
                params_sig in known_records or params_sig in enqueued_signatures
            ):
                duplicate_skips += 1
                continue
            if params:
                try:
                    study.enqueue_trial(dict(params))
                    if params_sig:
                        enqueued_signatures.add(str(params_sig))
                except Exception:
                    pass
            seed_items_filtered.append(dict(preset or {}))
        seed_items = list(seed_items_filtered)

    idx_next, seed_telemetry = _run_optuna_seed_trials(
        total=int(total),
        seed_items=list(seed_items or []),
        seed_to_params=seed_to_params,
        ask_new_trial=_ask_new_trial,
        eval_one=eval_one,
        consume_one=consume_one,
        tell_trial=_tell,
        finalize_telemetry=_finalize_telemetry,
    )
    if seed_telemetry is not None:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return dict(seed_telemetry or {})
    if idx_next > total:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return _finalize_telemetry()

    remaining = int(total - idx_next + 1)
    if workers <= 1 or remaining <= 1:
        serial_telemetry = _run_optuna_serial_trials(
            idx_next=int(idx_next),
            total=int(total),
            ask_new_trial=_ask_new_trial,
            eval_one=eval_one,
            consume_one=consume_one,
            tell_trial=_tell,
            finalize_telemetry=_finalize_telemetry,
            pruner=pruner,
            make_pruning_hook=_make_pruning_hook,
            trial_pruned_cls=_trial_pruned_cls,
            pruned_state=_pruned_state,
            study=study,
            reserved_signatures=reserved_signatures,
        )
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return dict(serial_telemetry or _finalize_telemetry() or {})

    chunk_size = int(_auto_trial_chunk_size(workers))

    def _eval_with_hook(idx, preset, trial_obj_ref):
        if pruner is not None and trial_obj_ref is not None:
            _set_pruning_hook(_make_pruning_hook(trial_obj_ref))
        try:
            return eval_one(int(idx), dict(preset))
        finally:
            _clear_pruning_hook()

    parallel_telemetry = _run_optuna_parallel_trials(
        idx_next=int(idx_next),
        total=int(total),
        workers=int(workers),
        chunk_size=int(chunk_size),
        ask_new_trial=_ask_new_trial,
        eval_with_hook=_eval_with_hook,
        consume_one=consume_one,
        tell_trial=_tell,
        finalize_telemetry=_finalize_telemetry,
        trial_pruned_cls=_trial_pruned_cls,
        pruned_state=_pruned_state,
        study=study,
        reserved_signatures=reserved_signatures,
    )
    _log_optuna_duplicate_summary(
        duplicate_skips=int(duplicate_skips),
        study_name=study_name,
    )
    return dict(parallel_telemetry or _finalize_telemetry() or {})
