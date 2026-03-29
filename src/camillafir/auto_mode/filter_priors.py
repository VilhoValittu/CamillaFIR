from __future__ import annotations

import json
from pathlib import Path

from .shared import _auto_filter_cache_key


_BOOL_KEYS = {
    "bass_first_ai",
    "comparison_mode",
    "enable_afdw",
    "enable_tdc",
}

_INT_KEYS = {
    "filter_smooth",
}

_PRIORS_CACHE: dict | None = None


def _priors_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "auto_mode_filter_priors.json"


def _normalize_scalar(key: str, value):
    if key in _BOOL_KEYS:
        return bool(value)
    if key in _INT_KEYS:
        try:
            return int(round(float(value)))
        except Exception:
            return value
    if isinstance(value, float):
        try:
            if float(value).is_integer():
                return int(round(float(value)))
        except Exception:
            return value
    return value


def _normalize_mapping(payload: dict) -> dict:
    out = {}
    for key, value in dict(payload or {}).items():
        out[str(key)] = _normalize_scalar(str(key), value)
    return out


def load_auto_mode_filter_priors(*, force_reload: bool = False) -> dict:
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None and not bool(force_reload):
        return dict(_PRIORS_CACHE)
    path = _priors_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {"filters": {}}
    filters = {}
    for filter_key, payload in dict(data.get("filters", {}) or {}).items():
        filters[str(filter_key)] = {
            "filter_type": str(dict(payload or {}).get("filter_type", "") or ""),
            "auto_defaults": _normalize_mapping(dict(payload or {}).get("auto_defaults", {}) or {}),
            "seed_preset": _normalize_mapping(dict(payload or {}).get("seed_preset", {}) or {}),
        }
    _PRIORS_CACHE = {
        "version": int(dict(data or {}).get("version", 1) or 1),
        "source": str(dict(data or {}).get("source", "") or ""),
        "filters": filters,
    }
    return dict(_PRIORS_CACHE)


def get_auto_mode_filter_prior(filter_type: str | None) -> dict:
    filter_key = str(_auto_filter_cache_key(filter_type=str(filter_type or "")))
    priors = load_auto_mode_filter_priors()
    return dict(dict(priors.get("filters", {}) or {}).get(filter_key, {}) or {})


def get_auto_mode_filter_auto_defaults(filter_type: str | None) -> dict:
    return dict(get_auto_mode_filter_prior(filter_type).get("auto_defaults", {}) or {})


def get_auto_mode_filter_seed_preset(filter_type: str | None) -> dict:
    return dict(get_auto_mode_filter_prior(filter_type).get("seed_preset", {}) or {})
