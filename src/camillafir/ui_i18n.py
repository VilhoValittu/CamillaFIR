from __future__ import annotations

from typing import Any, Callable

from .resources.i8n.camillafir_i18n import TRANSLATIONS

LAYOUT_MONO = "mono"
LAYOUT_STEREO = "stereo"

LVL_MODE_AUTO = "auto"
LVL_MODE_MANUAL = "manual"

LVL_ALGO_MEDIAN = "median"
LVL_ALGO_AVERAGE = "average"

TDC_PRESET_SAFE = "safe"
TDC_PRESET_NORMAL = "normal"
TDC_PRESET_AGGRESSIVE = "aggressive"

AFDW_PRESET_TIGHT = "tight"
AFDW_PRESET_BALANCED = "balanced"
AFDW_PRESET_SAFE = "safe"
AFDW_PRESET_MINIMAL = "minimal"

LAYOUT_OPTION_LABEL_KEYS = {
    LAYOUT_MONO: "layout_mono",
    LAYOUT_STEREO: "layout_stereo",
}

LVL_MODE_OPTION_LABEL_KEYS = {
    LVL_MODE_AUTO: "lvl_mode_auto",
    LVL_MODE_MANUAL: "lvl_mode_manual",
}

LVL_ALGO_OPTION_LABEL_KEYS = {
    LVL_ALGO_MEDIAN: "lvl_algo_median",
    LVL_ALGO_AVERAGE: "lvl_algo_average",
}

TDC_PRESET_LABEL_KEYS = {
    TDC_PRESET_SAFE: "tdc_preset_safe",
    TDC_PRESET_NORMAL: "tdc_preset_normal",
    TDC_PRESET_AGGRESSIVE: "tdc_preset_aggressive",
}

AFDW_PRESET_LABEL_KEYS = {
    AFDW_PRESET_TIGHT: "afdw_preset_tight",
    AFDW_PRESET_BALANCED: "afdw_preset_balanced",
    AFDW_PRESET_SAFE: "afdw_preset_safe",
    AFDW_PRESET_MINIMAL: "afdw_preset_minimal",
}

_LAYOUT_LEGACY_LABELS = {
    LAYOUT_MONO: ("Mono",),
    LAYOUT_STEREO: ("Stereo",),
}

_LVL_MODE_LEGACY_LABELS = {
    LVL_MODE_AUTO: ("Auto",),
    LVL_MODE_MANUAL: ("Manual",),
}

_LVL_ALGO_LEGACY_LABELS = {
    LVL_ALGO_MEDIAN: ("Median",),
    LVL_ALGO_AVERAGE: ("Average",),
}

_TDC_PRESET_LEGACY_LABELS = {
    TDC_PRESET_SAFE: ("Safe",),
    TDC_PRESET_NORMAL: ("Normal",),
    TDC_PRESET_AGGRESSIVE: ("Aggressive",),
}

_AFDW_PRESET_LEGACY_LABELS = {
    AFDW_PRESET_TIGHT: ("Tight",),
    AFDW_PRESET_BALANCED: ("Balanced",),
    AFDW_PRESET_SAFE: ("Safe",),
    AFDW_PRESET_MINIMAL: ("Minimal",),
}


def tr_options(t: Callable[[str], str], spec: dict[str, str]) -> dict[str, str]:
    return {stable_key: t(label_key) for stable_key, label_key in spec.items()}


def _as_text(value: Any) -> str:
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _translation_variants(label_key: str, t: Callable[[str], str] | None = None) -> set[str]:
    variants: set[str] = set()

    if callable(t):
        try:
            translated = _as_text(t(label_key))
        except Exception:
            translated = ""
        if translated and translated != label_key:
            variants.add(translated)

    for catalog in TRANSLATIONS.values():
        try:
            translated = _as_text((catalog or {}).get(label_key))
        except Exception:
            translated = ""
        if translated and translated != label_key:
            variants.add(translated)

    return variants


def _normalize_choice_value(
    value: Any,
    *,
    default: str | None,
    label_keys: dict[str, str],
    legacy_labels: dict[str, tuple[str, ...]],
    t: Callable[[str], str] | None = None,
    allow_none: bool = False,
) -> str | None:
    raw = _as_text(value)
    if not raw:
        return None if allow_none else default

    aliases: dict[str, str] = {}
    for stable_key, label_key in label_keys.items():
        aliases[_as_text(stable_key).casefold()] = stable_key
        for legacy in legacy_labels.get(stable_key, ()):
            aliases[_as_text(legacy).casefold()] = stable_key
        for translated in _translation_variants(label_key, t=t):
            aliases[_as_text(translated).casefold()] = stable_key

    return aliases.get(raw.casefold(), None if allow_none else default)


def normalize_layout_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LAYOUT_MONO,
            label_keys=LAYOUT_OPTION_LABEL_KEYS,
            legacy_labels=_LAYOUT_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_lvl_mode_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LVL_MODE_AUTO,
            label_keys=LVL_MODE_OPTION_LABEL_KEYS,
            legacy_labels=_LVL_MODE_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_lvl_algo_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LVL_ALGO_MEDIAN,
            label_keys=LVL_ALGO_OPTION_LABEL_KEYS,
            legacy_labels=_LVL_ALGO_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_tdc_preset_key(value: Any, t: Callable[[str], str] | None = None) -> str | None:
    return _normalize_choice_value(
        value,
        default=None,
        label_keys=TDC_PRESET_LABEL_KEYS,
        legacy_labels=_TDC_PRESET_LEGACY_LABELS,
        t=t,
        allow_none=True,
    )


def normalize_afdw_preset_key(value: Any, t: Callable[[str], str] | None = None) -> str | None:
    return _normalize_choice_value(
        value,
        default=None,
        label_keys=AFDW_PRESET_LABEL_KEYS,
        legacy_labels=_AFDW_PRESET_LEGACY_LABELS,
        t=t,
        allow_none=True,
    )


def layout_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Stereo" if normalize_layout_value(value, t=t) == LAYOUT_STEREO else "Mono"


def lvl_mode_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Manual" if normalize_lvl_mode_value(value, t=t) == LVL_MODE_MANUAL else "Auto"


def lvl_algo_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Average" if normalize_lvl_algo_value(value, t=t) == LVL_ALGO_AVERAGE else "Median"


__all__ = [
    "AFDW_PRESET_BALANCED",
    "AFDW_PRESET_LABEL_KEYS",
    "AFDW_PRESET_MINIMAL",
    "AFDW_PRESET_SAFE",
    "AFDW_PRESET_TIGHT",
    "LAYOUT_MONO",
    "LAYOUT_OPTION_LABEL_KEYS",
    "LAYOUT_STEREO",
    "LVL_ALGO_AVERAGE",
    "LVL_ALGO_MEDIAN",
    "LVL_ALGO_OPTION_LABEL_KEYS",
    "LVL_MODE_AUTO",
    "LVL_MODE_MANUAL",
    "LVL_MODE_OPTION_LABEL_KEYS",
    "TDC_PRESET_AGGRESSIVE",
    "TDC_PRESET_LABEL_KEYS",
    "TDC_PRESET_NORMAL",
    "TDC_PRESET_SAFE",
    "layout_legacy_name",
    "lvl_algo_legacy_name",
    "lvl_mode_legacy_name",
    "normalize_afdw_preset_key",
    "normalize_layout_value",
    "normalize_lvl_algo_value",
    "normalize_lvl_mode_value",
    "normalize_tdc_preset_key",
    "tr_options",
]
