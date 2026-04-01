"""Advanced-tab preset and summary helpers for NiceGUI."""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl

SAFE = "safe"
NORMAL = "normal"
AGGRESSIVE = "aggressive"
PRESET_KEYS = (SAFE, NORMAL, AGGRESSIVE)

SHAPING_SUMMARY_FIELDS = (
    "max_slope_db_per_oct",
    "max_cut_db",
    "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct",
    "trans_width",
    "filter_smooth",
    "phase_limit",
    "reg_strength",
    "df_smoothing",
)

BASS_SAFETY_SUMMARY_FIELDS = (
    "exc_prot",
    "exc_freq",
    "low_bass_cut_enable",
    "low_bass_cut_hz",
    "low_bass_cut_strength",
    "hpf_enable",
    "hpf_freq",
    "hpf_slope",
    "bass_first_ai",
    "bass_first_mode_max_hz",
)

CONF_PULL_SUMMARY_FIELDS = (
    "conf_pull_floor",
    "conf_pull_ceil",
)

_FILTER_SMOOTH_LABEL_KEYS = {
    1: "filter_smooth_1",
    3: "filter_smooth_3",
    6: "filter_smooth_6",
    12: "filter_smooth_12",
    24: "filter_smooth_24",
    48: "filter_smooth_48",
    96: "filter_smooth_96",
}

_SHAPING_PRESETS = {
    SAFE: {
        "max_slope_db_per_oct": 8.0,
        "max_cut_db": 18.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "trans_width": 140,
        "filter_smooth": 12,
        "phase_limit": 300.0,
        "reg_strength": 35.0,
        "df_smoothing": False,
    },
    NORMAL: {
        "max_slope_db_per_oct": 12.0,
        "max_cut_db": 30.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "trans_width": 100,
        "filter_smooth": 12,
        "phase_limit": 400.0,
        "reg_strength": 30.0,
        "df_smoothing": False,
    },
    AGGRESSIVE: {
        "max_slope_db_per_oct": 18.0,
        "max_cut_db": 36.0,
        "max_slope_boost_db_per_oct": 6.0,
        "max_slope_cut_db_per_oct": 6.0,
        "trans_width": 60,
        "filter_smooth": 24,
        "phase_limit": 500.0,
        "reg_strength": 20.0,
        "df_smoothing": True,
    },
}

_BASS_SAFETY_PRESETS = {
    SAFE: {
        "exc_prot": True,
        "exc_freq": 25.0,
        "low_bass_cut_enable": True,
        "low_bass_cut_hz": 30.0,
        "low_bass_cut_strength": 1.0,
        "hpf_enable": True,
        "hpf_freq": 20.0,
        "hpf_slope": 24,
        "bass_first_ai": False,
        "bass_first_mode_max_hz": 200.0,
    },
    NORMAL: {
        "exc_prot": True,
        "exc_freq": 22.0,
        "low_bass_cut_enable": True,
        "low_bass_cut_hz": 25.0,
        "low_bass_cut_strength": 0.70,
        "hpf_enable": False,
        "hpf_freq": 20.0,
        "hpf_slope": 24,
        "bass_first_ai": False,
        "bass_first_mode_max_hz": 200.0,
    },
    AGGRESSIVE: {
        "exc_prot": False,
        "exc_freq": 20.0,
        "low_bass_cut_enable": False,
        "low_bass_cut_hz": 25.0,
        "low_bass_cut_strength": 0.50,
        "hpf_enable": False,
        "hpf_freq": 18.0,
        "hpf_slope": 24,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
    },
}

_CONF_PULL_PRESETS = {
    SAFE: {
        "conf_pull_floor": 0.20,
        "conf_pull_ceil": 0.75,
    },
    NORMAL: {
        "conf_pull_floor": 0.10,
        "conf_pull_ceil": 0.90,
    },
    AGGRESSIVE: {
        "conf_pull_floor": 0.05,
        "conf_pull_ceil": 1.00,
    },
}


def apply_shaping_preset(preset_key: str, *, t: Callable | None = None) -> None:
    _apply_preset(_SHAPING_PRESETS, preset_key)


def apply_bass_safety_preset(preset_key: str, *, t: Callable | None = None) -> None:
    _apply_preset(_BASS_SAFETY_PRESETS, preset_key)


def apply_conf_pull_preset(preset_key: str, *, t: Callable | None = None) -> None:
    _apply_preset(_CONF_PULL_PRESETS, preset_key)


def build_shaping_summary(*, t: Callable) -> str:
    parts = [
        f"{t('adv_summary_global_rail')}: {_fmt_float(ctrl.value('max_slope_db_per_oct', 12.0), 1)} dB/oct",
        f"{t('adv_summary_boost_rail')}: {_fmt_rail(ctrl.value('max_slope_boost_db_per_oct', 0.0), t=t)}",
        f"{t('adv_summary_cut_rail')}: {_fmt_rail(ctrl.value('max_slope_cut_db_per_oct', 0.0), t=t)}",
        f"{t('adv_summary_max_cut')}: {_fmt_float(ctrl.value('max_cut_db', 30.0), 1)} dB",
        f"{t('adv_summary_transition')}: {_fmt_float(ctrl.value('trans_width', 100.0), 0)} Hz",
        f"{t('adv_summary_smoothing')}: {_filter_smoothing_label(ctrl.value('filter_smooth', 12), t=t)}",
        f"{t('adv_summary_phase_limit')}: {_fmt_float(ctrl.value('phase_limit', 400.0), 1)} Hz",
        f"{t('reg_strength')}: {_fmt_float(ctrl.value('reg_strength', 30.0), 1)} dB",
        f"{t('df_smoothing_label')}: {_state_label(bool(ctrl.value('df_smoothing', False)), t=t)}",
    ]
    return " | ".join(parts)


def build_bass_safety_summary(*, t: Callable) -> str:
    exc_on = bool(ctrl.value("exc_prot", False))
    exc_text = t("state_off")
    if exc_on:
        exc_text = f"{t('state_on')} @ {_fmt_float(ctrl.value('exc_freq', 25.0), 1)} Hz"

    low_cut_on = bool(ctrl.value("low_bass_cut_enable", True))
    low_cut_text = t("state_off")
    if low_cut_on:
        low_cut_text = (
            f"{t('state_on')} @ {_fmt_float(ctrl.value('low_bass_cut_hz', 25.0), 1)} Hz, "
            f"{t('low_bass_cut_strength_label')}: {_fmt_float(ctrl.value('low_bass_cut_strength', 0.7), 2)}"
        )

    hpf_on = bool(ctrl.value("hpf_enable", False))
    hpf_text = t("state_off")
    if hpf_on:
        hpf_text = (
            f"{t('state_on')} @ {_fmt_float(ctrl.value('hpf_freq', 20.0), 1)} Hz / "
            f"{_fmt_float(ctrl.value('hpf_slope', 24), 0)} dB/oct"
        )

    bass_first_on = bool(ctrl.value("bass_first_ai", False))
    bass_first_text = t("state_off")
    if bass_first_on:
        bass_first_text = f"{t('state_on')} <= {_fmt_float(ctrl.value('bass_first_mode_max_hz', 200.0), 1)} Hz"

    parts = [
        f"{t('adv_summary_exc_prot')}: {exc_text}",
        f"{t('adv_summary_low_bass_cut')}: {low_cut_text}",
        f"{t('adv_summary_hpf')}: {hpf_text}",
        f"{t('adv_summary_bass_first')}: {bass_first_text}",
    ]
    return " | ".join(parts)


def build_conf_pull_summary(*, t: Callable) -> str:
    floor_v = _as_float(ctrl.value("conf_pull_floor", 0.0), 0.0)
    ceil_v = _as_float(ctrl.value("conf_pull_ceil", 1.0), 1.0)
    span_v = ceil_v - floor_v
    parts = [
        f"{t('adv_summary_floor')}: {_fmt_float(floor_v, 2)}",
        f"{t('adv_summary_ceil')}: {_fmt_float(ceil_v, 2)}",
        f"{t('adv_summary_span')}: {_fmt_float(span_v, 2)}",
    ]
    return " | ".join(parts)


def render_shaping_summary(*, t: Callable | None = None) -> None:
    _render_summary("adv_shaping_summary_scope", build_shaping_summary(t=_resolve_t(t)))


def render_bass_safety_summary(*, t: Callable | None = None) -> None:
    _render_summary("adv_bass_safety_summary_scope", build_bass_safety_summary(t=_resolve_t(t)))


def render_conf_pull_summary(*, t: Callable | None = None) -> None:
    _render_summary("adv_conf_pull_summary_scope", build_conf_pull_summary(t=_resolve_t(t)))


def update_advanced_guidance_ui(*, t: Callable) -> None:
    render_shaping_summary(t=t)
    render_bass_safety_summary(t=t)
    render_conf_pull_summary(t=t)


def _apply_preset(presets: dict[str, dict[str, object]], preset_key: str) -> None:
    key = _normalize_preset_key(preset_key)
    preset = presets.get(key)
    if preset is None:
        return
    for name, value in preset.items():
        ctrl.set_value(name, value)


def _normalize_preset_key(preset_key: str) -> str:
    key = str(preset_key or "").strip().lower()
    return key if key in PRESET_KEYS else NORMAL


def _resolve_t(t: Callable | None) -> Callable:
    if t is not None:
        return t
    from ..resources.i8n.camillafir_i18n import t as default_t  # noqa: PLC0415

    return default_t


def _render_summary(scope_name: str, summary: str) -> None:
    container = ctrl.get_container(scope_name)
    if container is None:
        return
    try:
        from nicegui import ui  # noqa: PLC0415

        container.clear()
        with container:
            ui.label(summary).classes("w-full text-sm text-gray-300").style("white-space: normal; line-height: 1.5;")
    except Exception:
        return


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt_float(value: object, decimals: int) -> str:
    number = _as_float(value, 0.0)
    text = f"{number:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_rail(value: object, *, t: Callable) -> str:
    rail = _as_float(value, 0.0)
    if abs(rail) <= 1e-9:
        return t("state_off")
    return f"{_fmt_float(rail, 1)} dB/oct"


def _state_label(enabled: bool, *, t: Callable) -> str:
    return t("state_on") if enabled else t("state_off")


def _filter_smoothing_label(value: object, *, t: Callable) -> str:
    try:
        key = _FILTER_SMOOTH_LABEL_KEYS.get(int(float(value)))
    except Exception:
        key = None
    if key is not None:
        return t(key)
    return t("smoothing_level")
