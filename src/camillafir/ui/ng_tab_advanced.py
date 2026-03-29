"""NiceGUI Advanced tab builder.

Replaces build_advanced_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl

_SLOPE_OPTS = [6, 12, 18, 24, 36, 48]


def build_advanced_tab(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    from nicegui import ui

    ui.markdown(f"#### {t('tab_adv')}")
    ui.separator()

    # ---- [A] Filter Shaping / Correction rails ----
    ui.label(f"🧩 {t('ui_correction_shaping_rails')}").classes("text-sm font-semibold")

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "max_slope_db_per_oct",
            ui.number(
                label=t("max_slope_db_per_oct"),
                value=float(get_val("max_slope_db_per_oct", 12.0) or 12.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "max_cut_db",
            ui.number(
                label=t("max_cut_db"),
                value=float(get_val("max_cut_db", 30.0) or 30.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "max_slope_boost_db_per_oct",
            ui.number(
                label=t("max_slope_boost_db_per_oct"),
                value=float(get_val("max_slope_boost_db_per_oct", 0.0) or 0.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "max_slope_cut_db_per_oct",
            ui.number(
                label=t("max_slope_cut_db_per_oct"),
                value=float(get_val("max_slope_cut_db_per_oct", 0.0) or 0.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "trans_width",
            ui.number(
                label="Transition Width (Hz)",
                value=int(get_val("trans_width", 100) or 100),
                format="%d",
            ).props("dense outlined").classes("flex-1"),
        )

    ui.separator()

    ctrl.register(
        "filter_smooth",
        ui.select(
            options={
                1:  "1/1 Octave",
                3:  "1/3 Octave",
                6:  "1/6 Octave",
                12: "1/12 Octave (Standard)",
                24: "1/24 Octave (Fine)",
                48: "1/48 Octave (Ultra)",
                96: "1/96 Octave (HC)",
            },
            value=get_val("filter_smooth", get_val("smoothing_level", 12)),
            label=t("smoothing_level"),
        ).props("dense outlined").classes("w-full"),
    )

    ctrl.register(
        "df_smoothing",
        ui.checkbox(
            f"{t('df_smoothing_label')} [EXPERIMENTAL]",
            value=bool(get_val("df_smoothing", False)),
        ),
    )
    ctrl.register(
        "reg_strength",
        ui.number(
            label=t("reg_strength"),
            value=float(get_val("reg_strength", 30.0) or 30.0),
            format="%.1f",
        ).props("dense outlined").classes("w-full"),
    )
    ctrl.register(
        "phase_limit",
        ui.number(
            label=t("phase_limit"),
            value=float(get_val("phase_limit", 400.0) or 400.0),
            format="%.1f",
        ).props("dense outlined").classes("w-full"),
    )

    ui.separator()

    # ---- [B] Bass First AI ----
    ui.label(f"🧠 {t('bass_first_title')}").classes("text-sm font-semibold")
    ctrl.register(
        "bass_first_ai",
        ui.checkbox(
            t("bass_first_enable_label"),
            value=bool(get_val("bass_first_ai", False)),
        ),
    )
    bass_first_col = ui.column().classes("w-full")
    ctrl.register_container("bass_first_max_hz_scope", bass_first_col)
    with bass_first_col:
        ctrl.register(
            "bass_first_mode_max_hz",
            ui.number(
                label="Bass First max Hz",
                value=float(get_val("bass_first_mode_max_hz", 200.0) or 200.0),
                format="%.1f",
            ).props("dense outlined").classes("w-full"),
        )
    bass_first_col.set_visibility(False)

    ui.separator()

    # ---- [C] Channel Linking ----
    ui.label(f"🔗 {t('enable_link')}").classes("text-sm font-semibold")
    ctrl.register(
        "stereo_link",
        ui.checkbox(
            t("enable_link"),
            value=bool(get_val("stereo_link", False)),
        ),
    )
    ctrl.register(
        "stereo_link_strategy",
        ui.select(
            options={
                "auto":   t("stereo_link_mode_auto"),
                "hybrid": t("stereo_link_mode_hybrid"),
                "shared": t("stereo_link_mode_shared_legacy"),
            },
            value=str(get_val("stereo_link_strategy", "auto") or "auto"),
            label=t("stereo_link_mode"),
        ).props("dense outlined").classes("w-full"),
    )

    ui.separator()

    # ---- [D] Bass Safety ----
    ui.label("🛡️ Bass Safety").classes("text-sm font-semibold")

    with ui.row().classes("w-full gap-4 items-end"):
        ctrl.register(
            "exc_prot",
            ui.checkbox(
                t("exc_prot_title"),
                value=bool(get_val("exc_prot", False)),
            ),
        )
        ctrl.register(
            "exc_freq",
            ui.number(
                label=t("exc_freq"),
                value=float(get_val("exc_freq", 25.0) or 25.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )

    with ui.expansion(t("guide_exc_prot_title")).classes("w-full"):
        ui.markdown(t("guide_exc_prot_body"))

    ui.separator()

    ctrl.register(
        "low_bass_cut_enable",
        ui.checkbox(
            t("low_bass_cut_hz"),
            value=bool(get_val("low_bass_cut_enable", True)),
        ),
    )
    bass_cut_col = ui.column().classes("w-full")
    ctrl.register_container("low_bass_cut_scope", bass_cut_col)
    with bass_cut_col:
        ctrl.register(
            "low_bass_cut_hz",
            ui.number(
                label=t("low_bass_cut_hz"),
                value=float(get_val("low_bass_cut_hz", 30.0) or 30.0),
                format="%.1f",
            ).props("dense outlined").classes("w-full"),
        )
        ctrl.register(
            "low_bass_cut_strength",
            ui.number(
                label="Bass cut strength",
                value=float(get_val("low_bass_cut_strength", 1.0) or 1.0),
                format="%.2f",
            ).props("dense outlined").classes("w-full"),
        )

    with ui.expansion(t("guide_low_bass_cut_title")).classes("w-full"):
        ui.markdown(t("guide_low_bass_cut_body"))

    ui.separator()

    with ui.row().classes("w-full gap-4 items-end"):
        ctrl.register(
            "hpf_enable",
            ui.checkbox(
                t("hpf_enable"),
                value=bool(get_val("hpf_enable", False)),
            ),
        )
        ctrl.register(
            "hpf_freq",
            ui.number(
                label=t("hpf_freq"),
                value=float(get_val("hpf_freq", 20.0) or 20.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "hpf_slope",
            ui.select(
                _SLOPE_OPTS,
                value=get_val("hpf_slope", 24),
                label=t("hpf_slope"),
            ).props("dense outlined").classes("flex-1"),
        )

    ui.separator()

    # Confidence pull (ADVANCED only, hidden by default)
    conf_pull_col = ui.column().classes("w-full")
    ctrl.register_container("conf_pull_scope", conf_pull_col)
    with conf_pull_col:
        ui.label("Confidence Pull").classes("text-sm font-semibold")
        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "conf_pull_floor",
                ui.number(
                    label="Floor",
                    value=float(get_val("conf_pull_floor", 0.0) or 0.0),
                    format="%.2f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "conf_pull_ceil",
                ui.number(
                    label="Ceil",
                    value=float(get_val("conf_pull_ceil", 1.0) or 1.0),
                    format="%.2f",
                ).props("dense outlined").classes("flex-1"),
            )
    conf_pull_col.set_visibility(False)

    ui.separator()

    # Expert / Raw DSP (ADVANCED only, in expansion)
    raw_dsp_col = ui.column().classes("w-full")
    ctrl.register_container("unsafe_raw_dsp_scope", raw_dsp_col)
    with ui.expansion("🔧 Expert / Raw DSP").classes("w-full"):
        ctrl.register(
            "unsafe_raw_dsp",
            ui.checkbox(
                "Enable raw DSP overrides",
                value=bool(get_val("unsafe_raw_dsp", False)),
            ),
        )
    raw_dsp_col.set_visibility(False)

    ui.separator()

    # Plot smoothing (visual only, in expansion)
    with ui.expansion(f"📈 {t('ui_plots_visual_only')}").classes("w-full"):
        ctrl.register(
            "plot_smoothing_level",
            ui.select(
                options={
                    "Psychoacoustic": t("smooth_safe_reference"),
                    12: "1/12 Octave",
                    24: "1/24 Octave",
                    48: "1/48 Octave",
                    96: "1/96 Octave",
                },
                value=get_val("plot_smoothing_level", "Psychoacoustic"),
                label=t("smooth_type"),
            ).props("dense outlined").classes("w-full"),
        )
