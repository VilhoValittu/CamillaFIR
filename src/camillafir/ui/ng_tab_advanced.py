"""NiceGUI Advanced tab builder.

Replaces build_advanced_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl
from .ng_advanced_presets import (
    AGGRESSIVE,
    NORMAL,
    SAFE,
    apply_bass_safety_preset,
    apply_conf_pull_preset,
    apply_shaping_preset,
    render_bass_safety_summary,
    render_conf_pull_summary,
    render_shaping_summary,
)

_SLOPE_OPTS = [6, 12, 18, 24, 36, 48]
_FILTER_SMOOTH_OPTIONS = {
    1: "filter_smooth_1",
    3: "filter_smooth_3",
    6: "filter_smooth_6",
    12: "filter_smooth_12",
    24: "filter_smooth_24",
    48: "filter_smooth_48",
    96: "filter_smooth_96",
}


def build_advanced_tab(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    from nicegui import ui

    ui.markdown(f"#### {t('tab_adv')}")
    ui.separator()

    with ui.card().classes("w-full gap-3"):
        ui.label(t("adv_shaping_title")).classes("text-sm font-semibold")
        _build_preset_row(
            labels={
                SAFE: t("preset_safe"),
                NORMAL: t("preset_normal"),
                AGGRESSIVE: t("preset_aggressive"),
            },
            on_pick=lambda key: _apply_guided_preset(
                key,
                t=t,
                apply_fn=apply_shaping_preset,
                render_fn=lambda: render_shaping_summary(t=t),
            ),
        )
        ctrl.register_container("adv_shaping_summary_scope", ui.column().classes("cf-adv-summary w-full"))

        with ui.row().classes("w-full gap-4 items-end"):
            ctrl.register(
                "filter_smooth",
                ui.select(
                    options={value: t(label_key) for value, label_key in _FILTER_SMOOTH_OPTIONS.items()},
                    value=get_val("filter_smooth", get_val("smoothing_level", 12)),
                    label=t("smoothing_level"),
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "phase_limit",
                ui.number(
                    label=t("phase_limit"),
                    value=float(get_val("phase_limit", 400.0) or 400.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.expansion(t("adv_shaping_fine_tune_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
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
                            label=t("trans_width_label"),
                            value=int(get_val("trans_width", 100) or 100),
                            format="%d",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "reg_strength",
                        ui.number(
                            label=t("reg_strength"),
                            value=float(get_val("reg_strength", 30.0) or 30.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )

                ctrl.register(
                    "df_smoothing",
                    ui.checkbox(
                        f"{t('df_smoothing_label')} [EXPERIMENTAL]",
                        value=bool(get_val("df_smoothing", False)),
                    ),
                )

    ui.separator()

    with ui.card().classes("w-full gap-3"):
        ui.label(t("adv_bass_safety_title")).classes("text-sm font-semibold")
        _build_preset_row(
            labels={
                SAFE: t("preset_safe"),
                NORMAL: t("preset_normal"),
                AGGRESSIVE: t("preset_aggressive"),
            },
            on_pick=lambda key: _apply_guided_preset(
                key,
                t=t,
                apply_fn=apply_bass_safety_preset,
                render_fn=lambda: render_bass_safety_summary(t=t),
                extra_refreshers=(_refresh_bass_safety_ui,),
            ),
        )
        ctrl.register_container("adv_bass_safety_summary_scope", ui.column().classes("cf-adv-summary w-full"))

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

        with ui.expansion(t("adv_bass_safety_fine_tune_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
                ctrl.register(
                    "low_bass_cut_enable",
                    ui.checkbox(
                        t("low_bass_cut_hz"),
                        value=bool(get_val("low_bass_cut_enable", True)),
                    ),
                )
                bass_cut_col = ui.column().classes("w-full gap-3")
                ctrl.register_container("low_bass_cut_scope", bass_cut_col)
                with bass_cut_col:
                    with ui.row().classes("w-full gap-4"):
                        ctrl.register(
                            "low_bass_cut_hz",
                            ui.number(
                                label=t("low_bass_cut_hz"),
                                value=float(get_val("low_bass_cut_hz", 30.0) or 30.0),
                                format="%.1f",
                            ).props("dense outlined").classes("flex-1"),
                        )
                        ctrl.register(
                            "low_bass_cut_strength",
                            ui.number(
                                label=t("low_bass_cut_strength_label"),
                                value=float(get_val("low_bass_cut_strength", 1.0) or 1.0),
                                format="%.2f",
                            ).props("dense outlined").classes("flex-1"),
                        )
                bass_cut_col.set_visibility(bool(get_val("low_bass_cut_enable", True)))

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
                            label=t("bass_first_max_hz_label"),
                            value=float(get_val("bass_first_mode_max_hz", 200.0) or 200.0),
                            format="%.1f",
                        ).props("dense outlined").classes("w-full"),
                    )
                bass_first_col.set_visibility(bool(get_val("bass_first_ai", False)))

        with ui.expansion(t("guide_exc_prot_title")).classes("w-full"):
            ui.markdown(t("guide_exc_prot_body"))

        with ui.expansion(t("guide_low_bass_cut_title")).classes("w-full"):
            ui.markdown(t("guide_low_bass_cut_body"))

    ui.separator()

    conf_pull_col = ui.column().classes("w-full")
    ctrl.register_container("conf_pull_scope", conf_pull_col)
    with conf_pull_col:
        with ui.card().classes("w-full gap-3"):
            ui.label(t("adv_conf_pull_title")).classes("text-sm font-semibold")
            _build_preset_row(
                labels={
                    SAFE: t("preset_safe"),
                    NORMAL: t("preset_normal"),
                    AGGRESSIVE: t("preset_aggressive"),
                },
                on_pick=lambda key: _apply_guided_preset(
                    key,
                    t=t,
                    apply_fn=apply_conf_pull_preset,
                    render_fn=lambda: render_conf_pull_summary(t=t),
                ),
            )
            ctrl.register_container("adv_conf_pull_summary_scope", ui.column().classes("cf-adv-summary w-full"))

            with ui.expansion(t("adv_conf_pull_tuning_title")).classes("w-full"):
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "conf_pull_floor",
                        ui.number(
                            label=t("conf_pull_floor_label"),
                            value=float(get_val("conf_pull_floor", 0.0) or 0.0),
                            format="%.2f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "conf_pull_ceil",
                        ui.number(
                            label=t("conf_pull_ceil_label"),
                            value=float(get_val("conf_pull_ceil", 1.0) or 1.0),
                            format="%.2f",
                        ).props("dense outlined").classes("flex-1"),
                    )
    conf_pull_col.set_visibility(False)

    ui.separator()

    ui.label(t("enable_link")).classes("text-sm font-semibold")
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
                "auto": t("stereo_link_mode_auto"),
                "hybrid": t("stereo_link_mode_hybrid"),
                "shared": t("stereo_link_mode_shared_legacy"),
            },
            value=str(get_val("stereo_link_strategy", "auto") or "auto"),
            label=t("stereo_link_mode"),
        ).props("dense outlined").classes("w-full"),
    )

    ui.separator()

    raw_dsp_col = ui.column().classes("w-full")
    ctrl.register_container("unsafe_raw_dsp_scope", raw_dsp_col)
    with ui.expansion(t("ui_expert_raw_dsp_title")).classes("w-full"):
        ctrl.register(
            "unsafe_raw_dsp",
            ui.checkbox(
                t("unsafe_raw_dsp_enable_label"),
                value=bool(get_val("unsafe_raw_dsp", False)),
            ),
        )
    raw_dsp_col.set_visibility(False)

    ui.separator()

    with ui.expansion(t("ui_plots_visual_only")).classes("w-full"):
        ctrl.register(
            "plot_smoothing_level",
            ui.select(
                options={
                    "Psychoacoustic": t("smooth_safe_reference"),
                    12: t("filter_smooth_12"),
                    24: t("filter_smooth_24"),
                    48: t("filter_smooth_48"),
                    96: t("filter_smooth_96"),
                },
                value=get_val("plot_smoothing_level", "Psychoacoustic"),
                label=t("smooth_type"),
            ).props("dense outlined").classes("w-full"),
        )


def _build_preset_row(*, labels: dict[str, str], on_pick: Callable[[str], None]) -> None:
    from nicegui import ui

    with ui.row().classes("cf-adv-preset-row"):
        for key in (SAFE, NORMAL, AGGRESSIVE):
            ui.button(
                labels[key],
                on_click=lambda preset_key=key: on_pick(preset_key),
            ).props('size="sm" outline no-caps')


def _apply_guided_preset(
    preset_key: str,
    *,
    t: Callable,
    apply_fn: Callable[..., None],
    render_fn: Callable[[], None],
    extra_refreshers: tuple[Callable[[], None], ...] = (),
) -> None:
    if str(ctrl.value("mode", "BASIC") or "BASIC").upper() == "AUTO":
        from .ng_health import show_toast  # noqa: PLC0415

        show_toast(t("toast_adv_preset_locked_auto"), color="info", duration=1.8)
        return

    apply_fn(preset_key, t=t)
    for refresher in extra_refreshers:
        refresher()
    render_fn()


def _refresh_bass_safety_ui() -> None:
    from .ng_mode_controls import update_bass_first_ui, update_low_bass_cut_ui  # noqa: PLC0415

    update_low_bass_cut_ui()
    update_bass_first_ui()
