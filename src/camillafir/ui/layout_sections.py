import base64
import importlib.resources as pkgres

from pywebio.input import FLOAT, NUMBER
from pywebio.output import (
    put_button,
    put_buttons,
    put_collapse,
    put_grid,
    put_html,
    put_markdown,
    put_row,
    put_scope,
    put_tabs,
    put_text,
)
from pywebio.pin import put_checkbox, put_file_upload, put_input, put_radio, put_select

from .camillafir_housecurve import _normalize_hc_mode_key
from .camillafir_ui_helpers import put_guide_section


def _inject_dark_css():
    put_html(
        """
<style>
  /* ===== CamillaFIR Matte Dark — PyWebIO specific ===== */

  :root{
    color-scheme: dark;

    --cf-bg: #0b0f14;
    --cf-surface: rgba(255,255,255,0.035);
    --cf-surface-2: rgba(255,255,255,0.055);
    --cf-surface-3: rgba(255,255,255,0.075);
    --cf-border: rgba(255,255,255,0.10);
    --cf-border-2: rgba(255,255,255,0.16);

    --cf-text: rgba(255,255,255,0.92);
    --cf-muted: rgba(255,255,255,0.70);
    --cf-faint: rgba(255,255,255,0.52);

    --cf-accent: rgba(160,210,255,0.80);
    --cf-focus: rgba(160,210,255,0.22);

    --cf-radius: 14px;
    --cf-radius-sm: 10px;
    --cf-shadow: 0 8px 22px rgba(0,0,0,0.35);
  }

  /* ===== Root containers ===== */
  body.webio-theme-dark {
    background: var(--cf-bg) !important;
    color: var(--cf-text) !important;
  }

  .pywebio,
  #output-container,
  #input-container {
    background: var(--cf-bg) !important;
    color: var(--cf-text) !important;
  }

  /* ===== Markdown output ===== */
  .markdown-body {
    color: var(--cf-text) !important;
  }
  .markdown-body h1 { font-size: 28px; }
  .markdown-body h2 { font-size: 20px; margin-top: 18px; }
  .markdown-body h3 { font-size: 16px; opacity: 0.95; }
  .markdown-body p,
  .markdown-body li { color: var(--cf-text); }
  .markdown-body em,
  .markdown-body small { color: var(--cf-muted); }
  .markdown-body hr { border-color: rgba(255,255,255,0.08); }

  /* ===== Cards / collapses ===== */
  .card,
  .collapse,
  .panel,
  .well {
    background: var(--cf-surface) !important;
    border: 1px solid var(--cf-border) !important;
    border-radius: var(--cf-radius) !important;
    box-shadow: var(--cf-shadow) !important;
  }

  .card-header,
  .collapse > .title {
    color: var(--cf-text) !important;
    font-weight: 700;
  }

  /* ===== Tabs (Bootstrap via PyWebIO) ===== */
  .nav-tabs {
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
  }
  .nav-tabs .nav-link {
    color: var(--cf-muted) !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 14px !important;
  }
  .nav-tabs .nav-link.active {
    color: var(--cf-text) !important;
    background: var(--cf-surface-2) !important;
    border: 1px solid var(--cf-border) !important;
    border-bottom: 0 !important;
  }

  .tab-content {
    background: var(--cf-surface) !important;
    border: 1px solid var(--cf-border) !important;
    border-top: 0 !important;
    border-radius: 0 0 var(--cf-radius) var(--cf-radius) !important;
    padding: 14px 14px 6px !important;
  }

  /* ===== Inputs ===== */
  input,
  select,
  textarea {
    background: var(--cf-surface-2) !important;
    color: var(--cf-text) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
  }
  input:hover,
  select:hover,
  textarea:hover {
    background: var(--cf-surface-3) !important;
    border-color: rgba(255,255,255,0.18) !important;
  }
  input:focus,
  select:focus,
  textarea:focus {
    border-color: rgba(160,210,255,0.45) !important;
    box-shadow: 0 0 0 3px var(--cf-focus) !important;
  }

  label { color: var(--cf-text) !important; }
  .help-block,
  .form-text,
  .input-help {
    color: var(--cf-faint) !important;
    font-size: 13px;
  }

  /* ===== Native select dropdown (critical fix) ===== */
  select { color-scheme: dark; }
  select option,
  select optgroup {
    color: #0b0f14 !important;
    background: #ffffff !important;
  }
  select option:checked {
    background: #dbeafe !important;
    color: #0b0f14 !important;
  }

  /* ===== Buttons ===== */
  button,
  .btn,
  .pywebio-button {
    background: var(--cf-surface-2) !important;
    color: var(--cf-text) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
  }
  button:hover,
  .btn:hover,
  .pywebio-button:hover {
    background: var(--cf-surface-3) !important;
    border-color: rgba(255,255,255,0.22) !important;
  }

  /* ===== Tables ===== */
  table { border-collapse: separate !important; border-spacing: 0 !important; }
  th {
    background: var(--cf-surface-2) !important;
    color: var(--cf-text) !important;
    font-weight: 700;
  }
  td { color: var(--cf-text); }
  tr:hover td { background: rgba(255,255,255,0.03); }

  /* ===== Scrollbars (Chromium) ===== */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 999px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }
  ::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }

  /* ===== Alerts (PyWebIO put_success/put_info/put_warning/put_error) ===== */
  .alert{
    border-radius: var(--cf-radius) !important;
    border: 1px solid var(--cf-border) !important;
    color: var(--cf-text) !important;
    box-shadow: none !important;
  }
  .alert-success{
    background: rgba(34, 197, 94, 0.10) !important;
    border-color: rgba(34, 197, 94, 0.25) !important;
    color: rgba(235, 255, 245, 0.92) !important;
    font-weight: 700 !important;
  }
  .alert-info{
    background: rgba(59, 130, 246, 0.10) !important;
    border-color: rgba(59, 130, 246, 0.25) !important;
    color: rgba(235, 245, 255, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-warning{
    background: rgba(234, 179, 8, 0.12) !important;
    border-color: rgba(234, 179, 8, 0.28) !important;
    color: rgba(255, 250, 235, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-danger{
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: rgba(239, 68, 68, 0.28) !important;
    color: rgba(255, 235, 235, 0.92) !important;
    font-weight: 650 !important;
  }
</style>
"""
    )


def _put_logo_header(version: str):
    with pkgres.files("camillafir.ui.assets").joinpath("camillafir_logo.png").open("rb") as f:
        img_bytes = f.read()

    img_b64 = base64.b64encode(img_bytes).decode()

    put_html(
        f"""
<div style="
    display:flex;
    align-items:center;
    gap:18px;
    margin: 18px 0 18px 0;
">
    <img src="data:image/png;base64,{img_b64}"
         style="height:96px; width:auto;" />

    <div style="display:flex; flex-direction:column;">
        <div style="
            font-size:32px;
            font-weight:650;
            color:#ffffff;
            line-height:1.1;
        ">
            CamillaFIR
        </div>

        <div style="
            font-size:14px;
            color:#9aa4b2;
            margin-top:6px;
            letter-spacing:0.4px;
        ">
            {version}
        </div>
    </div>
</div>
"""
    )


def build_header(*, t, version: str):
    _inject_dark_css()
    _put_logo_header(version)
    put_html('<hr style="border:1px solid #1f2937; margin: 0 0 14px 0;">')
    put_collapse(
        t("about_title"),
        [put_markdown(t("about_body"))],
    )
    put_guide_section()
    put_markdown("---")


def build_input_section(*, t, get_val):
    return [
        put_markdown(f"### 📂 {t('tab_files')}"),
        put_markdown("---"),
        put_markdown(t("wav_recommended_info")),
        put_markdown("---"),
        put_markdown(f"### 🧾 {t('input_files_title')}"),
        put_html(
            f"<div style='opacity:0.75; font-size:13px; margin-top:-6px;'>"
            f"{t('input_files_help')}"
            f"</div>"
        ),
        put_file_upload("file_l", label=t("upload_l"), accept=".txt,.wav"),
        put_file_upload("file_r", label=t("upload_r"), accept=".txt,.wav"),
        put_collapse(
            t("ui_local_paths_optional"),
            [
                put_input("local_path_l", label=t("path_l"), value=get_val("local_path_l", ""), help_text=t("path_help")),
                put_input("local_path_r", label=t("path_r"), value=get_val("local_path_r", ""), help_text=t("path_help")),
            ],
            open=False,
        ),
        put_select("fmt", label=t("fmt"), options=["WAV", "TXT"], value=get_val("fmt", "WAV"), help_text=t("fmt_help")),
        put_radio(
            "layout",
            label=t("layout"),
            options=[t("layout_mono"), t("layout_stereo")],
            value=get_val("layout", t("layout_stereo")),
            inline=True,
        ),
        put_checkbox(
            "multi_rate_opt",
            options=[{"label": t("multi_rate"), "value": True}],
            value=[True] if get_val("multi_rate_opt", False) else [],
            help_text=t("multi_rate_help"),
        ),
        put_checkbox(
            "comparison_mode",
            options=[{"label": t("comparison_mode"), "value": True}],
            value=[True] if get_val("comparison_mode", True) else [],
            help_text=t("comparison_mode_help"),
        ),
        put_scope("taps_auto_info_scope_files"),
    ]


def build_filter_section(*, t, get_val, fs_opts, taps_opts, on_mode_apply_defaults):
    mode_value = str(get_val("mode", "AUTO") or "AUTO").strip().upper()
    if mode_value not in ("BASIC", "ADVANCED", "AUTO"):
        mode_value = "AUTO"
    if bool(get_val("camillafir_automatic_mode", False)):
        mode_value = "AUTO"
    auto_goal_value = str(get_val("auto_goal", "balanced") or "balanced").strip().lower()
    if auto_goal_value not in ("room-safe", "balanced", "low-ripple", "flat"):
        auto_goal_value = "balanced"

    return [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        put_markdown("---"),
        put_row(
            [
                put_select(
                    "mode",
                    label=t("mode_label"),
                    options=[
                        {"label": t("mode_auto_label"), "value": "AUTO"},
                        {"label": t("mode_basic_label"), "value": "BASIC"},
                        {"label": t("mode_advanced_label"), "value": "ADVANCED"},
                    ],
                    value=mode_value,
                    help_text=t("mode_help"),
                ),
                put_button(
                    t("mode_apply_defaults_btn"),
                    onclick=on_mode_apply_defaults,
                    color="secondary",
                ).style("margin-top:28px;"),
            ]
        ),
        put_markdown(f"_{t('mode_apply_defaults_help')}_"),
        put_scope("mode_desc_scope"),
        put_select(
            "auto_goal",
            label=t("auto_goal_label"),
            options=[
                {"label": t("auto_goal_balanced"), "value": "balanced"},
                {"label": t("auto_goal_room_safe"), "value": "room-safe"},
                {"label": t("auto_goal_low_ripple"), "value": "low-ripple"},
                {"label": t("auto_goal_flat"), "value": "flat"},
            ],
            value=auto_goal_value,
            help_text=t("auto_goal_help"),
        ),
        put_markdown("---"),
        put_markdown(f"#### 🧱 {t('ui_fir_engine')}"),
        put_row(
            [
                put_select("fs", label=t("fs"), options=fs_opts, value=get_val("fs", 44100), help_text=t("fs_help")),
                put_select("taps", label=t("taps"), options=taps_opts, value=get_val("taps", 65536), help_text=t("taps_help")),
            ]
        ),
        put_markdown(f"_{t('taps_warn_over')}_"),
        put_scope("engine_metrics_scope"),
        put_scope("taps_auto_info_scope_basic"),
        put_markdown("---"),
        put_markdown(f"#### 🧭 {t('ui_filter_type')}"),
        put_row(
            [
                put_select(
                    "filter_type",
                    label=t("filter_type"),
                    options=[t("ft_linear"), t("ft_min"), t("ft_mixed"), t("ft_asymmetric")],
                    value=get_val("filter_type", t("ft_linear")),
                    help_text=(t("ft_help")),
                ),
                put_scope("update_mixed_freq_scope"),
            ]
        ),
    ]


def build_target_section(*, t, get_val, hc_opts, max_safe_boost: float):
    return [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_markdown("---"),
        put_markdown(f"#### 👀 {t('ui_target_preview')}"),
        put_scope("target_preview_scope"),
        put_markdown("---"),
        put_select(
            "hc_mode",
            label=t("hc_mode"),
            options=hc_opts,
            value=_normalize_hc_mode_key(get_val("hc_mode", "Harman6")),
            help_text=t("hc_mode_help"),
        ),
        put_file_upload(
            "hc_custom_file",
            label=t("hc_custom"),
            accept=".txt",
            help_text=t("hc_custom_help"),
        ),
        put_markdown(f"#### 🎚 {t('ui_leveling_gain')}"),
        put_row(
            [
                put_select(
                    "lvl_algo",
                    label=t("lvl_algo"),
                    options=["Median", "Average"],
                    value=get_val("lvl_algo", "Median"),
                    help_text=t("lvl_algo_help"),
                ),
                put_input("gain", label=t("gain"), type=FLOAT, value=get_val("gain", 0.0), help_text=t("gain_help")),
            ]
        ),
        put_row(
            [
                put_input(
                    "lvl_min",
                    label=t("lvl_min"),
                    type=FLOAT,
                    value=get_val("lvl_min", 500.0),
                    help_text=t("lvl_min_help_auto"),
                ),
                put_input(
                    "lvl_max",
                    label=t("lvl_max"),
                    type=FLOAT,
                    value=get_val("lvl_max", 2000.0),
                    help_text=t("lvl_max_help_auto"),
                ),
            ]
        ),
        put_row(
            [
                put_select(
                    "lvl_mode",
                    label=t("lvl_mode"),
                    options=[
                        {"label": t("lvl_mode_auto"), "value": "Auto"},
                        {"label": t("lvl_mode_manual"), "value": "Manual"},
                    ],
                    value=get_val("lvl_mode", "Auto"),
                ),
                put_scope("lvl_manual_scope"),
            ]
        ),
        put_markdown("---"),
        put_checkbox("mag_correct", options=[{"label": t("enable_corr"), "value": True}], value=[True] if get_val("mag_correct", True) else []),
        put_html(
            f"""
            <div style="
                margin-top:28px;
                margin-bottom:12px;
                font-size:15px;
                font-weight:600;
                color:#b8c2cc;
                letter-spacing:0.5px;
            ">
                {t("magnitude_correction_limits")}
            </div>
            """
        ),
        put_row(
            [
                put_input(
                    "mag_c_min",
                    label=t("min_freq"),
                    type=FLOAT,
                    value=get_val("mag_c_min", 10.0),
                    help_text=t("hc_range_help"),
                ),
                put_input(
                    "mag_c_max",
                    label=t("max_freq"),
                    type=FLOAT,
                    value=get_val("mag_c_max", 200.0),
                    help_text=t("hc_range_help"),
                ),
            ]
        ),
        put_markdown("---"),
        put_input(
            "max_boost",
            label=t("max_boost"),
            type=FLOAT,
            value=get_val("max_boost", 5.0),
            help_text=t("max_boost_help"),
        ),
        put_html(
            f"<div style='color:#b00020; font-weight:600; font-size:13px; margin-top:4px;'>"
            f"⚠ {t('max_boost_help_cap').format(value=f'{max_safe_boost:.1f}')}"
            f"</div>"
        ),
    ]


def build_advanced_section(*, t, get_val, slope_opts, on_afdw_preset):
    return [
        put_markdown(f"### 🛠️ {t('tab_adv')}"),
        put_markdown("---"),
        put_collapse(
            f"📈 {t('ui_plots_visual_only')}",
            [
                put_select(
                    "plot_smoothing_level",
                    label=t("smooth_type"),
                    options=[
                        {"label": t("smooth_safe_reference"), "value": "Psychoacoustic"},
                        {"label": "1/12 Octave", "value": 12},
                        {"label": "1/24 Octave", "value": 24},
                        {"label": "1/48 Octave", "value": 48},
                        {"label": "1/96 Octave", "value": 96},
                    ],
                    value=get_val("plot_smoothing_level", "Psychoacoustic"),
                    help_text=t("smooth_help"),
                ),
            ],
            open=False,
        ),
        put_markdown("---"),
        put_scope("unsafe_raw_dsp_scope"),
        put_markdown("---"),
        put_markdown(f"#### 🧠 {t('bass_first_title')}"),
        put_checkbox(
            "bass_first_ai",
            options=[{"label": t("bass_first_enable_label"), "value": True}],
            value=[True] if get_val("bass_first_ai", False) else [],
            help_text=t("bass_first_enable_help"),
        ),
        put_input(
            "bass_first_mode_max_hz",
            label=t("bass_first_max_hz_label"),
            type=FLOAT,
            value=float(get_val("bass_first_mode_max_hz", 200.0) or 200.0),
            help_text=t("bass_first_max_hz_help"),
        ),
        put_markdown("---"),
        put_markdown(f"### 🧩 {t('ui_correction_shaping_rails')}"),
        put_row(
            [
                put_input(
                    "max_slope_db_per_oct",
                    label=t("max_slope_db_per_oct"),
                    type=FLOAT,
                    value=get_val("max_slope_db_per_oct", 12.0),
                    help_text=t("max_slope_db_per_oct_help"),
                ),
                put_input(
                    "max_cut_db",
                    label=t("max_cut_db"),
                    type=FLOAT,
                    value=get_val("max_cut_db", 30.0),
                    help_text=t("max_cut_db_help"),
                ),
            ]
        ),
        put_row(
            [
                put_input(
                    "max_slope_boost_db_per_oct",
                    label=t("max_slope_boost_db_per_oct"),
                    type=FLOAT,
                    value=get_val("max_slope_boost_db_per_oct", 0.0),
                    help_text=t("max_slope_boost_db_per_oct_help"),
                ),
                put_input(
                    "max_slope_cut_db_per_oct",
                    label=t("max_slope_cut_db_per_oct"),
                    type=FLOAT,
                    value=get_val("max_slope_cut_db_per_oct", 0.0),
                    help_text=t("max_slope_cut_db_per_oct_help"),
                ),
            ]
        ),
        put_markdown("---"),
        put_row(
            [
                put_input(
                    "trans_width",
                    type=NUMBER,
                    label="Transition Width (Hz)",
                    value=get_val("trans_width", 100),
                    help_text=t("trans_width"),
                ),
            ]
        ),
        put_markdown("---"),
        put_select(
            "filter_smooth",
            label=t("smoothing_level"),
            options=[
                {"label": "1/1 Octave", "value": 1},
                {"label": "1/3 Octave", "value": 3},
                {"label": "1/6 Octave", "value": 6},
                {"label": "1/12 Octave (Standard)", "value": 12},
                {"label": "1/24 Octave (Fine)", "value": 24},
                {"label": "1/48 Octave (Ultra)", "value": 48},
                {"label": "1/96 Octave (HC)", "value": 96},
            ],
            value=get_val("filter_smooth", get_val("smoothing_level", 12)),
            help_text=t("smoothing_level_help"),
        ),
        put_text(t("smoothing_level_saw")),
        put_input(
            "phase_limit",
            label=t("phase_limit"),
            type=FLOAT,
            value=get_val("phase_limit", 400.0),
            help_text=t("phase_limit_help"),
        ),
        put_markdown("---"),
        put_checkbox(
            "df_smoothing",
            options=[{"label": f"{t('df_smoothing_label')} {t('badge_experimental')}", "value": True}],
            value=[True] if get_val("df_smoothing", False) else [],
            help_text=t("df_smoothing_help"),
        ),
        put_markdown("---"),
        put_input("reg_strength", label=t("reg_strength"), type=FLOAT, value=get_val("reg_strength", 30.0), help_text=t("reg_help")),
        put_markdown("---"),
        put_checkbox(
            "stereo_link",
            options=[{"label": t("enable_link"), "value": True}],
            value=[True] if get_val("stereo_link", False) else [],
            help_text=t("link_help"),
        ),
        put_select(
            "stereo_link_strategy",
            label=t("stereo_link_mode"),
            options=[
                {"label": t("stereo_link_mode_auto"), "value": "auto"},
                {"label": t("stereo_link_mode_hybrid"), "value": "hybrid"},
                {"label": t("stereo_link_mode_shared_legacy"), "value": "shared"},
            ],
            value=str(get_val("stereo_link_strategy", "auto") or "auto"),
            help_text=t("stereo_link_mode_help"),
        ),
        put_markdown("### 🛡️ Bass Safety"),
        put_markdown("---"),
        put_row(
            [
                put_checkbox(
                    "exc_prot",
                    options=[{"label": t("exc_prot_title"), "value": True}],
                    value=[True] if get_val("exc_prot", False) else [],
                    help_text=t("exc_prot_help_ui"),
                ),
                put_input(
                    "exc_freq",
                    label=t("exc_freq"),
                    type=FLOAT,
                    value=get_val("exc_freq", 25.0),
                    help_text=t("exc_freq_help_ui"),
                ),
            ]
        ),
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('exc_prot_hint')}"
            f"</div>"
        ),
        put_collapse(
            t("guide_exc_prot_title"),
            [put_markdown(t("guide_exc_prot_body"))],
        ),
        put_html("<div style='height:12px'></div>"),
        put_markdown("---"),
        put_checkbox(
            "low_bass_cut_enable",
            options=[{"label": t("low_bass_cut_hz"), "value": True}],
            value=[True] if get_val("low_bass_cut_enable", True) else [],
        ),
        put_scope("low_bass_cut_scope"),
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('low_bass_cut_hint')}"
            f"</div>"
        ),
        put_collapse(
            t("guide_low_bass_cut_title"),
            [put_markdown(t("guide_low_bass_cut_body"))],
        ),
        put_markdown("---"),
        put_row(
            [
                put_checkbox("hpf_enable", options=[{"label": t("hpf_enable"), "value": True}], value=[True] if get_val("hpf_enable", False) else []),
                put_input("hpf_freq", label=t("hpf_freq"), type=FLOAT, value=get_val("hpf_freq", 20.0), help_text=t("hpf_freq_help")),
                put_select("hpf_slope", label=t("hpf_slope"), options=slope_opts, value=get_val("hpf_slope", 24)),
            ]
        ),
        put_markdown("---"),
        put_scope("conf_pull_scope"),
    ]


def build_export_section(*, t, get_val, on_afdw_preset):
    return [
        put_markdown(f"🪟 {t('tab_window_tdc')}"),
        put_markdown("---"),
        put_scope("ir_export_window_mode_scope"),
        put_row(
            [
                put_scope("ir_export_window_shape_scope"),
                put_scope("ir_tukey_alpha_scope"),
            ]
        ),
        put_collapse(
            t("ir_export_window_help_long_title"),
            [put_markdown(t("ir_export_window_help_long"))],
        ),
        put_scope("ir_lr_window_scope"),
        put_markdown("---"),
        put_checkbox(
            "enable_afdw",
            options=[{"label": "⏳ Adaptive Frequency-Domain Windowing (A-FDW)", "value": True}],
            value=[True] if get_val("enable_afdw", True) else [],
            help_text=t("afdw_help"),
        ),
        put_row(
            [
                put_buttons(
                    [
                        {"label": t("afdw_preset_tight"), "value": "Tight"},
                        {"label": t("afdw_preset_balanced"), "value": "Balanced"},
                        {"label": t("afdw_preset_safe"), "value": "Safe"},
                        {"label": t("afdw_preset_minimal"), "value": "Minimal"},
                    ],
                    onclick=on_afdw_preset,
                    small=True,
                ),
            ]
        ),
        put_html(f"<div style='opacity:0.65; font-size:13px'>{t('afdw_preset_help')}</div>"),
        put_row(
            [
                put_scope("afdw_cycles_scope"),
            ]
        ),
        put_markdown("---"),
        put_checkbox(
            "enable_tdc",
            options=[{"label": "⏳ Temporal Decay Control (TDC)", "value": True}],
            value=[True] if get_val("enable_tdc", True) else [],
            help_text=t("tdc_help"),
        ),
        put_scope("tdc_controls_scope"),
    ]


def build_results_section(*, t, get_val, slope_opts):
    return [
        put_markdown(f"### ❌ {t('tab_xo')}"),
        put_html(
            f"<div style='opacity:0.75; font-size:13px; margin-top:-6px;'>"
            f"{t('tab_xo_help')}"
            f"</div>"
        ),
        put_markdown("---"),
        put_grid(
            [
                [
                    put_input(
                        f"xo{i}_f",
                        label=f"XO {i} Hz",
                        type=FLOAT,
                        value=get_val(f"xo{i}_f", None),
                        help_text=t("xo_freq_help"),
                    ),
                    put_select(
                        f"xo{i}_s",
                        label="dB/oct",
                        options=slope_opts,
                        value=get_val(f"xo{i}_s", 12),
                        help_text=t("xo_slope_help"),
                    ),
                ]
                for i in range(1, 6)
            ]
        ),
    ]


def build_tabs(*, t, get_val, max_safe_boost: float, on_mode_apply_defaults, on_afdw_preset):
    hc_opts = [
        {"label": t("hc_harman"), "value": "Harman6"},
        {"label": t("hc_harman8"), "value": "Harman8"},
        {"label": t("hc_harman4"), "value": "Harman4"},
        {"label": t("hc_harman10"), "value": "Harman10"},
        {"label": t("hc_harman12"), "value": "Harman12"},
        {"label": t("hc_studio_tilt"), "value": "Studio"},
        {"label": t("hc_nearfield"), "value": "Nearfield"},
        {"label": t("hc_hifi_loudness"), "value": "HiFi"},
        {"label": t("hc_speech"), "value": "Speech"},
        {"label": t("hc_toole"), "value": "Toole"},
        {"label": t("hc_bk_light"), "value": "BK_Light"},
        {"label": t("hc_bk_medium"), "value": "BK_Medium"},
        {"label": t("hc_bk_strong"), "value": "BK_Strong"},
        {"label": t("hc_flat"), "value": "Flat"},
        {"label": t("hc_cinema"), "value": "Cinema"},
        {"label": t("hc_mode_upload"), "value": "Upload"},
    ]
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
    taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    slope_opts = [6, 12, 18, 24, 36, 48]

    tab_files = build_input_section(t=t, get_val=get_val)
    tab_basic = build_filter_section(t=t, get_val=get_val, fs_opts=fs_opts, taps_opts=taps_opts, on_mode_apply_defaults=on_mode_apply_defaults)
    tab_target = build_target_section(t=t, get_val=get_val, hc_opts=hc_opts, max_safe_boost=max_safe_boost)
    tab_adv = build_advanced_section(t=t, get_val=get_val, slope_opts=slope_opts, on_afdw_preset=on_afdw_preset)
    tab_window_tdc = build_export_section(t=t, get_val=get_val, on_afdw_preset=on_afdw_preset)
    tab_xo = build_results_section(t=t, get_val=get_val, slope_opts=slope_opts)

    put_tabs(
        [
            {"title": t("tab_files"), "content": tab_files},
            {"title": t("tab_basic"), "content": tab_basic},
            {"title": t("tab_target"), "content": tab_target},
            {"title": t("tab_adv"), "content": tab_adv},
            {"title": t("tab_window_tdc"), "content": tab_window_tdc},
            {"title": t("tab_xo"), "content": tab_xo},
        ]
    )
