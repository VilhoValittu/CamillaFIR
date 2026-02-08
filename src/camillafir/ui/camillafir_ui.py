import json
import logging
import math
import typing
import os
import re
import sys
from datetime import datetime
from textwrap import dedent

import numpy as np
import scipy.io.wavfile

from pywebio import config
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.session import set_env
import pywebio.output as pwo
from pywebio.output import put_html

from ..config.camillafir_config import load_config, save_config
from ..resources.i8n.camillafir_i18n import t
from .camillafir_housecurve import _normalize_hc_mode_key, get_house_curve_by_name, load_target_curve, load_house_curve
from camillafir.io.measurements_loader import load_measurements_lr
from camillafir.io.measurements_txt import parse_measurements_from_path
from .camillafir_ui_helpers import (
    update_mode_desc,
    apply_mode_defaults_to_ui,
    update_taps_auto_info,
    update_lvl_ui,
    apply_tdc_preset,
    apply_afdw_preset,
    put_guide_section,
    update_ir_tukey_ui,
    update_ir_export_window_mode_ui,
    update_mixed_freq_ui,
    update_ir_window_shape_ui,
    update_ir_lr_window_ui,
)
from ..config.camillafir_pipeline import (
    collect_ui_data,
    log_df_smoothing_toggle,
    build_xos_hpf,
    filter_type_short,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
    build_filter_config,
)
from ..dsp import camillafir_dsp as dsp
from . import camillafir_plot as plots
import camillafir.config.models as models
from camillafir.config.models import FilterConfig
from .camillafir_modes import apply_mode_to_cfg, MODE_DEFAULTS
from .camillafir_utils import scale_taps_with_fs
from pywebio.pin import pin_update
from .camillafir_ui_helpers import (
    _max_boost_help_with_cap,
    _toast,
    _warn_taps_if_over_cap,
    _warn_max_boost_if_over_cap,
)


logger = logging.getLogger("CamillaFIR")

def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    """
    Adapter for camillafir.py:
    - Injects required globals into this module
    - Returns the PyWebIO 'main' callable
    """
    g = globals()
    g["process_run"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    g["update_status"] = update_status
    return main




def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50; margin-bottom: 10px;')

def _inject_dark_css():
    put_html("""
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
    background: rgba(34, 197, 94, 0.10) !important;   /* matte green */
    border-color: rgba(34, 197, 94, 0.25) !important;
    color: rgba(235, 255, 245, 0.92) !important;
    font-weight: 700 !important;
  }
  .alert-info{
    background: rgba(59, 130, 246, 0.10) !important;  /* matte blue */
    border-color: rgba(59, 130, 246, 0.25) !important;
    color: rgba(235, 245, 255, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-warning{
    background: rgba(234, 179, 8, 0.12) !important;   /* matte amber */
    border-color: rgba(234, 179, 8, 0.28) !important;
    color: rgba(255, 250, 235, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-danger{
    background: rgba(239, 68, 68, 0.12) !important;   /* matte red */
    border-color: rgba(239, 68, 68, 0.28) !important;
    color: rgba(255, 235, 235, 0.92) !important;
    font-weight: 650 !important;
  }
</style>
""")


@config(theme="dark")
def main():
    set_env(output_max_width='1850px') 
    _inject_dark_css()
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    put_guide_section(); put_markdown("---")
    d = load_config(); get_val = lambda k, def_v: d.get(k, def_v)
    hc_opts = [
        {'label': t('hc_harman'),        'value': 'Harman6'},   # default
        {'label': t('hc_harman8'),       'value': 'Harman8'},
        {'label': t('hc_harman4'),       'value': 'Harman4'},
        {'label': t('hc_harman10'),      'value': 'Harman10'},
        {'label': t('hc_studio_tilt'),   'value': 'Studio'},
        {'label': t('hc_nearfield'),     'value': 'Nearfield'},
        {'label': t('hc_hifi_loudness'), 'value': 'HiFi'},
        {'label': t('hc_speech'),        'value': 'Speech'},
        {'label': t('hc_toole'),         'value': 'Toole'},
        {'label': t('hc_bk'),            'value': 'BK'},
        {'label': t('hc_flat'),          'value': 'Flat'},
        {'label': t('hc_cinema'),        'value': 'Cinema'},
        {'label': t('hc_mode_upload'),   'value': 'Custom'},
    ]
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]; 
    taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; slope_opts = [6, 12, 18, 24, 36, 48]
    
#--- #1 Files
    
    tab_files = [
        put_markdown(f"### 📂 {t('tab_files')}"),
        put_markdown("---"),
        put_markdown(t('wav_recommended_info')),
        put_markdown("---"),
        put_markdown(f"### 🧾 {t('input_files_title')}"),
        put_html(
            f"<div style='opacity:0.75; font-size:13px; margin-top:-6px;'>"
            f"{t('input_files_help')}"
            f"</div>"
        ),
        put_file_upload('file_l', label=t('upload_l'), accept='.txt,.wav'), 
        put_input('local_path_l', label=t('path_l'), value=get_val('local_path_l', ''), help_text=t('path_help')),
        put_file_upload('file_r', label=t('upload_r'), accept='.txt,.wav'), 
        put_input('local_path_r', label=t('path_r'), value=get_val('local_path_r', ''), help_text=t('path_help')),
        put_select('fmt', label=t('fmt'), options=['WAV', 'TXT'], value=get_val('fmt', 'WAV'), help_text=t('fmt_help')),
        put_radio('layout', label=t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=get_val('layout', t('layout_stereo')), inline=True),
        put_checkbox('multi_rate_opt', options=[{'label': t('multi_rate'), 'value': True}], value=[True] if get_val('multi_rate_opt', False) else [], help_text=t('multi_rate_help')),
        put_checkbox('comparison_mode',
                     options=[{'label': t('comparison_mode'), 'value': True}],
                     value=[True] if get_val('comparison_mode', True) else [],
                     help_text=t('comparison_mode_help')),
        put_scope('taps_auto_info_scope_files'),
    ]
    
#--- #2 Basic Settings

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        put_markdown(t('---')),
        # Mode + Apply defaults (manual apply is safer UX)
        put_row([
            put_select(
                'mode',
                label=t('mode_label'),
                options=[
                    {'label': t('mode_basic_label'), 'value': 'BASIC'},
                    {'label': t('mode_advanced_label'), 'value': 'ADVANCED'},
                ],
                value=str(get_val('mode', 'BASIC') or 'BASIC').strip().upper(),
                help_text=t('mode_help'),
            ),
            put_button(
                t('mode_apply_defaults_btn'),
                onclick=apply_mode_defaults_to_ui,
                color='secondary'
            ).style("margin-top:28px;"),
        ]),
        put_markdown(f"_{t('mode_apply_defaults_help')}_"),
        put_scope('mode_desc_scope'),

        put_markdown("---"),

        # Row 1: Sample rate and Taps
        put_row([
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')),  # type: ignore
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),
        put_markdown(f"_{t('taps_warn_over')}_"),

        # Auto-taps info (shown only when multi_rate_opt enabled)
        put_scope('taps_auto_info_scope_basic'),
        
        # Row 2: Filter type and Mixed frequency
        put_row([
            put_select(
                'filter_type',
                label=t('filter_type'),
                options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), t('ft_asymmetric')],
                value=get_val('filter_type', t('ft_linear')),
                help_text=(t('ft_help') + t('ft_asym_note'))
            ),
            put_scope("update_mixed_freq_scope"),
        ]),
        
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        
        put_select('lvl_algo', label=t('lvl_algo'), options=['Median', 'Average'], value=get_val('lvl_algo', 'Median'), help_text=t('lvl_algo_help')),
        put_select(
            'plot_smoothing_level',
            label=t('smooth_type'),
            options=[
                {'label': 'Psychoacoustic', 'value': 'Psychoacoustic'},
                {'label': '1/12 Octave', 'value': 12},
                {'label': '1/24 Octave', 'value': 24},
                {'label': '1/48 Octave', 'value': 48},
                {'label': '1/96 Octave', 'value': 96},
            ],
            value=get_val('plot_smoothing_level', 'Psychoacoustic'),
            help_text=t('smooth_help')
        ),
        # Row 3: Mode selection and target level (split into two parts for readability)
        # Level match range (help_text goes to the right place directly under fields)
        put_row([
            put_input(
                'lvl_min',
                label=t('lvl_min'),
                type=FLOAT,
                value=get_val('lvl_min', 500.0),
                help_text=t('lvl_min_help_auto')  # default Auto mode
            ),
            put_input(
                'lvl_max',
                label=t('lvl_max'),
                type=FLOAT,
                value=get_val('lvl_max', 2000.0),
                help_text=t('lvl_max_help_auto')  # default Auto mode
            ),
        ]),

        # lvl_mode + lvl_manual_db (shown always, but locked in Auto mode)
        put_row([
            put_select(
                'lvl_mode',
                label=t('lvl_mode'),
                options=[
                    {'label': t('lvl_mode_auto'), 'value': 'Auto'},
                    {'label': t('lvl_mode_manual'), 'value': 'Manual'},
                ],
                value=get_val('lvl_mode', 'Auto')
            ),
            put_scope('lvl_manual_scope'),
        ]),


        
            ]
#--- #3 Target
    
    tab_target = [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_markdown(t('---')),
        put_select(
            'hc_mode',
            label=t('hc_mode'),
            options=hc_opts,
            value=_normalize_hc_mode_key(get_val('hc_mode', 'Harman6')),
            help_text=t('hc_mode_help')
        ),
        
        
        put_file_upload('hc_custom_file', label=t('hc_custom'), accept='.txt', help_text=t('hc_custom_help')),
        put_markdown("---"),
        put_checkbox('mag_correct', options=[{'label': t('enable_corr'), 'value': True}], value=[True] if get_val('mag_correct', True) else []),
        put_markdown("---"),
        put_row([
            put_input('mag_c_min', label=t('min_freq'), type=FLOAT, value=get_val('mag_c_min', 10.0), help_text=t('hc_range_help')), 
            put_input('mag_c_max', label=t('max_freq'), type=FLOAT, value=get_val('mag_c_max', 200.0), help_text=t('hc_range_help'))
        ]),
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=_max_boost_help_with_cap()),
        put_row([
            put_input('max_cut_db', label=t('max_cut_db'), type=FLOAT, value=get_val('max_cut_db', 30.0),
                      help_text=t('max_cut_db_help')),
            put_input('max_slope_db_per_oct', label=t('max_slope_db_per_oct'), type=FLOAT, value=get_val('max_slope_db_per_oct', 12.0),
                      help_text=t('max_slope_db_per_oct_help'))
        ]),
        put_row([
            put_input('max_slope_boost_db_per_oct', label=t('max_slope_boost_db_per_oct'), type=FLOAT,
                      value=get_val('max_slope_boost_db_per_oct', 0.0),
                      help_text=t('max_slope_boost_db_per_oct_help')),
            put_input('max_slope_cut_db_per_oct', label=t('max_slope_cut_db_per_oct'), type=FLOAT,
                      value=get_val('max_slope_cut_db_per_oct', 0.0),
                      help_text=t('max_slope_cut_db_per_oct_help'))
         ]),
        
        put_input('trans_width', type=NUMBER, label="1/1 Transition Width (Hz)", value=100, help_text=t('trans_width')),
        put_markdown("---"),
        put_select(
                    'filter_smooth',
                    label=t('smoothing_level'),
                    options=[
                        {'label': '1/1 Octave', 'value': 1},
                        {'label': '1/3 Octave', 'value': 3},
                        {'label': '1/6 Octave', 'value': 6},
                        {'label': '1/12 Octave (Standard)', 'value': 12},
                        {'label': '1/24 Octave (Fine)', 'value': 24},
                        {'label': '1/48 Octave (Ultra)', 'value': 48},
                        {'label': '1/96 Octave (HC)', 'value': 96},
                    ],
                    value=get_val('filter_smooth', get_val('smoothing_level', 12)),
                    help_text=t('smoothing_level_help'),
                ),
        put_text(t('smoothing_level_saw')),
        
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 1000.0), help_text=t('phase_limit_help')),
        
        put_checkbox(
            'phase_safe_2058',
            options=[{'label': t('phase_safe_2058'), 'value': True}],
            value=[True] if get_val('phase_safe_2058', False) else [],
            help_text=t('phase_safe_2058_help')
        ),
        
    ]
#--- #4 Advanced
    tab_adv = [
        put_markdown(f"### 🛠️ {t('tab_adv')}"),

            put_markdown("---"),
            put_markdown(f"#### 🧠 {t('bass_first_title')}"),
            put_checkbox(
                'bass_first_ai',
                options=[{'label': t('bass_first_enable_label'), 'value': True}],
                value=[True] if get_val('bass_first_ai', False) else [],
                help_text=t('bass_first_enable_help')
            ),
            put_input(
                'bass_first_mode_max_hz',
                label=t('bass_first_max_hz_label'),
                type=FLOAT,
                value=float(get_val('bass_first_mode_max_hz', 200.0) or 200.0),
                help_text=t('bass_first_max_hz_help')
            ),
            
put_markdown("---"),

        put_checkbox('df_smoothing', options=[{'label': f"{t('df_smoothing_label')} {t('badge_experimental')}", 'value': True}],
             value=[True] if get_val('df_smoothing', False) else [],
             help_text=t('df_smoothing_help')),
        put_markdown("---"),
        
        put_input('reg_strength', label=t('reg_strength'), type=FLOAT, value=get_val('reg_strength', 30.0), help_text=t('reg_help')),
        put_markdown("---"),
        
        

        put_row([
            put_checkbox('normalize_opt', options=[{'label': t('enable_norm'), 'value': True}], value=[True] if get_val('normalize_opt', True) else [], help_text=t('norm_help')), 
            put_checkbox('align_opt', options=[{'label': t('enable_align'), 'value': True}], value=[True] if get_val('align_opt', True) else [], help_text=t('align_help')), 
            put_checkbox('stereo_link', options=[{'label': t('enable_link'), 'value': True}], value=[True] if get_val('stereo_link', False) else [], help_text=t('link_help'))
        ]),
        
        # --- Bass Safety (Advanced tab) ---
put_markdown("### 🛡️ Bass Safety"),
put_markdown("---"),

        # 1) Excursion Protection (Driver Safety)
        put_row([
            put_checkbox(
                'exc_prot',
                options=[{'label': t('exc_prot_title'), 'value': True}],
                value=[True] if get_val('exc_prot', False) else [],
                help_text=t('exc_prot_help_ui')
            ),
            put_input(
                'exc_freq',
                label=t('exc_freq'),
                type=FLOAT,
                value=get_val('exc_freq', 25.0),
                help_text=t('exc_freq_help_ui')
            ),
        ]),

        # micro-hint (small, grey)
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('exc_prot_hint')}"
            f"</div>"
        ),

        # guide (collapsible)
        put_collapse(
            t('guide_exc_prot_title'),
            [put_markdown(t('guide_exc_prot_body'))]
        ),

        # spacing between the two tools
        put_html("<div style='height:12px'></div>"),

        # 2) Low-bass boost lock (policy limiter)
        put_input(
            'low_bass_cut_hz',
            label=t('low_bass_cut_hz'),
            type=FLOAT,
            value=get_val('low_bass_cut_hz', 40.0),
            help_text=t('low_bass_cut_hz_help')
        ),

        # micro-hint (small, grey)
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('low_bass_cut_hint')}"
            f"</div>"
        ),

        # guide (collapsible)
        put_collapse(
            t('guide_low_bass_cut_title'),
            [put_markdown(t('guide_low_bass_cut_body'))]
        ),

        
        put_markdown("---"),
        put_row([
            put_checkbox('hpf_enable', options=[{'label': t('hpf_enable'), 'value': True}], value=[True] if get_val('hpf_enable', False) else []), 
            put_input('hpf_freq', label=t('hpf_freq'), type=FLOAT, value=get_val('hpf_freq', 20.0), help_text=t('hpf_freq_help')), 
            put_select('hpf_slope', label=t('hpf_slope'), options=slope_opts, value=get_val('hpf_slope', 24)) # type: ignore
        ])
    ]

#--- #5 Window & TDC
    tab_window_tdc = [
        put_markdown(f"🪟 {t('tab_window_tdc')}"),
        put_markdown("---"),
        put_scope("ir_export_window_mode_scope"),


        put_row([
        put_scope("ir_export_window_shape_scope"),
        put_scope("ir_tukey_alpha_scope"),
        ]),

        put_collapse(
            t('ir_export_window_help_long_title'),
            [put_markdown(t('ir_export_window_help_long'))]
        ),
        
        
        put_scope("ir_lr_window_scope"),
        
        put_markdown("---"),

        # A-FDW
        put_markdown("#### ⏳ Adaptive Frequency-Domain Windowing (A-FDW)"),
        put_checkbox('enable_afdw', options=[{'label': t('enable_afdw'), 'value': True}], 
             value=[True] if get_val('enable_afdw', True) else [], help_text=t('afdw_help')),
        
        put_row([
            put_buttons(
                [
                    {"label": t("afdw_preset_tight"),    "value": "Tight"},
                    {"label": t("afdw_preset_balanced"), "value": "Balanced"},
                    {"label": t("afdw_preset_safe"),     "value": "Safe"},
                    {"label": t("afdw_preset_minimal"),  "value": "Minimal"},
                ],
                onclick=lambda preset: apply_afdw_preset(preset),
                small=True,
            ),
        ]),
        put_html(f"<div style='opacity:0.65; font-size:13px'>{t('afdw_preset_help')}</div>"),

        put_row([
            put_input('fdw_cycles', label=t('fdw'), type=FLOAT, value=get_val('fdw_cycles', 8.0), help_text=t('fdw_help'))
        ]),
        put_markdown("---"),
        # --- TDC aka Trinnov-mode (PyWebIO)

        put_markdown("#### ⏳ Temporal Decay Control (TDC)"),
        put_row([
            put_buttons(
                [
                    {"label": t("tdc_preset_safe"), "value": "Safe"},
                    {"label": t("tdc_preset_normal"), "value": "Normal"},
                    {"label": t("tdc_preset_aggressive"), "value": "Aggressive"},
                ],
                onclick=lambda preset: apply_tdc_preset(preset),
                small=True,
            ),
        ]),
        put_html(f"<div style='opacity:0.65; font-size:13px'>{t('tdc_preset_help')}</div>"),
        put_html(f"<div style='opacity:0.70; font-size:13px; margin-top:6px'>{t('tdc_summary_hint')}</div>"),


        put_checkbox(
            'enable_tdc',
            options=[{'label': t('enable_tdc'), 'value': True}],
            value=[True] if get_val('enable_tdc', True) else [],
            help_text=t('tdc_help')
        ),

        put_row([
            put_input(
                'tdc_strength',
                label=t('tdc_strength'),
                type=FLOAT,
                value=get_val('tdc_strength', 50.0),
                help_text=t('tdc_help')
            ),
            put_input(
                'tdc_max_reduction_db',
                label=t('tdc_max_reduction_db'),
                type=FLOAT,
                value=get_val('tdc_max_reduction_db', 9.0),
                help_text=t('tdc_max_reduction_db_help')
            ),
            put_input(
                'tdc_slope_db_per_oct',
                label=t('tdc_slope_db_per_oct'),
                type=FLOAT,
                value=get_val('tdc_slope_db_per_oct', 6.0),
                help_text=t('tdc_slope_db_per_oct_help')
            ),
        ]),
    ]
#--- #6 XO
    tab_xo = [
        put_markdown(f"### ❌ {t('tab_xo')}"),
        put_html(
            f"<div style='opacity:0.75; font-size:13px; margin-top:-6px;'>"
            f"{t('tab_xo_help')}"
            f"</div>"
        ),
        put_markdown("---"),
        put_grid([[
            put_input(
                f'xo{i}_f',
                label=f"XO {i} Hz",
                type=FLOAT,
                value=get_val(f'xo{i}_f', None),
                help_text=t('xo_freq_help')
            ),
            put_select(
                f'xo{i}_s',
                label="dB/oct",
                options=slope_opts,
                value=get_val(f'xo{i}_s', 12),
                help_text=t('xo_slope_help')
            )
        ] for i in range(1, 6)]) # type: ignore
    ]


    # Draw tabs
    put_tabs([
        {'title': t('tab_files'), 'content': tab_files}, 
        {'title': t('tab_basic'), 'content': tab_basic}, 
        {'title': t('tab_target'), 'content': tab_target}, 
        {'title': t('tab_adv'), 'content': tab_adv}, 
        {'title': t('tab_window_tdc'), 'content': tab_window_tdc},
        {'title': t('tab_xo'), 'content': tab_xo},
    ])

    # Only sanitize range when BOTH ends are valid numbers.
    # Do NOT call update_lvl_ui() here (it rerenders the scope and can reset inputs).
    def _on_lvl_range_change(_=None):
        try:
            a = pin.get('lvl_min', None)
            b = pin.get('lvl_max', None)
            if a is None or b is None:
                return
            # If user is mid-edit, values can be '' -> ignore until valid
            a = float(a)
            b = float(b)
            if not np.isfinite(a) or not np.isfinite(b):
                return
            if a > b:
                pin_update('lvl_min', value=b)
                pin_update('lvl_max', value=a)
        except Exception:
            return

    # update ui (initial render)
    update_lvl_ui()
    update_ir_tukey_ui()
    update_ir_export_window_mode_ui()
    update_mixed_freq_ui()
    update_ir_window_shape_ui()
    update_ir_lr_window_ui()

    # --- orchestrators (avoid racing handlers) ---

    def _refresh_ir_window_controls(_=None):
    # --- HARD POLICY: BASIC mode => windowing ALWAYS auto ---
        try:
            m = str(_pin_get('mode', 'BASIC') or 'BASIC').strip().upper()
            if m == "BASIC":
                if str(pin.get('ir_export_window_mode', '') or '').lower() != 'auto':
                    pin_update('ir_export_window_mode', value='auto')
        except Exception:
            pass

        # 1) sanitize/export-mode first (may force value back to "auto")
        update_ir_export_window_mode_ui()

        # 2) dependent scopes
        update_ir_window_shape_ui()
        update_ir_tukey_ui()
        update_ir_lr_window_ui()


    def _on_filter_type_change(_=None):
        # Mixed freq depends on filter_type
        update_mixed_freq_ui()
        # Filter type may indirectly force mode to auto -> refresh whole IR window group
        _refresh_ir_window_controls()

    # pins
    pin_on_change('ir_export_window_mode', onchange=_refresh_ir_window_controls)
    pin_on_change('filter_type', onchange=_on_filter_type_change)

    # keep these (independent)
    pin_on_change('lvl_mode', onchange=update_lvl_ui)

    # Tukey alpha depends on shape, but our refresh covers mode/filter changes
    pin_on_change('ir_export_window_shape', onchange=update_ir_tukey_ui)

    # Range change: sanitize only, no rerender
    pin_on_change('lvl_min', onchange=_on_lvl_range_change)
    pin_on_change('lvl_max', onchange=_on_lvl_range_change)
    pin_on_change('lvl_manual_db', onchange=_on_lvl_range_change)

    pin_on_change('taps', onchange=_warn_taps_if_over_cap)
    _warn_taps_if_over_cap()

    # Mode description: initial render + live updates
    def _on_mode_change(_=None):
        update_mode_desc()
        try:
            m = str(_pin_get('mode', 'BASIC') or 'BASIC').strip().upper()
            v = (MODE_DEFAULTS.get(m, {}) or {}).get('ir_export_window_mode', None)
            if isinstance(v, str) and v.strip():
                pin_update('ir_export_window_mode', value=v.strip())
                _refresh_ir_window_controls()
        except Exception:
            pass


    pin_on_change('mode', onchange=_on_mode_change)
    _on_mode_change()

    # Auto-taps UI updater: react when multi-rate toggles (tab_files) or basic changes
    pin_on_change('multi_rate_opt', onchange=update_taps_auto_info)
    pin_on_change('fs', onchange=update_taps_auto_info)
    pin_on_change('taps', onchange=update_taps_auto_info)
    update_taps_auto_info()

    pin_on_change('max_boost', onchange=_warn_max_boost_if_over_cap)
    _warn_max_boost_if_over_cap()

    put_markdown("---")

    
    # Button update: Completely clean text without background or border
    put_button("🚀 START", onclick=process_run).style("""
        width: 100%; 
        margin-top: 30px; 
        padding: 15px; 
        font-size: 24px; 
        font-weight: 900; 
        letter-spacing: 3px;
        
        background-color: transparent;  /* Ei taustaväriä */
        border: none;                  /* Poistaa kehykset kokonaan */
        color: #ffffff;                /* Teksti on puhdas valkoinen */
        
        transition: 0.3s;
        cursor: pointer;
    """)
def _log_df_smoothing_for_fs(cfg, fs_v, df_on):
    if df_on:
        try:
            fsmooth = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 12)) or 12)
            if fsmooth <= 0: fsmooth = 12
            base_sigma = 60 // (fsmooth / 12 if fsmooth > 0 else 1)
            df_ref = 44100.0 / 65536.0
            sigma_hz = base_sigma * df_ref
            df_cur = (fs_v / cfg.num_taps)
            sigma_bins = sigma_hz / df_cur if df_cur > 0 else base_sigma

            logger.info(
                f"{fs_v//1000} kHz -> DF smoothing ON "
                f"(sigma = {sigma_bins:.1f} bins -> {sigma_hz:.1f} Hz)"
            )
        except Exception:
            logger.info(f"{fs_v//1000} kHz -> DF smoothing ON")
    else:
        logger.info(f"{fs_v//1000} kHz -> DF smoothing OFF")

def _pin_get(key, default=None):
    """
    Robust read from PyWebIO pin.
    pin behaves like a dict, but membership/tests can be fragile depending on session state.
    This helper NEVER raises.
    """
    try:
        v = pin.get(key, None)
        if v is None:
            return default
        return v
    except Exception:
        try:
            # Fallback to __getitem__ if available
            return pin[key]
        except Exception:
            return default

def _json_safe(obj, *, _depth=0, _max_depth=12):
    """
    Best-effort conversion of nested stats objects to JSON-serializable types.
    - numpy scalars -> float/int
    - numpy arrays -> lists
    - dict/list/tuple -> recursively converted
    - unknown objects -> string repr
    """
    try:
        if _depth > _max_depth:
            return str(obj)
        # Basic primitives
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj

        # numpy scalars / arrays (avoid importing numpy here; rely on duck-typing)
        try:
            import numpy as _np  # local import (already dependency)
            if isinstance(obj, _np.generic):
                # e.g. np.float64, np.int64
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        # dict
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # ensure keys are strings
                try:
                    ks = str(k)
                except Exception:
                    ks = "key"
                out[ks] = _json_safe(v, _depth=_depth + 1, _max_depth=_max_depth)
            return out

        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]

        # bytes
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return str(obj)

        # fallback
        return str(obj)
    except Exception:
        return str(obj)


def _build_diagnostics_dict(data, fs_v, l_st, r_st):
    """
    Single source of truth diagnostics object for Summary.txt and future parsing.
   Keep it stable; bump schema_version if changing structure.
    """
    # Extract a compact leveling block (StereoLink uses same window/offset for both)
    def _leveling_block(st):
        if not isinstance(st, dict):
            return {}
        win = st.get("smart_scan_range", None)
        try:
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                win = [float(win[0]), float(win[1])]
            else:
                win = None
        except Exception:
            win = None
        return {
            "method": st.get("offset_method", None),
            "window_hz": win,
            "offset_db": st.get("offset_db", None),
            "eff_target_db": st.get("eff_target_db", None),
            "tilt_slope_db_per_oct": st.get("tilt_slope_db_per_oct", None),
            "avg_confidence_pct": st.get("avg_confidence", None),
        }

    diag = {
        "schema_version": 1,
        "meta": {
            "program": PROGRAM_NAME,
            "version": VERSION,
            "fs_hz": int(fs_v),
            "taps": int(float(data.get("taps", 0) or 0)),
            "filter_type": str(data.get("filter_type", "") or ""),
            "multi_rate": bool(data.get("multi_rate_opt", False)),
            "ir_export_window_mode": str(data.get("ir_export_window_mode", "") or ""),
            "ir_export_window_tag": str(_irwin_tag(data.get("ir_export_window_mode"))),
        },
        "settings": _json_safe(data),
        "leveling": {
            "stereo_link": bool(data.get("stereo_link", False)),
            "left": _leveling_block(l_st),
            "right": _leveling_block(r_st),
        },
        "left": _json_safe(l_st),
        "right": _json_safe(r_st),
    }
    return diag

def _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer):
    update_status(t('stat_plot'))
    import time
    time.sleep(0.05)
    set_processbar('bar', 0.8)
    print("plot_smoothing_level =", data.get("plot_smoothing_level"))
    print("filter_smooth =", data.get("filter_smooth"))
    psl = data.get('plot_smoothing_level', 'Psychoacoustic')
    psl_str = psl if isinstance(psl, str) else f"1/{int(psl)} octave"
    with use_scope('results', clear=True):
        if l_st_f is None or r_st_f is None:
            put_error("Error: No results captured.")
            return
        

      
            
        # --- Acoustic Intelligence UI (single source of truth: SAME as Summary.txt) ---
        # No separate "measured vs filtered" logic in UI. We display the Summary-based result.
        l_ai = plots.calc_ai_summary_from_stats(l_st_f)
        r_ai = plots.calc_ai_summary_from_stats(r_st_f)

        l_score = float(l_ai.get("score") or 0.0)
        r_score = float(r_ai.get("score") or 0.0)
        avg_pred = (l_score + r_score) / 2.0
        avg_orig = avg_pred
        improvement = 0.0

        l_match = l_ai.get("match")
        r_match = r_ai.get("match")
        if (l_match is None) or (r_match is None):
            avg_match = 0.0
        else:
            avg_match = (float(l_match) + float(r_match)) / 2.0

        def _fmt_tilt(st, warn_thr=1.5):
            tilt = st.get('tilt_slope_db_per_oct', None)
            if tilt is None:
                return "—"
            try:
                tilt = float(tilt)
                if abs(tilt) > warn_thr:
                    return put_html(
                        f'<span title="Large broadband tilt detected during leveling, house curve not suitable for speaker in room.">'
                        f'{tilt:+.2f} dB/oct ⚠️'
                        f'</span>'
                    )
                else:
                    return f"{tilt:+.2f} dB/oct"
            except Exception:
                return "—"

        put_table([
            ['Speaker', 'L', 'R'],
            ['Target Level', f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"],
            ['Smart Scan Range',
             f"{l_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz",
             f"{r_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz"],
            ['Leveling Tilt',_fmt_tilt(l_st_f),_fmt_tilt(r_st_f)],
            ['Offset to Meas.', f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"],
            ['Acoustic Confidence', f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"],
            ['Estimated RT60', f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"],
            ['TDC (Temporal Decay Control)',
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             ),
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             )
            ]
        ])

        put_markdown(f"###  {t('rep_header')}")
        with put_collapse(" DSP info"):
            # Phase clamp reporting (always-on safety)
            def _phase_clamp_str(st: dict) -> str:
                try:
                    lim = float((st or {}).get('phase_corr_clamp_deg', 0.0) or 0.0)
                    bef = float((st or {}).get('phase_corr_max_before_deg', 0.0) or 0.0)
                    clipped = bool((st or {}).get('phase_corr_clipped', False))
                    if lim <= 0.0:
                        return "—"
                    if clipped:
                        return f"max={bef:.1f}° -> {lim:.1f}°"
                    return f"max={bef:.1f}° (limit {lim:.1f}°)"
                except Exception:
                    return "—"

            def _xo_fc_wrapped_str(st: dict) -> str:
                """
                Show per-XO wrapped phase delta at crossover frequency (debug sanity check).
                Looks for keys like: xo1_dphi_wrapped_deg@fc, xo2_dphi_wrapped_deg@fc, ...
                """
                try:
                    if not isinstance(st, dict) or not st:
                        return "—"

                    # Pull XO freqs from the human-readable summary to label the numbers (best-effort).
                    # Example xo_summary: "500.0Hz/12dB/oct, 2800.0Hz/12dB/oct"
                    xo_summary = str(st.get("xo_summary", "") or "")
                    freqs = []
                    for part in xo_summary.split(","):
                        part = part.strip()
                        if "Hz" in part:
                            try:
                                freqs.append(float(part.split("Hz")[0].strip()))
                            except Exception:
                                freqs.append(None)

                    items = []
                    for i in range(1, 6):
                        k = f"xo{i}_dphi_wrapped_deg@fc"
                        if k not in st:
                            continue
                        try:
                            v = float(st.get(k))
                        except Exception:
                            continue
                        # Label with frequency if available, else "XO i"
                        f_lbl = None
                        if i <= len(freqs) and freqs[i-1] is not None:
                            f_lbl = f"{int(round(freqs[i-1]))}Hz"
                        else:
                            f_lbl = f"XO{i}"
                        items.append(f"{f_lbl}:{v:+.1f}°")

                    return " | ".join(items) if items else "—"
                except Exception:
                    return "—"
            def _xo_fc_gd_str(st: dict) -> str:
                """
                Show per-XO group delay delta at crossover frequency (ms).
                Keys: xo{i}_dgd_ms@fc
                """
                try:
                    if not isinstance(st, dict) or not st:
                        return "—"

                    xo_summary = str(st.get("xo_summary", "") or "")
                    freqs = []
                    for part in xo_summary.split(","):
                        part = part.strip()
                        if "Hz" in part:
                            try:
                                freqs.append(float(part.split("Hz")[0].strip()))
                            except Exception:
                                freqs.append(None)

                    items = []
                    for i in range(1, 6):
                        k = f"xo{i}_dgd_ms@fc"
                        if k not in st:
                            continue
                        try:
                            v = float(st.get(k))
                        except Exception:
                            continue
                        
                        if i <= len(freqs) and freqs[i-1] is not None:
                            lbl = f"{int(round(freqs[i-1]))}Hz"
                        else:
                            lbl = f"XO{i}"
                        items.append(f"{lbl}:{v:+.2f} ms")

                    return " | ".join(items) if items else "—"
                except Exception:
                    return "—"

            # XO/HPF phase model reporting (from DSP stats; falls back safely)
            def _xo_phase_model_str(st: dict) -> str:
                try:
                    s = (st or {}).get("xo_summary", None)
                    if s is None or str(s).strip() == "":
                        return "—"
                    return str(s)
                except Exception:
                    return "—"

            def _xo_diff_raw_str(st: dict) -> str:
                try:
                    p = (st or {}).get("xo_diff_raw_max_phase_deg", None)
                    pf = (st or {}).get("xo_diff_raw_max_phase_hz", None)
                    pfc = (st or {}).get("xo_diff_raw_max_phase_xo_fc_hz", None)
                    g = (st or {}).get("xo_diff_raw_max_gd_ms", None)
                    gf = (st or {}).get("xo_diff_raw_max_gd_hz", None)
                    gfc = (st or {}).get("xo_diff_raw_max_gd_xo_fc_hz", None)
                    if p is None and g is None:
                        return "—"
                    parts = []
                    if p is not None and pf is not None:
                        if pfc is not None:
                            parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz (XO {float(pfc):.0f} Hz)")
                        else:
                            parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz")
                    if g is not None and gf is not None:
                        if gfc is not None:
                            parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz (XO {float(gfc):.0f} Hz)")
                        else:
                            parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz")
                    return " | ".join(parts) if parts else "—"
                except Exception:
                    return "—"

            def _xo_fc_gd_badge(st: dict) -> str:
                """
                Color badge based on worst |ΔGD@fc| across XO points.
                Thresholds (ms): <0.7 green, 0.7–1.5 yellow, >1.5 red.
                """
                try:
                    if not isinstance(st, dict) or not st:
                        return ""
                    vals = []
                    for i in range(1, 6):
                        k = f"xo{i}_dgd_ms@fc"
                        if k not in st:
                            continue
                        try:
                            vals.append(abs(float(st.get(k))))
                        except Exception:
                            pass
                    if not vals:
                        return ""
                    worst = max(vals)

                    if worst < 0.7:
                        label = "LOW"
                        bg = "rgba(46, 125, 50, 0.15)"
                        fg = "rgba(46, 125, 50, 1.0)"
                        title = "Small XO ΔGD@fc (typically subtle)."
                    elif worst < 1.5:
                        label = "MED"
                        bg = "rgba(255, 143, 0, 0.15)"
                        fg = "rgba(255, 143, 0, 1.0)"
                        title = "Moderate XO ΔGD@fc (often audible improvement with XO phase correction)."
                    else:
                        label = "HIGH"
                        bg = "rgba(211, 47, 47, 0.15)"
                        fg = "rgba(211, 47, 47, 1.0)"
                        title = "Large XO ΔGD@fc (aggressive crossover / lots of time smear)."

                    return (
                        f"<span title='{title}' "
                        f"style='display:inline-block; margin-left:6px; padding:1px 6px; "
                        f"border-radius:10px; font-size:11px; font-weight:600; "
                        f"background:{bg}; color:{fg}; vertical-align:middle;'>"
                        f"{label}</span>"
                    )
                except Exception:
                    return "" 

            def _hpf_diff_raw_str(st: dict) -> str:
                try:
                    p = (st or {}).get("hpf_diff_raw_max_phase_deg", None)
                    pf = (st or {}).get("hpf_diff_raw_max_phase_hz", None)
                    g = (st or {}).get("hpf_diff_raw_max_gd_ms", None)
                    gf = (st or {}).get("hpf_diff_raw_max_gd_hz", None)
                    if p is None and g is None:
                        return "—"
                    parts = []
                    if p is not None and pf is not None:
                        parts.append(f"max Δφ {float(p):.1f}° @ {float(pf):.0f} Hz")
                    if g is not None and gf is not None:
                        parts.append(f"max ΔGD {float(g):.2f} ms @ {float(gf):.0f} Hz")
                    return " | ".join(parts) if parts else "—"
                except Exception:
                    return "—"


            def _hpf_model_str(st: dict) -> str:
                try:
                    s = (st or {}).get("hpf_summary", None)
                    if s is None or str(s).strip() == "":
                        return "—"
                    return str(s)
                except Exception:
                    return "—"
            def _format_ir_window(data: dict) -> str:
                """
                Human-readable IR export window description.
                Avoids tool-specific terminology and reflects actual behavior.
                """
                mode = str(data.get('ir_export_window_mode', '') or '').lower()

                if mode == 'rew_asym':
                    l = data.get('ir_window_left', None)
                    r = data.get('ir_window_right', data.get('ir_window', None))
                    try:
                        if l is not None and r is not None:
                            return f"Asymmetric (Left {float(l):.1f} ms, Right {float(r):.1f} ms)"
                    except Exception:
                        pass
                    return "Asymmetric"

                # Auto / fallback
                return "Auto (adaptive)"

            # XO ΔGD@fc line with color badge (HTML for inline styling)
            _xo_gd_line = (
                f"XO ΔGD@fc: L {_xo_fc_gd_str(l_st_f)} | R {_xo_fc_gd_str(r_st_f)}"
                f"{_xo_fc_gd_badge(l_st_f) or _xo_fc_gd_badge(r_st_f)}"
            )

            put_markdown(dedent(f"""
            - **Lenght:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
            - **Resolution:** {data['fs']/data['taps']:.2f} Hz
            - **IR window:** {_format_ir_window(data)}
            - **FDW:** {data['fdw_cycles']}
            - **House curve:** {data['hc_mode']} — {data.get('hc_source', 'Unknown')} ({data['mag_c_min']}-{data['mag_c_max']} Hz)
            - **Filter type:** {data['filter_type']}
            - **XO phase model:** L {_xo_phase_model_str(l_st_f)} | R {_xo_phase_model_str(r_st_f)}
            - **XO Δφ@fc (wrapped):** L {_xo_fc_wrapped_str(l_st_f)} | R {_xo_fc_wrapped_str(r_st_f)}
            - {_xo_gd_line}
            - **XO effect (theoretical raw):**
              - **L:** {_xo_diff_raw_str(l_st_f)}
              - **R:** {_xo_diff_raw_str(r_st_f)}
            - **HPF effect (theoretical raw):**
              - **L:** {_hpf_diff_raw_str(l_st_f)}
              - **R:** {_hpf_diff_raw_str(r_st_f)}
            - **Phase correction clamp:** L {_phase_clamp_str(l_st_f)} | R {_phase_clamp_str(r_st_f)}
            - **Smoothing view:** {psl_str}
            - **Leveling algo:** {data.get('lvl_algo', '')}
            """), sanitize=False)

        put_tabs([
            {'title': 'Left Channel', 'content': put_html(plots.generate_prediction_plot(
                f_l, m_l, p_l, l_imp_f, data['fs'], "Left",
                None, l_st_f, data['mixed_freq'], "low",
                create_full_html=False,
                plot_smoothing_level=data.get('plot_smoothing_level', 'Psychoacoustic')
            ))},
            {'title': 'Right Channel', 'content': put_html(plots.generate_prediction_plot(
                f_r, m_r, p_r, r_imp_f, data['fs'], "Right",
                None, r_st_f, data['mixed_freq'], "low",
                create_full_html=False,
                plot_smoothing_level=data.get('plot_smoothing_level', 'Psychoacoustic')
            ))}
        ])
        put_file(fname, zip_buffer.getvalue(), label=" DOWNLOAD FILTER ZIP")
        
        put_success(t('done_msg'))
        update_status(t('stat_done'))
        set_processbar('bar', 1.0)
    return main
