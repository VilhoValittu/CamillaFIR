import io
import json
import logging
import os
import sys
import zipfile
import typing
import scipy.io.wavfile
from datetime import datetime
from textwrap import dedent

import numpy as np
from pywebio import config, start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.session import set_env
from camillafir_config import load_config, save_config
from camillafir_i18n import t
from camillafir_housecurve import _normalize_hc_mode_key, get_house_curve_by_name, load_target_curve, load_house_curve
from camillafir_io.measurements_loader import load_measurements_lr
from camillafir_io.measurements_txt import parse_measurements_from_path
from camillafir_ui_helpers import (
    update_mode_desc,
    apply_mode_defaults_to_ui,
    update_taps_auto_info,
    update_lvl_ui,
    apply_tdc_preset,
    apply_afdw_preset,
    put_guide_section,
)
from camillafir_pipeline import (
    collect_ui_data,
    log_df_smoothing_toggle,
    build_xos_hpf,
    filter_type_short,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
    build_filter_config,
)
import camillafir_dsp as dsp
import camillafir_plot as plots
import models
from models import FilterConfig
from camillafir_modes import apply_mode_to_cfg, MODE_DEFAULTS

#from camillafir_rew_api import *
print("USING models.py      =", models.__file__)
print("USING camillafir_dsp =", dsp.__file__)
print("USING camillafir_plot=", plots.__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")




VERSION = "v2.8.2.1"  # changed file structure

# Change log:
# v.2.8.2.1 changed file structure to more debug-friendly format
# v2.8.2: [UI] improved robustness of file upload parsing from browser & added xo_help translation
# v2.8.1.2" [UI/DSP] bug fix for modes selection, that was not saving ui state correctly
# v2.8.1.1" [UI] small ui-update for modes selection
# v2.8.1: [DSP] fix A-FDW bandwidth limits & UI display
# v2.8.0: [UI] removed html dashboard export (now PNG only)
# v2.7.9: [UI] fix custom house curve upload
# v2.7.8: [IO] fix WAV parsing – phase unwrap
# v2.7.7: [DSP] fix HF phase handling
# v2.7.6: [IO] fix WAV parsing smoothing

PROGRAM_NAME = "CamillaFIR"
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000


def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50; margin-bottom: 10px;')

@config(theme="dark")
def main():
    set_env(output_max_width='1850px') 
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
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]; taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; slope_opts = [6, 12, 18, 24, 36, 48]
    
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
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')), 
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),


        # Auto-taps info (shown only when multi_rate_opt enabled)
        put_scope('taps_auto_info_scope_basic'),
        
        # Row 2: Filter type and Mixed frequency
        put_row([
            put_radio('filter_type', label=t('filter_type'), 
                    options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), t('ft_asymmetric')], 
                    value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')), 
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help'))
        ]),
        
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        
        put_select('lvl_algo', label="Algo", options=['Median', 'Average'], value=get_val('lvl_algo', 'Median'), help_text=t('lvl_algo_help')),
        put_select(
                    'smoothing_type',
                    label=t('smooth_type'),
                    options=[
                        {'label': t('smooth_std'), 'value': 'Standard'},
                        {'label': t('smooth_psy'), 'value': 'Psychoacoustic'}
                    ],
                    value=get_val('smoothing_type', 'Psychoacoustic'),
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
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=t('max_boost_help')),
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
                    'smoothing_level',
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
                    value=get_val('smoothing_level', 12),
                    help_text=t('smoothing_level_help'),
                ),
              
        
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
        
        put_markdown("#### ⏱️ Asymmetric Linear - windowing"),
        put_row([
            put_input('ir_window_left', label="Left Window (ms)", type=FLOAT, value=get_val('ir_window_left', 100.0), help_text=t('ir_matala')),
            put_input('ir_window', label="Right Window (ms)", type=FLOAT, value=get_val('ir_window', 500.0), help_text=t('ir_korkea'))
        ]),
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
            put_select('hpf_slope', label=t('hpf_slope'), options=slope_opts, value=get_val('hpf_slope', 24))
        ])
    ]

#--- #5 XO
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
        ] for i in range(1, 6)])
    ]

    # Draw tabs
    put_tabs([
        {'title': t('tab_files'), 'content': tab_files}, 
        {'title': t('tab_basic'), 'content': tab_basic}, 
        {'title': t('tab_target'), 'content': tab_target}, 
        {'title': t('tab_adv'), 'content': tab_adv}, 
        {'title': t('tab_xo'), 'content': tab_xo}
    ])



    update_lvl_ui()

    pin_on_change('lvl_mode', onchange=update_lvl_ui)
    pin_on_change('lvl_min', onchange=update_lvl_ui)
    pin_on_change('lvl_max', onchange=update_lvl_ui)

    # Mode description: initial render + live updates
    pin_on_change('mode', onchange=update_mode_desc)
    update_mode_desc()

    # Auto-taps UI updater: react when multi-rate toggles (tab_files) or basic changes
    pin_on_change('multi_rate_opt', onchange=update_taps_auto_info)
    pin_on_change('fs', onchange=update_taps_auto_info)
    pin_on_change('taps', onchange=update_taps_auto_info)
    update_taps_auto_info()

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
            base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)
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


def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(pin['enable_afdw']) if 'enable_afdw' in pin else bool(data.get('enable_afdw', False))
        enable_tdc = bool(pin['enable_tdc']) if 'enable_tdc' in pin else bool(data.get('enable_tdc', False))
        tdc_strength = float(data.get('tdc_strength', 0.0) or 0.0)
        fdw_cycles = float(data.get('fdw_cycles', 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0

        df_on = bool(pin['df_smoothing']) if 'df_smoothing' in pin else bool(data.get('df_smoothing', False))
        df_ref = 44100.0 / 65536.0
        base_sigma = 60 // (data.get('smoothing_level', 12) / 12 if (data.get('smoothing_level', 12) or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = (float(fs_v) / float(data.get('taps', 65536) or 65536))
        sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Sample rate: {int(fs_v)} Hz\n"

        if enable_afdw:
            summary_content += "FDW mode: Adaptive (A-FDW)\n"
            summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"
            summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width -> {afdw_min_oct_width:.3f})\n"
            summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
        else:
            summary_content += "FDW mode: Fixed\n"
            summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"

        summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
        if enable_tdc:
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength/100.0:.3f})\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"
    except Exception:
        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += "Could not compute effective params (unexpected data/pin state).\n"

    return summary_content

def _append_acoustic_events(summary_content, l_st, r_st):
    for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
        reflections = st.get('reflections') or []
        if reflections:
            summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
            summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'Error (ms)':<12} {'Dist (m)':<10}\n"
            summary_content += "-" * 50 + "\n"
            for rev in reflections:
                freq = float(rev.get('freq', 0) or 0)
                ev_type = str(rev.get('type', 'Event') or 'Event')
                gd_error = float(rev.get('gd_error', 0) or 0)
                dist = float(rev.get('dist', 0) or 0)
                summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {dist:<10}\n"
        # Always report headroom/normalization per side (even if no events)
        summary_content += f"\n=== HEADROOM MANAGEMENT ({side}) ===\n"
        summary_content += f"Normalize: {'ON' if bool(st.get('do_normalize', False)) else 'OFF'}\n"
        summary_content += f"Peak Gain (pre-headroom): {float(st.get('peak_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Headroom: {float(st.get('auto_headroom_db', 0.0)):.2f} dB\n"
        summary_content += f"Final Max (gain+global+headroom): {float(st.get('final_max_db', 0.0)):.2f} dB\n"
        # Diagnostics for boost/cut processing
        summary_content += f"\n=== BOOST/CUT DIAGNOSTICS ({side}) ===\n"
        summary_content += f"Config: max_boost_db={float(st.get('max_boost_db', 0.0)):.2f} dB, "
        summary_content += f"max_cut_db={float(st.get('max_cut_db', 0.0)):.2f} dB\n"
        summary_content += f"Config: low_bass_cut_hz={float(st.get('low_bass_cut_hz', 0.0)):.1f} Hz, "
        summary_content += f"exc_prot={'ON' if bool(st.get('exc_prot', False)) else 'OFF'}, "
        summary_content += f"exc_freq={float(st.get('exc_freq', 0.0)):.1f} Hz, "
        summary_content += f"max_slope_db_per_oct={float(st.get('max_slope_db_per_oct', 0.0)):.1f}\n"
        summary_content += f"Result (post-clamp): boost_peak={float(st.get('boost_peak_db', 0.0)):.2f} dB, "
        summary_content += f"cut_peak={float(st.get('cut_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_bins', 0))}\n"
        summary_content += f"Net boost peak (post global/headroom): {float(st.get('net_boost_peak_db', 0.0)):.2f} dB\n"
        summary_content += f"Candidate (pre-softclip): boost_peak={float(st.get('boost_candidate_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_candidate_bins', 0))}, "
        summary_content += f"lowbass_boost_bins={int(st.get('boost_candidate_bins_lowbass', 0))}, "
        summary_content += f"excprot_boost_bins={int(st.get('boost_candidate_bins_excprot', 0))}\n"
        summary_content += f"Boost blocked reason: {str(st.get('boost_blocked_reason', 'n/a'))}\n"
        summary_content += f"\n=== CLAMP DIAGNOSTICS ({side}) ===\n"
        summary_content += f"{str(st.get('clamp_summary', 'n/a'))}\n"
        summary_content += (
            f"soft_clip: boost_bins={int(st.get('softclip_boost_bins', 0))}, "
            f"cut_bins={int(st.get('softclip_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('softclip_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('softclip_worst_over_cut_db', 0.0)):.2f} dB\n"
        )
        summary_content += (
            f"hard_clamp: boost_bins={int(st.get('hardclamp_boost_bins', 0))}, "
            f"cut_bins={int(st.get('hardclamp_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('hardclamp_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('hardclamp_worst_over_cut_db', 0.0)):.2f} dB\n"
        )


        # --- Stage checkpoints table ---
        probes = st.get("stage_probes") or {}
        if isinstance(probes, dict) and probes:
            summary_content += f"\n=== STAGE CHECKPOINTS ({side}) ===\n"
            summary_content += f"{'Stage':<22} {'BoostPk':>8} {'CutPk':>8} {'BoostBins':>10} {'CutBins':>8} {'NetBoostPk':>11}\n"
            summary_content += "-" * 75 + "\n"
            order = [
                "after_gain_apply",
                "after_lowbass_policy",
                "after_slope",
                "after_fade",
                "pre_softclip",
                "post_softclip",
                "post_hardclamp",
            ]
            for key in order:
                p = probes.get(key)
                if not isinstance(p, dict):
                    continue
                stage = str(p.get("stage", key))
                bpk = float(p.get("boost_peak_db", 0.0) or 0.0)
                cpk = float(p.get("cut_peak_db", 0.0) or 0.0)
                bb  = int(p.get("boost_bins", 0) or 0)
                cb  = int(p.get("cut_bins", 0) or 0)
                nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
                summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"

            summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
            summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

            # --- Mode peak (robust formatting; fixes lost 'n/a' line) ---
            pk_hz = st.get('bass_first_mode_peak_hz', None)
            pk_sc = st.get('bass_first_mode_peak_score', None)
            if (pk_hz is not None) and (pk_sc is not None):
                summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
            else:
                summary_content += "Mode peak: n/a\n"

            summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"

            # --- BF debug stats (if present) ---
            rm_max = st.get('bass_first_roommode_max_20_200', None)
            rel_mean = st.get('bass_first_rel_mean_20_200', None)
            rel_min = st.get('bass_first_rel_min_20_200', None)
            conf_eff_mean = st.get('bass_first_conf_eff_mean_20_200', None)
            conf_eff_min = st.get('bass_first_conf_eff_min_20_200', None)
            floor_applied = bool(st.get('bass_first_conf_floor_applied', False))
            if (rm_max is not None) or (rel_mean is not None) or (rel_min is not None) or (conf_eff_mean is not None):
                summary_content += (
                    f"BF masks (20–200): "
                    f"roommode_max={float(rm_max or 0.0):.3f}, "
                    f"rel_mean(raw)={float(rel_mean or 0.0):.3f}, "
                    f"rel_min(raw)={float(rel_min or 0.0):.3f}, "
                    f"conf_eff_mean={float(conf_eff_mean or 0.0):.3f}, "
                    f"conf_eff_min={float(conf_eff_min or 0.0):.3f}, "
                    f"conf_floor_applied={'YES' if floor_applied else 'NO'}\n"
                )

            # --- Optional source tag (only if caller stored it in stats) ---
            # e.g. st["bass_first_source"] = "WAV" or "TXT/REW"
            src = st.get("bass_first_source", None)
            if isinstance(src, str) and src.strip():
                summary_content += f"BassFirst source: {src.strip()}\n"


 

    return summary_content

def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    write_dashboards: bool = True,
):
    sum_name = f"Summary_{ft_short}_{fs_v}Hz.txt"
    l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.png"
    r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.png"

    summary_content = plots.format_summary_content(data, l_st, r_st)
    # Include explicit house-curve provenance (preset vs upload/local file)
    try:
        hc_src = str(data.get('hc_source', '') or '').strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except Exception:
        pass
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    summary_content = _append_acoustic_events(summary_content, l_st, r_st)

    if 'auto_align' in l_st:
        res = l_st['auto_align']
        summary_content += "\n=== AUTO-ALIGN ===\n"
        summary_content += f"Delay: {res['delay_ms']} ms\n"
        summary_content += f"Distance Diff: {res['distance_cm']} cm\n"
        summary_content += f"Gain Diff: {res['gain_diff_db']} dB\n"

    zf.writestr(sum_name, summary_content)

    # Policy: ZIP size control
    # We store only ONE dashboard pair into the ZIP (forced, no UI choice).
    # Dashboard format: PNG only (no HTML), so it opens everywhere without Plotly JS.
    if bool(write_dashboards):
        html_l, fig_l = plots.generate_prediction_plot(
            f_l, m_l, p_l, l_imp, fs_v, "Left",
            None, l_st, data['mixed_freq'], "low",
            create_full_html=False,
            return_fig=True
        )
        if fig_l is not None:
            zf.writestr(l_dash_name, plots.plotly_fig_to_png(fig_l, scale=2))
        else:
            # keep a clue in the ZIP if plotly rendering failed
            zf.writestr(l_dash_name.replace(".png", ".txt"), str(html_l))

        html_r, fig_r = plots.generate_prediction_plot(
            f_r, m_r, p_r, r_imp, fs_v, "Right",
            None, r_st, data['mixed_freq'], "low",
            create_full_html=False,
            return_fig=True
        )
        if fig_r is not None:
            zf.writestr(r_dash_name, plots.plotly_fig_to_png(fig_r, scale=2))
        else:
            zf.writestr(r_dash_name.replace(".png", ".txt"), str(html_r))

    hlc_cfg = generate_hlc_config(fs_v, ft_short, file_ts)
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz.cfg", hlc_cfg)
    yaml_content = generate_raspberry_yaml(
        fs_v,
        ft_short,
        file_ts,
        master_gain_db=float(data.get('gain', 0.0) or 0.0)
    )
    zf.writestr(f"camilladsp_{ft_short}_{fs_v}Hz.yml", yaml_content)

def _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer):
    update_status(t('stat_plot'))
    set_processbar('bar', 1.0)

    with use_scope('results', clear=True):
        if l_st_f is None or r_st_f is None:
            put_error("Error: No results captured.")
            return

        put_success(t('done_msg'))

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


        put_table([
            ['Speaker', 'L', 'R'],
            ['Target Level', f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"],
            ['Smart Scan Range',
             f"{l_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz",
             f"{r_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz"],
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
            put_markdown(dedent(f"""
            - **Lenght:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
            - **Resolution:** {data['fs']/data['taps']:.2f} Hz
            - **IR window:** {data['ir_window']} ms
            - **FDW:** {data['fdw_cycles']}
            - **House curve:** {data['hc_mode']} — {data.get('hc_source', 'Unknown')} ({data['mag_c_min']}-{data['mag_c_max']} Hz)
            - **Filter type:** {data['filter_type']}
            - **Smoothing:** {data['lvl_algo']}
            """))

        put_tabs([
            {'title': 'Left Channel', 'content': put_html(plots.generate_prediction_plot(f_l, m_l, p_l, l_imp_f, data['fs'], "Left", None, l_st_f, data['mixed_freq'], "low", create_full_html=False))},
            {'title': 'Right Channel', 'content': put_html(plots.generate_prediction_plot(f_r, m_r, p_r, r_imp_f, data['fs'], "Right", None, r_st_f, data['mixed_freq'], "low", create_full_html=False))}
        ])
        put_file(fname, zip_buffer.getvalue(), label=" DOWNLOAD FILTER ZIP")


def process_run():
    # 1) UI -> data dict (new unified collector)
    data = collect_ui_data(pin)
    save_config(data)

    # 2) Measurements (upload OR local paths)
    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    if f_l is None or f_r is None:
        toast("Measurement files missing! Load Left/Right or give local.", color='red')
        return

    # 3) Target / house curve
    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path
    )
    data['hc_source'] = hc_source
    logger.info(f"House curve source: {hc_source}")
    # 4) XO + HPF
    xos, hpf = build_xos_hpf(data)

    # 5) (Optional) DF smoothing log
    df_on = log_df_smoothing_toggle(pin, logger)


    # 6) Sample rates list
    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(target_rates, multi_rate_on=multi_rate_on, forced_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ))


    put_processbar('bar')
    put_scope('status_area')
    update_status(t('stat_reading'))
    set_processbar('bar', 0.2)
    zip_buffer = io.BytesIO()
    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = filter_type_short(data['filter_type'])
    split, zoom = data['mixed_freq'], t('zoom_hint')
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, fs_v in enumerate(target_rates):
            if data['multi_rate_opt']:
                taps_v = scale_taps_with_fs(fs_v)
                logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> 65536 taps)")
            else:
                taps_v = int(data['taps'])
            update_status(f"Lasketaan {fs_v}Hz...")
            set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))

            cfg = build_filter_config(
                FilterConfig_cls=FilterConfig,
                fs_v=fs_v,
                taps_v=taps_v,
                data=data,
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin=pin,
            )



            is_wav = detect_is_wav_source(data, pin)
            try:
                setattr(cfg, "is_wav_source", bool(is_wav))
            except Exception:
                pass



            _log_df_smoothing_for_fs(cfg, fs_v, df_on)

            l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
            r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

            if bool(data.get('stereo_link', False)):
                try:
                    from camillafir_leveling import compute_leveling

                    if isinstance(l_st, dict) and isinstance(r_st, dict):
                        fx_l = np.asarray(l_st.get('freq_axis') or [], dtype=float)
                        fx_r = np.asarray(r_st.get('freq_axis') or [], dtype=float)
                        if fx_l.size > 32 and fx_l.size == fx_r.size and np.allclose(fx_l, fx_r, rtol=0, atol=1e-9):
                            # reconstruct m_anal (A-FDW already applied inside DSP):
                            mL = np.asarray(l_st.get('measured_mags') or [], dtype=float) + float(l_st.get('offset_db', 0.0) or 0.0)
                            mR = np.asarray(r_st.get('measured_mags') or [], dtype=float) + float(r_st.get('offset_db', 0.0) or 0.0)
                            tgt = np.asarray(l_st.get('target_mags') or [], dtype=float)
                            if mL.size == fx_l.size and mR.size == fx_l.size and tgt.size == fx_l.size:
                                m_avg = 0.5 * (mL + mR)

                                (
                                    _tl,
                                    off,
                                    _mlw,
                                    _tlw,
                                    _meth,
                                    smin,
                                    smax,
                                ) = compute_leveling(cfg, fx_l, m_avg, tgt)

                                # Force identical window+offset for both channels
                                cfg.stereo_link = True
                                cfg.lvl_force_window = (float(smin), float(smax))
                                cfg.lvl_force_offset_db = float(off)

                                # Regenerate with forced leveling
                                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
                                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

                                # Tag method in stats for visibility
                                if isinstance(l_st, dict):
                                    l_st['offset_method'] = str(l_st.get('offset_method') or '') + " (StereoLink)"
                                if isinstance(r_st, dict):
                                    r_st['offset_method'] = str(r_st.get('offset_method') or '') + " (StereoLink)"
                except Exception as e:
                    logger.warning(f"Stereo link leveling failed (continuing without link): {e}")


            # ------------------------------------------------------------------
            # RT60 reliability tagging for scoring:
            # - WAV/IR path: RT60 is IR-derived => higher reliability
            # - TXT/REW FR path: RT60 is proxy/estimate => lower reliability
            # This is used by the new Acoustic Score formula (bonus is weighted).
            # ------------------------------------------------------------------
            try:
                rt_rel = 1.0 if bool(is_wav) else 0.25
                rt_src = "WAV" if bool(is_wav) else "TXT/REW"
                if isinstance(l_st, dict):
                    l_st["rt60_reliability"] = float(rt_rel)
                    l_st["rt60_source"] = rt_src
                if isinstance(r_st, dict):
                    r_st["rt60_reliability"] = float(rt_rel)
                    r_st["rt60_source"] = rt_src
            except Exception:
                pass


            l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
            r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)
            # Build comparison grid per sample-rate (needed for correct UI scoring with WAV)
            if bool(data.get("comparison_mode", False)):
                try:
                    l_st = plots._make_comparison_stats(l_st, int(fs_v), int(taps_v))
                    r_st = plots._make_comparison_stats(r_st, int(fs_v), int(taps_v))
                except Exception as e:
                    logger.warning(f"Comparison-mode stats failed: {e}")

            # ------------------------------------------------------------------
            # Time alignment
            #
            # TXT-compatible behavior: if generate_filter() produced explicit
            # delay estimates (delay_samples), prefer those for alignment.
            # This avoids the "peak-pick" method drifting when the main impulse
            # peak is not stable (common with heavy LF energy / long tails).
            #
            # If delay_samples are missing, fall back to the legacy peak-pick.
            # ------------------------------------------------------------------
            if data['align_opt']:
                d_s = None

                # Prefer delay_samples (TXT-compatible)
                try:
                    dl = l_st.get('delay_samples', None) if isinstance(l_st, dict) else None
                    dr = r_st.get('delay_samples', None) if isinstance(r_st, dict) else None
                    if dl is not None and dr is not None:
                        dl_i = int(round(float(dl)))
                        dr_i = int(round(float(dr)))
                        d_s = dl_i - dr_i
                except Exception:
                    d_s = None

                # Fallback: align by impulse peak
                if d_s is None:
                    d_s = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))

                if d_s > 0:
                    r_imp = np.roll(r_imp, d_s)
                elif d_s < 0:
                    l_imp = np.roll(l_imp, -d_s)

            # UI "results" view: show the same fs as the (single) dashboard fs in multi-rate.
            if fs_v == dash_fs:
                l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp

            

            if 'delay_samples' in l_st and 'delay_samples' in r_st:
                diff_samples = r_st['delay_samples'] - l_st['delay_samples']
                delay_ms = round((diff_samples / fs_v) * 1000, 3)
                distance_cm = round((delay_ms / 1000) * 34300, 2)
                gain_diff = round(l_st['offset_db'] - r_st['offset_db'], 2)
                l_st['auto_align'] = {'delay_ms': delay_ms, 'distance_cm': distance_cm, 'gain_diff_db': gain_diff}

            wav_l, wav_r = io.BytesIO(), io.BytesIO()
            scipy.io.wavfile.write(wav_l, fs_v, l_imp.astype(np.float32))
            scipy.io.wavfile.write(wav_r, fs_v, r_imp.astype(np.float32))
            zf.writestr(f"L_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_l.getvalue())
            zf.writestr(f"R_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_r.getvalue())

            _write_fs_outputs(
                zf,
                data,
                fs_v,
                ft_short,
                file_ts,
                f_l,
                m_l,
                p_l,
                l_imp,
                l_st,
                f_r,
                m_r,
                p_r,
                r_imp,
                r_st,
                write_dashboards=(not multi_rate_on) or (int(fs_v) == int(dash_fs))
            )

    # --- Save ZIP into filters/ directory ---
    filters_dir = os.path.join(os.getcwd(), "filters")
    os.makedirs(filters_dir, exist_ok=True)

    fname = f"CamillaFIR_{ft_short}_{ts}.zip"
    out_path = os.path.join(filters_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Saved: {os.path.abspath(out_path)}"
    except Exception:
        save_msg = "Zip saving failed."


    # --- Ensure UI has stats even if fs selection didn't hit (e.g. WAV/local path quirks) ---
    if l_st_f is None:
        l_st_f = l_st
    if r_st_f is None:
        r_st_f = r_st
    if l_imp_f is None:
        l_imp_f = l_imp
    if r_imp_f is None:
        r_imp_f = r_imp

    # --- Ensure UI scoring has filter_mags (so Measured != Filtered) ---
    try:
        fs_sel = int(data.get('fs') or 44100)
    except Exception:
        fs_sel = 44100
    _inject_filter_mags_for_ui(l_st_f, l_imp_f, fs_sel)
    _inject_filter_mags_for_ui(r_st_f, r_imp_f, fs_sel)

    logger.info(f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
                f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis',[]))}/{len(l_st_f.get('cmp_measured_mags',[]))}/{len(l_st_f.get('cmp_target_mags',[]))}")

    _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer)

#snipet
def generate_raspberry_yaml(fs, ft_short, file_ts, master_gain_db=0.0):
    import textwrap

    # FIR .wav files (CamillaDSP replaces $samplerate$ at runtime)
    l_wav = f"../coeffs/L_{ft_short}_$samplerate$Hz_{file_ts}.wav"
    r_wav = f"../coeffs/R_{ft_short}_$samplerate$Hz_{file_ts}.wav"

    # sanitize
    try:
        g = float(master_gain_db)
    except Exception:
        g = 0.0

    return textwrap.dedent(f"""
    description: null
    devices:
      capture:
        type: Stdin
        channels: 2
        format: S32LE
      playback:
        type: Alsa
        device: plughw:0,0
        channels: 2
        format: S32LE
      samplerate: {int(fs)}
      chunksize: 4096
      queuelimit: 1
      volume_ramp_time: 150

    filters:
      ir_left:
        type: Conv
        parameters:
          type: Wav
          filename: {l_wav}
          channel: 0

      ir_right:
        type: Conv
        parameters:
          type: Wav
          filename: {r_wav}
          channel: 0

      mastergain:
        type: Gain
        parameters:
          gain: {g:.6g}

    mixers:
      stereo:
        channels:
          in: 2
          out: 2
        mapping:
          - dest: 0
            sources:
              - channel: 0
                gain: 0
          - dest: 1
            sources:
              - channel: 1
                gain: 0

    pipeline:
      - type: Mixer
        name: stereo
      - type: Filter
        channels: [0]
        names: [mastergain, ir_left]
      - type: Filter
        channels: [1]
        names: [mastergain, ir_right]

    processors: null
    title: {ft_short}
    """).strip()




def generate_hlc_config(fs, ft_short, file_ts):
    """
    Luo standardin .cfg konfiguraatiotiedoston (HLC, Convolver VST, BruteFIR).
    Generoi tiedostonimet sisäisesti samoilla säännöillä kuin YAML-funktio.
    """
    # Generoidaan tiedostonimet täsmälleen samalla kaavalla kuin tallennuksessa
    l_name = f"L_{ft_short}_{fs}Hz_{file_ts}.wav"
    r_name = f"R_{ft_short}_{fs}Hz_{file_ts}.wav"

    config = [
        f"{int(fs)} 2 2 0",  # Header: SampleRate, 2 In, 2 Out, 0 Offset
        "0 0",
        "0 0",
        f"{l_name}",         # Vasen tiedosto
        "0",                 # Input Index (L)
        "0.0",
        "0.0",                 # Output Index (L)
        f"{r_name}",         # Oikea tiedosto
        "0",                 # Input Index (R)
        "1.0",
        "1.0"                  # Output Index (R)
    ]
    return "\n".join(config)


def _ui_pick(stats, key):
    """
    UI helper: pick comparison-grid data if analysis_mode == 'comparison'
    """
    if not stats:
        return None
    mode = str(stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _pick_cmp(stats, key):
    """
    Return comparison-mode arrays for UI scoring if available.
    """
    if not stats:
        return None
    if str(stats.get("analysis_mode", "native")).lower() == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _ensure_scoring_keys(st, f_in, m_in, hc_f, hc_m):
    """
    Ensure UI scoring keys exist in stats dict (WAV/TXT safe).
    - freq_axis, measured_mags, target_mags, confidence_mask
    """
    try:
        if st is None:
            return st

        f = np.asarray(f_in or [], dtype=float)
        m = np.asarray(m_in or [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        # target mags (fallback from house curve if missing)
        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f or [], dtype=float)
                hm = np.asarray(hc_m or [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

        # confidence mask (fallback to ones if missing)
        if st.get("confidence_mask") is None:
            if f.size > 1:
                st["confidence_mask"] = np.ones_like(f, dtype=float)
        return st
    except Exception:
        return st

_HOUSE_FREQS = np.array([
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
], dtype=float)

def _resample_to_freq_axis(freqs_dst: np.ndarray, arr: np.ndarray, freqs_src: np.ndarray) -> np.ndarray:
    """Safe 1D interpolation in log-frequency domain."""
    if arr.size == 0 or freqs_src.size == 0 or freqs_dst.size == 0:
        return arr
    # clip to valid region
    f1 = np.maximum(freqs_src.astype(float), 1.0)
    f2 = np.maximum(freqs_dst.astype(float), 1.0)
    lf1 = np.log10(f1)
    lf2 = np.log10(f2)
    # Ensure monotonic source
    order = np.argsort(lf1)
    lf1 = lf1[order]
    a1 = arr.astype(float)[order]
    return np.interp(lf2, lf1, a1, left=a1[0], right=a1[-1])


def calculate_target_match(st):
    """Calculates how well the corrected response follows the target curve (0-100%)."""
    if not st:
        return 0.0

    freqs = np.asarray(_ui_pick(st, 'freq_axis') or [], dtype=float)
    meas  = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt  = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if freqs.size == 0 or meas.size == 0 or target.size == 0:
        return 0.0

    # WAV-polulla filter_mags voi puuttua -> tulkitaan 0 dB korjaukseksi
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)
    # If filter mags are missing (common in some UI paths), treat as 0 dB correction
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)

    # If WAV measurement: measured/target are dense FFT grid, but filter may be on 19-point house grid.
    # Resample target/filter to the measurement freq_axis when shapes differ.
    if target.size != freqs.size:
        # common case: target on house grid
        if target.size == _HOUSE_FREQS.size:
            target = _resample_to_freq_axis(freqs, target, _HOUSE_FREQS)
        else:
            # last resort: truncate
            n = min(freqs.size, meas.size, target.size)
            freqs, meas, target = freqs[:n], meas[:n], target[:n]

    if filt.size != freqs.size:
        if filt.size == _HOUSE_FREQS.size:
            filt = _resample_to_freq_axis(freqs, filt, _HOUSE_FREQS)
        else:
            n = min(freqs.size, meas.size, filt.size, target.size)
            freqs, meas, target, filt = freqs[:n], meas[:n], target[:n], filt[:n]

    # RMS virhe (dB) korjatusta vasteesta
    diff = (meas + filt) - target
    rms = float(np.sqrt(np.mean(diff * diff)))

    # Sama muunnos kuin Summaryssä (sigmoidi)
    m0 = 3.2   # dB @ 50%
    s0 = 0.9   # jyrkkyys
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))



def _avg_confidence_pct(st: dict) -> float:
    """
    UI helper: returns average confidence in percent.
    Supports comparison-mode keys (cmp_avg_confidence / cmp_confidence_mask).
    """
    if not st:
        return 0.0
    mode = str(st.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        v = st.get("cmp_avg_confidence", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        cm = np.asarray(st.get("cmp_confidence_mask", []) or [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    # native
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm = np.asarray(st.get("confidence_mask", []) or [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


def calculate_target_match_unfiltered(st: dict) -> float:
    """
    Target match for *unfiltered* response (measured vs target).
    Uses the same sigmoid mapping as calculate_target_match().
    """
    if not st:
        return 0.0
    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    if meas.size == 0 or target.size == 0:
        return 0.0
    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]
    diff = meas - target
    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))

def _inject_filter_mags_for_ui(st: dict, filt_ir, fs: int):
    """Ensure st has filter_mags on the same freq_axis as measured, so UI can score 'Filtered' correctly.

    Some pipelines didn't store filter_mags into stats; then UI 'Measured' and 'Filtered' collapse to the same value.
    This computes |FFT(filter_ir)| and interpolates it to st['freq_axis'] (or cmp_freq_axis if in comparison mode),
    storing it as (cmp_)filter_mags in dB.
    """
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        if st.get(key_g) is not None:
            return

        f_axis = np.asarray(st.get(key_f, []) or [], dtype=float)
        if f_axis.size < 4:
            return

        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 8:
            return

        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return

        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

        f_min = float(np.min(f_fft))
        f_max = float(np.max(f_fft))
        f_q = np.clip(f_axis, f_min, f_max)
        st[key_g] = np.interp(f_q, f_fft, g_db).tolist()
    except Exception:
        return


def calculate_score(st, is_predicted=False):
    """UI score (0..99) for Measured / Filtered.

    Note: this is *not* the same as Target Curve Match.
    It combines:
      - target match (sigmoid RMS mapping)
      - acoustic confidence
      - optional RT60 room-quality bonus/penalty (scaled by reliability)
    """
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if meas.size == 0 or target.size == 0:
        return float(np.clip(conf, 0.0, 99.0))

    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]

    if is_predicted:
        if filt.size >= n:
            filt = filt[:n]
        elif filt.size > 0:
            filt = np.pad(filt, (0, n - filt.size), mode='edge')
        else:
            filt = np.zeros(n, dtype=float)
        diff = (meas + filt) - target
    else:
        diff = meas - target

    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    match_pct = float(np.clip(match_pct, 0.0, 100.0))

    base = 0.55 * match_pct + 0.35 * conf  # 0..90

    rt_bonus = 0.0
    try:
        rt = float(st.get('rt60_val', None)) if st.get('rt60_val', None) is not None else None
    except Exception:
        rt = None
    try:
        rel = float(st.get('rt60_reliability', 0.0) or 0.0)
    except Exception:
        rel = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))

    if rt is not None and rt > 0:
        if rt <= 0.35:
            rt_bonus = ((0.35 - rt) / 0.25) * 15.0
        elif rt >= 0.55:
            rt_bonus = -min(15.0, ((rt - 0.55) / 0.35) * 15.0)
        rt_bonus *= rel

    events = st.get('cmp_reflections', st.get('reflections', [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(8.0, float(len(events)) * 1.0) * penalty_mult

    score = base + rt_bonus - event_penalty
    return float(np.clip(score, 0.0, 99.0))


if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
