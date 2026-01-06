import sys
import os
import io
import json
import locale
import zipfile 
import logging
import numpy as np
import scipy.io.wavfile
from datetime import datetime
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio import start_server, config

# IMPORT LOCAL MODULES
import camillafir_dsp as dsp
import camillafir_plot as plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

CONFIG_FILE = 'config.json'
TRANS_FILE = 'translations.json'

# --- VERSION HISTORY ---
# v2.5.3: Moved SigmaStudio guide to translations.json
# v2.5.2: Added SigmaStudio/ADAU1701 Guide
# v2.5.1: Added SigmaStudio support (TXT export + Low tap counts)
# v2.5.0: Major DSP overhaul (TOF Correction, Freq-Dependent Reg)
VERSION = "v2.5.3" 
PROGRAM_NAME = "CamillaFIR"
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0

def load_translations():
    try:
        with open(TRANS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

TRANSLATIONS = load_translations()

def t(key):
    lang = locale.getlocale()[0]
    lang = 'fi' if lang and 'fi' in lang.lower() else 'en'
    return TRANSLATIONS.get(lang, TRANSLATIONS.get('en', {})).get(key, key)

def put_guide_section():
    guides = [
        ('guide_taps', t('guide_title')),
        ('guide_ft', t('guide_ft_title')),
        ('guide_sigma', t('guide_sigma_title')), # Fetches title from JSON
        ('guide_mix', t('guide_mix_title')),
        ('guide_fdw', t('guide_fdw_title')),
        ('guide_reg', t('guide_reg_title')),
        ('guide_lvl', t('guide_lvl_title')),
        ('guide_sl', t('guide_sl_title')),
        ('guide_ep', t('guide_ep_title')),
    ]
    
    content = []
    for g_key, g_title in guides:
        if g_key == 'guide_sigma':
             # Fetches body text from JSON
             c = [put_markdown(t('guide_sigma_body'))]
             content.append(put_collapse(g_title, c))
        elif g_key == 'guide_reg':
            c = [put_text(t('guide_reg_desc')), put_markdown(f"**{t('guide_reg_why')}**"), put_text(t('guide_reg_how')), put_markdown(f"_{t('guide_reg_rec')}_").style('font-weight: bold;')]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_lvl':
            c = [put_text(t('guide_lvl_desc')), put_text(t('guide_lvl_algo')), put_text(t('guide_lvl_std'))]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_ep':
            c = [put_text(t('guide_ep_desc')), put_markdown(f"**{t('guide_ep_why')}**"), put_markdown(f"_{t('guide_ep_rec')}_").style('font-weight: bold;')]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_sl':
            c = [put_text(t('guide_sl_desc')), put_markdown(f"**{t('guide_sl_why')}**")]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_ft':
            c = [put_markdown(t('guide_ft_intro')), put_markdown("---"), put_markdown(f"**{t('guide_ft_lin_h')}**"), put_markdown(t('guide_ft_lin_body')), put_markdown("---"), put_markdown(f"**{t('guide_ft_min_h')}**"), put_markdown(t('guide_ft_min_body')), put_markdown("---"), put_markdown(f"**{t('guide_ft_mix_h')}**"), put_markdown(t('guide_ft_mix_body'))]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_taps':
            c = [put_markdown(f"**{t('guide_rule')}**"), put_markdown(f"`{t('guide_formula')}`"), put_markdown(f"**{t('guide_rec')}**"), put_markdown(t('guide_note'))]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_mix':
            c = [put_text(t('guide_mix_intro')), put_markdown(t('guide_mix_low')), put_markdown(t('guide_mix_mid')), put_markdown(t('guide_mix_high'))]
            content.append(put_collapse(g_title, c))
        elif g_key == 'guide_fdw':
            c = [put_text(t('guide_fdw_desc')), put_markdown(f"**{t('guide_fdw_auto')}**"), put_text(t('guide_fdw_low')), put_text(t('guide_fdw_high'))]
            content.append(put_collapse(g_title, c))

    put_collapse("❓ Ohjeet ja Oppaat (Klikkaa auki)", content)

def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50;')

def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Stereo', 'fs': 44100, 'taps': 65536,
        'filter_type': 'Linear Phase',
        'gain': 0.0, 
        'hc_mode': 'Harman (Standard +6dB)',
        'mag_correct': True,
        'smoothing_type': 'Psychoacoustic',
        'fdw_cycles': 15.0,
        'hc_min': 10.0, 'hc_max': 200.0, 
        'max_boost': 5.0,
        'lvl_mode': 'Automatic',
        'lvl_algo': 'Median', 
        'lvl_manual_db': 75.0,
        'lvl_min': 500.0, 'lvl_max': 2000.0,
        'normalize_opt': True,
        'align_opt': True,
        'multi_rate_opt': False,
        'reg_strength': 30.0,
        'stereo_link': False, 
        'exc_prot': False, 'exc_freq': 25.0,
        'hpf_enable': False, 'hpf_freq': 20.0, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12,
        'ir_window': 250.0, 'mixed_freq': 300.0, 'phase_limit': 20000.0
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                for k in ['mag_correct', 'normalize_opt', 'align_opt', 'multi_rate_opt', 'stereo_link', 'exc_prot', 'hpf_enable']:
                    if k in saved and isinstance(saved[k], list):
                        saved[k] = bool(saved[k])
                default_conf.update(saved)
        except: pass
    return default_conf

def save_config(data):
    try:
        clean_data = {k: v for k, v in data.items() if not k.startswith('file_')}
        with open(CONFIG_FILE, 'w') as f: json.dump(clean_data, f, indent=4)
    except: pass

def get_house_curve_by_name(name):
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0])
    if 'Harman8' in name or '+8dB' in name: mags = np.array([8.0, 7.9, 7.8, 7.6, 7.3, 6.9, 6.3, 5.5, 4.5, 3.4, 1.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    elif 'Toole' in name:
        freqs = np.array([20, 63, 100, 200, 400, 1000, 2000, 4000, 10000, 20000])
        mags = np.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0, -4.0, -6.0])
    elif 'Flat' in name: mags = np.zeros_like(freqs)
    elif 'Cinema' in name:
        freqs = np.array([20, 2000, 4000, 8000, 16000, 20000])
        mags = np.array([0.0, 0.0, -3.0, -9.0, -15.0, -18.0])
    else: mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    return freqs, mags

def parse_measurements_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        return data[:, 0], data[:, 1], data[:, 2] 
    except: return None, None, None

def parse_measurements_from_path(filepath):
    try:
        data = np.loadtxt(filepath, comments=['#', '*'])
        return data[:, 0], data[:, 1], data[:, 2]
    except: return None, None, None

@config(theme="dark")
def main():
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    
    put_guide_section()
    put_markdown("---")

    d = load_config()
    def get_val(k, default): return d.get(k, default)

    hc_options = [t('hc_harman'), t('hc_harman8'), t('hc_toole'), t('hc_bk'), t('hc_flat'), t('hc_cinema'), t('hc_mode_upload')]
    fs_options = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
    taps_options = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    slope_opts = [6, 12, 18, 24, 36, 48]

    # --- TABS UI ---
    
    tab_files = [
        put_markdown(f"### 📂 {t('grp_settings')}"),
        put_text(t('txt_paths')),
        
        put_file_upload('file_l', label=t('upload_l'), accept='.txt', placeholder=t('ph_l')),
        put_input('local_path_l', label=t('path_l'), value=get_val('local_path_l', ''), help_text=t('path_help')),
        
        put_file_upload('file_r', label=t('upload_r'), accept='.txt', placeholder=t('ph_r')),
        put_input('local_path_r', label=t('path_r'), value=get_val('local_path_r', ''), help_text=t('path_help')),
        
        put_select('fmt', label=t('fmt'), options=['WAV', 'TXT (SigmaStudio)'], value=get_val('fmt', 'WAV'), help_text="WAV: PC/CamillaDSP. TXT: Raw coefficients for Hardware DSPs."),
        put_radio('layout', label=t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=get_val('layout', t('layout_stereo')), inline=True),
        put_checkbox('multi_rate_opt', options=[{'label': t('multi_rate'), 'value': True}], value=[True] if get_val('multi_rate_opt', False) else [], help_text=t('multi_rate_help'))
    ]

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        put_row([
            put_select('fs', label=t('fs'), options=fs_options, value=get_val('fs', 44100), help_text=t('fs_help')),
            put_select('taps', label=t('taps'), options=taps_options, value=get_val('taps', 65536), help_text=t('taps_help')),
        ]),
        put_row([
            put_radio('filter_type', label=t('filter_type'), options=[t('ft_linear'), t('ft_min'), t('ft_mixed')], value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')),
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help')),
        ]),
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        put_select('lvl_mode', label=t('lvl_mode'), options=[t('lvl_auto'), t('lvl_man')], value=get_val('lvl_mode', t('lvl_auto')), help_text=t('lvl_mode_help')),
        put_input('lvl_manual_db', label=t('lvl_target_db'), type=FLOAT, value=get_val('lvl_manual_db', 75.0), help_text=t('lvl_manual_help'))
    ]

    tab_target = [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_select('hc_mode', label=t('hc_mode'), options=hc_options, value=get_val('hc_mode', t('hc_harman')), help_text=t('hc_mode_help')),
        put_row([
            put_input('hc_min', label=t('min_freq'), type=FLOAT, value=get_val('hc_min', 10.0), help_text=t('hc_range_help')),
            put_input('hc_max', label=t('max_freq'), type=FLOAT, value=get_val('hc_max', 200.0), help_text=t('hc_range_help')),
        ]),
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=t('max_boost_help')),
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 20000.0), help_text=t('phase_limit_help')),
        put_checkbox('mag_correct', options=[{'label': t('enable_corr'), 'value': True}], value=[True] if get_val('mag_correct', True) else [])
    ]

    tab_adv = [
        put_markdown(f"### 🛠️ {t('tab_adv')}"),
        put_row([
            put_select('smooth_type', label=t('smooth_type'), options=[t('smooth_std'), t('smooth_psy')], value=get_val('smoothing_type', t('smooth_psy')), help_text=t('smooth_help')),
            put_input('fdw_cycles', label=t('fdw'), type=FLOAT, value=get_val('fdw_cycles', 15.0), help_text=t('fdw_help')),
        ]),
        put_input('reg_strength', label=t('reg_strength'), type=FLOAT, value=get_val('reg_strength', 30.0), help_text=t('reg_help')),
        put_row([
            put_checkbox('normalize_opt', options=[{'label': t('enable_norm'), 'value': True}], value=[True] if get_val('normalize_opt', True) else [], help_text=t('norm_help')),
            put_checkbox('align_opt', options=[{'label': t('enable_align'), 'value': True}], value=[True] if get_val('align_opt', True) else [], help_text=t('align_help')),
            put_checkbox('stereo_link', options=[{'label': t('enable_link'), 'value': True}], value=[True] if get_val('stereo_link', False) else [], help_text=t('link_help')),
        ]),
        put_markdown(f"#### {t('sec_prot')}"),
        put_row([
            put_checkbox('exc_prot', options=[{'label': t('enable_exc'), 'value': True}], value=[True] if get_val('exc_prot', False) else []),
            put_input('exc_freq', label=t('exc_freq'), type=FLOAT, value=get_val('exc_freq', 25.0), help_text=t('exc_help')),
        ]),
        put_row([
            put_checkbox('hpf_enable', options=[{'label': t('hpf_enable'), 'value': True}], value=[True] if get_val('hpf_enable', False) else []),
            put_input('hpf_freq', label=t('hpf_freq'), type=FLOAT, value=get_val('hpf_freq', 20.0), help_text=t('hpf_freq_help')),
            put_select('hpf_slope', label=t('hpf_slope'), options=slope_opts, value=get_val('hpf_slope', 24)),
        ]),
        put_input('ir_window', label=t('ir_window'), type=FLOAT, value=get_val('ir_window', 250.0), help_text=t('ir_window_help'))
    ]

    tab_xo = [
        put_markdown(f"### ❌ {t('tab_xo')}"),
        put_grid([
            [put_input('xo1_f', label=f"XO 1 {t('xo_freq')}", type=FLOAT, value=get_val('xo1_f', None), help_text=t('xo_freq_help')), put_select('xo1_s', label=f"XO 1 {t('xo_slope')}", options=slope_opts, value=get_val('xo1_s', 12), help_text=t('xo_slope_help'))],
            [put_input('xo2_f', label=f"XO 2 {t('xo_freq')}", type=FLOAT, value=get_val('xo2_f', None), help_text=t('xo_freq_help')), put_select('xo2_s', label=f"XO 2 {t('xo_slope')}", options=slope_opts, value=get_val('xo2_s', 12), help_text=t('xo_slope_help'))],
            [put_input('xo3_f', label=f"XO 3 {t('xo_freq')}", type=FLOAT, value=get_val('xo3_f', None)), put_select('xo3_s', label=f"XO 3 {t('xo_slope')}", options=slope_opts, value=get_val('xo3_s', 12))],
            [put_input('xo4_f', label=f"XO 4 {t('xo_freq')}", type=FLOAT, value=get_val('xo4_f', None)), put_select('xo4_s', label=f"XO 4 {t('xo_slope')}", options=slope_opts, value=get_val('xo4_s', 12))],
            [put_input('xo5_f', label=f"XO 5 {t('xo_freq')}", type=FLOAT, value=get_val('xo5_f', None)), put_select('xo5_s', label=f"XO 5 {t('xo_slope')}", options=slope_opts, value=get_val('xo5_s', 12))],
        ])
    ]

    put_tabs([
        {'title': t('tab_files'), 'content': tab_files},
        {'title': t('tab_basic'), 'content': tab_basic},
        {'title': t('tab_target'), 'content': tab_target},
        {'title': t('tab_adv'), 'content': tab_adv},
        {'title': t('tab_xo'), 'content': tab_xo},
    ])

    put_markdown("---")

    def process_run():
        data = {
            'fs': pin.fs, 'taps': pin.taps, 'filter_type': pin.filter_type, 'mixed_freq': pin.mixed_freq,
            'gain': pin.gain, 'smoothing_type': pin.smooth_type, 'fdw_cycles': pin.fdw_cycles,
            'hc_mode': pin.hc_mode, 'hc_min': pin.hc_min, 'hc_max': pin.hc_max, 
            'max_boost': pin.max_boost, 'phase_limit': pin.phase_limit,
            'mag_correct': bool(pin.mag_correct), 'lvl_mode': pin.lvl_mode,
            'reg_strength': pin.reg_strength, 'normalize_opt': bool(pin.normalize_opt),
            'align_opt': bool(pin.align_opt), 'stereo_link': bool(pin.stereo_link),
            'exc_prot': bool(pin.exc_prot), 'exc_freq': pin.exc_freq,
            'hpf_enable': bool(pin.hpf_enable), 'hpf_freq': pin.hpf_freq, 'hpf_slope': pin.hpf_slope,
            'multi_rate_opt': bool(pin.multi_rate_opt), 'ir_window': pin.ir_window,
            'local_path_l': pin.local_path_l, 'local_path_r': pin.local_path_r,
            'fmt': pin.fmt, 'layout': pin.layout,
            'lvl_manual_db': pin.lvl_manual_db
        }
        for i in range(1, 6):
            data[f'xo{i}_f'] = pin[f'xo{i}_f']
            data[f'xo{i}_s'] = pin[f'xo{i}_s']

        data['lvl_algo'] = 'Median' 
        data['lvl_min'] = 500.0
        data['lvl_max'] = 2000.0

        save_config(data)
        
        f_l, m_l, p_l = None, None, None
        f_r, m_r, p_r = None, None, None
        
        if data['local_path_l']:
            f_l, m_l, p_l = parse_measurements_from_path(data['local_path_l'])
        if data['local_path_r']:
            f_r, m_r, p_r = parse_measurements_from_path(data['local_path_r'])
            
        if f_l is None:
            if pin.file_l:
                f_l, m_l, p_l = parse_measurements_from_bytes(pin.file_l['content'])
        
        if f_r is None:
            if pin.file_r:
                f_r, m_r, p_r = parse_measurements_from_bytes(pin.file_r['content'])

        if f_l is None or f_r is None:
            up_files = input_group("Lataa Mittaukset (Paths/Uploads tyhjät tai virheelliset)", [
                file_upload("Mittaus Vasen (L)", name='fl', accept='.txt'),
                file_upload("Mittaus Oikea (R)", name='fr', accept='.txt')
            ])
            f_l, m_l, p_l = parse_measurements_from_bytes(up_files['fl']['content'])
            f_r, m_r, p_r = parse_measurements_from_bytes(up_files['fr']['content'])
        
        if f_l is None or f_r is None:
            toast(t('err_missing_file'), color='red')
            return

        hc_f, hc_m = None, None
        if data['hc_mode'] == t('hc_mode_upload'):
            up_hc = input_group("Lataa Tavoite", [
                file_upload("Oma tavoite (.txt)", name='fhc', accept='.txt')
            ])
            hc_f, hc_m = io.BytesIO(up_hc['fhc']['content'])
            hc_f, hc_m = np.loadtxt(hc_f, comments=['#','*']).T[:2]
        else:
            hc_f, hc_m = get_house_curve_by_name(data['hc_mode'])

        put_processbar('bar'); put_scope('status_area'); update_status(t('stat_reading')); set_processbar('bar', 0.2)
        
        xos = []
        for i in range(1, 6):
            if data[f'xo{i}_f']:
                xos.append({'freq': data[f'xo{i}_f'], 'order': data[f'xo{i}_s']//6})
        
        hpf = {'enabled': data['hpf_enable'], 'freq': data['hpf_freq'], 'order': data['hpf_slope']//6} if data['hpf_freq'] else None
        
        target_rates = [data['fs']]
        if data['multi_rate_opt']:
            target_rates = [44100, 48000, 88200, 96000, 176400, 192000]
            
        zip_buffer = io.BytesIO()
        plot_l_html, plot_r_html = None, None
        
        ft_short = "Linear"
        if 'Min' in data['filter_type']: ft_short = "Minimum"
        elif 'Mixed' in data['filter_type']: ft_short = "Mixed"
        
        ts = datetime.now().strftime('%d%m%y_%H%M')
        
        l_st_sum, r_st_sum = None, None

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, fs_val in enumerate(target_rates):
                scale = fs_val / data['fs']
                taps_val = int(data['taps'] * scale)
                if taps_val % 2 != 0: taps_val += 1
                
                progress = 0.2 + (0.6 * (i / len(target_rates)))
                set_processbar('bar', progress)
                update_status(f"{t('msg_calc')} {fs_val}Hz ({taps_val} taps)...")
                
                args = (xos, 20, data['phase_limit'], data['hc_min'], data['hc_max'], hc_f, hc_m, fs_val, taps_val, FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf, data['mag_correct'], data['smoothing_type'], data['fdw_cycles'], 'Min' in data['filter_type'], data['filter_type'], data['lvl_mode'], data['lvl_min'], data['lvl_max'], data['lvl_manual_db'], data['lvl_algo'], data['normalize_opt'], data['reg_strength'], data['exc_prot'], data['exc_freq'], data['ir_window'], data['mixed_freq'])
                
                l_imp, l_min, l_max, l_st = dsp.generate_filter(f_l, m_l, p_l, *args)
                r_imp, r_min, r_max, r_st = dsp.generate_filter(f_r, m_r, p_r, *args)
                
                if data['align_opt']:
                    diff = np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp))
                    if diff > 0: r_imp = np.roll(r_imp, diff)
                    else: l_imp = np.roll(l_imp, -diff)
                    
                if data['stereo_link']:
                    scale = 0.891 / max(np.max(np.abs(l_imp)), np.max(np.abs(r_imp)))
                    l_imp *= scale; r_imp *= scale
                
                # --- EXPORT LOGIC (WAV or TXT) ---
                ext = "wav" if "WAV" in data['fmt'] else "txt"
                fn_l = f"L_corr_{ft_short}_{fs_val}Hz.{ext}"
                fn_r = f"R_corr_{ft_short}_{fs_val}Hz.{ext}"
                
                if "TXT" in data['fmt']:
                    # SigmaStudio compatible raw coefficients (one per line)
                    txt_l = io.BytesIO()
                    txt_r = io.BytesIO()
                    np.savetxt(txt_l, l_imp, fmt='%.10f')
                    np.savetxt(txt_r, r_imp, fmt='%.10f')
                    zf.writestr(fn_l, txt_l.getvalue())
                    zf.writestr(fn_r, txt_r.getvalue())
                else:
                    # Standard WAV
                    wav_l, wav_r = io.BytesIO(), io.BytesIO()
                    scipy.io.wavfile.write(wav_l, fs_val, l_imp.astype(np.float32))
                    scipy.io.wavfile.write(wav_r, fs_val, r_imp.astype(np.float32))
                    zf.writestr(fn_l, wav_l.getvalue())
                    zf.writestr(fn_r, wav_r.getvalue())
                
                if fs_val == data['fs']:
                    l_st['gd_min'] = l_min; l_st['gd_max'] = l_max
                    r_st['gd_min'] = r_min; r_st['gd_max'] = r_max
                    l_st_sum = l_st
                    r_st_sum = r_st
                    
                    zf.writestr("L_Plot.png", plots.generate_combined_plot_mpl(f_l, m_l, p_l, l_imp, fs_val, "Left", l_st))
                    zf.writestr("R_Plot.png", plots.generate_combined_plot_mpl(f_r, m_r, p_r, r_imp, fs_val, "Right", r_st))
                    zf.writestr("Summary.txt", plots.format_summary_content(data, l_st, r_st))
                    
                    split = data['mixed_freq'] if 'Mixed' in data['filter_type'] else None
                    plot_l_html = plots.generate_prediction_plot(f_l, m_l, p_l, l_imp, fs_val, "Left", None, l_st, split, t('zoom_hint'))
                    plot_r_html = plots.generate_prediction_plot(f_r, m_r, p_r, r_imp, fs_val, "Right", None, r_st, split, t('zoom_hint'))

            is_stereo = 'Stereo' in data['layout']
            yaml_content = "filters:\n"
            if is_stereo:
                fn_l = f"/home/camilladsp/coeffs/L_corr_{ft_short}_$samplerate$Hz.{ext}"
                fn_r = f"/home/camilladsp/coeffs/R_corr_{ft_short}_$samplerate$Hz.{ext}"
                yaml_content += f"  ir_l:\n    type: Convolution\n    parameters:\n      type: Wav\n      filename: {fn_l}\n"
                yaml_content += f"  ir_r:\n    type: Convolution\n    parameters:\n      type: Wav\n      filename: {fn_r}\n"
                yaml_content += "\npipeline:\n  - type: Filter\n    channel: 0\n    names:\n      - ir_l\n  - type: Filter\n    channel: 1\n    names:\n      - ir_r\n"
            else:
                fn_l = f"/home/camilladsp/coeffs/L_corr_{ft_short}_$samplerate$Hz.{ext}"
                fn_r = f"/home/camilladsp/coeffs/R_corr_{ft_short}_$samplerate$Hz.{ext}"
                yaml_content += f"  ir_l:\n    type: Convolution\n    parameters:\n      type: Wav\n      filename: {fn_l}\n"
                yaml_content += f"  ir_r:\n    type: Convolution\n    parameters:\n      type: Wav\n      filename: {fn_r}\n"
                yaml_content += "\npipeline:\n  - type: Filter\n    channel: 0\n    names:\n      - ir_l\n  - type: Filter\n    channel: 1\n    names:\n      - ir_r\n"
            zf.writestr("camilladsp.yml", yaml_content)

        update_status(t('stat_plot')); set_processbar('bar', 0.9)
        
        with use_scope('results', clear=True):
            put_success(t('done_msg'))
            
            put_markdown(f"### 📊 {t('rep_header')}")
            
            stats_data = [
                ['Suure', 'Vasen (L)', 'Oikea (R)'],
                ['Tavoitetaso', f"{l_st_sum['eff_target_db']:.2f} dB", f"{r_st_sum['eff_target_db']:.2f} dB"],
                ['Korjaus (Offset)', f"{l_st_sum['offset_db']:.2f} dB", f"{r_st_sum['offset_db']:.2f} dB"],
                ['Huippu (Peak)', f"{l_st_sum['peak_before_norm']:.2f} dB", f"{r_st_sum['peak_before_norm']:.2f} dB"],
                ['Normalisointi', "Kyllä (-1.0 dB)" if l_st_sum['normalized'] else "Ei", "Kyllä (-1.0 dB)" if r_st_sum['normalized'] else "Ei"],
                ['GD (Min)', f"{l_st_sum['gd_min']:.2f} ms", f"{r_st_sum['gd_min']:.2f} ms"],
                ['GD (Max)', f"{l_st_sum['gd_max']:.2f} ms", f"{r_st_sum['gd_max']:.2f} ms"]
            ]
            put_table(stats_data)
            
            with put_collapse("📋 Tarkemmat DSP-tiedot"):
                put_markdown(f"""
                - **Pituus:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
                - **Resoluutio:** {data['fs']/data['taps']:.2f} Hz
                - **Ikkuna:** {data['ir_window']} ms
                - **FDW:** {data['fdw_cycles']}
                - **Tavoite:** {data['hc_mode']} ({data['hc_min']}-{data['hc_max']} Hz)
                - **Tyyppi:** {data['filter_type']}
                """)
            
            put_markdown("---")
            
            put_tabs([
                {'title': t('tab_l'), 'content': put_html(plot_l_html)},
                {'title': t('tab_r'), 'content': put_html(plot_r_html)}
            ])
            
        fname = f"CamillaFIR_{ts}.zip"
        with open(fname, "wb") as f: f.write(zip_buffer.getvalue())
        put_text(f"{t('saved_local')} {fname}")
        set_processbar('bar', 1.0); update_status(t('stat_done'))

    put_button(f"▶️ {t('btn_run')}", onclick=process_run).style("font-size: 1.2em; font-weight: bold; width: 100%; margin-top: 20px;")

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
