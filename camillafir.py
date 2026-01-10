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
from pywebio.session import set_env

# IMPORT LOCAL MODULES
import camillafir_dsp as dsp
import camillafir_plot as plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

CONFIG_FILE = 'config.json'
TRANS_FILE = 'translations.json'

VERSION = "v2.6.5"
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
    if key == 'zoom_hint':
        return "(Vinkki: Voit zoomata hiirellä kuvaajaa)" if lang == 'fi' else "(Hint: Use mouse to zoom)"
    if key == 'lvl_algo_help':
        return "Mediaani on suositeltu: se on immuuni huonemoodeille. Keskiarvo sopii kaiuttimen lähimittauksiin." if lang == 'fi' else "Median is recommended."
    return TRANSLATIONS.get(lang, TRANSLATIONS.get('en', {})).get(key, key)

def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50; margin-bottom: 10px;')

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

def get_house_curve_by_name(name):
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0])
    if 'Harman8' in name or '+8dB' in name:
        mags = np.array([8.0, 7.9, 7.8, 7.6, 7.3, 6.9, 6.3, 5.5, 4.5, 3.4, 1.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    elif 'Toole' in name:
        freqs = np.array([20, 63, 100, 200, 400, 1000, 2000, 4000, 10000, 20000])
        mags = np.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0, -4.0, -6.0])
    elif 'Flat' in name: mags = np.zeros_like(freqs)
    elif 'Cinema' in name:
        freqs = np.array([20, 2000, 4000, 8000, 16000, 20000])
        mags = np.array([0.0, 0.0, -3.0, -9.0, -15.0, -18.0])
    else: mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    return freqs, mags

def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Mono', 'fs': 44100, 'taps': 65536,
        'filter_type': 'Linear Phase', 'gain': 0.0, 
        'hc_mode': 'Harman (Standard +6dB)', 'mag_correct': True,
        'smoothing_type': 'Psychoacoustic', 'fdw_cycles': 15.0,
        'hc_min': 10.0, 'hc_max': 200.0, 'max_boost': 5.0,
        'lvl_mode': 'Automatic', 'lvl_algo': 'Median', 
        'lvl_manual_db': 75.0, 'lvl_min': 500.0, 'lvl_max': 2000.0,
        'normalize_opt': True, 'align_opt': True, 'multi_rate_opt': False,
        'reg_strength': 30.0, 'stereo_link': False, 
        'exc_prot': False, 'exc_freq': 25.0,
        'hpf_enable': False, 'hpf_freq': 20.0, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12,
        'mixed_freq': 300.0, 'phase_limit': 1000.0,
        'filter_type': 'Linear Phase',
        'ir_window': 500.0,       # Oikea ikkuna (Right)
        'ir_window_left': 100.0,  # Vasen ikkuna (Left) - UUSI
        'enable_tdc': True,       # TDC oletuksena päälle
        'tdc_strength': 50.0,     # TDC voimakkuus 50%
        'enable_afdw': True,      # Adaptiivinen FDW oletuksena päälle
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                for k in ['mag_correct', 'normalize_opt', 'align_opt', 'multi_rate_opt', 'stereo_link', 'exc_prot', 'hpf_enable']:
                    if k in saved and isinstance(saved[k], list): saved[k] = bool(saved[k])
                default_conf.update(saved)
        except: pass
    return default_conf

def save_config(data):
    try:
        clean_data = {k: v for k, v in data.items() if not k.startswith('file_')}
        with open(CONFIG_FILE, 'w') as f: json.dump(clean_data, f, indent=4)
    except: pass
def put_guide_section():
    # Tämä lista ohjaa, mitä oppaita näytetään ja missä järjestyksessä
    guides = [
        ('guide_taps', t('guide_taps_title')),
        ('guide_ft', t('guide_ft_title')),
        ('guide_sigma', t('guide_sigma_title')),
        ('guide_mix', t('guide_mix_title')),
        ('guide_fdw', t('guide_fdw_title')),
        ('guide_reg', t('guide_reg_title')),
        ('guide_lvl', t('guide_lvl_title')),
        ('guide_sl', t('guide_sl_title')),
        ('guide_ep', t('guide_ep_title')),
        # LISÄÄ NÄMÄ UUDET RIVIT:
        ('guide_asy', t('guide_asy_title')),
        ('guide_ai', t('guide_ai_title')),
        ('guide_summary', t('guide_summary_title')),
        ('guide_afdw', t('guide_afdw_title'))
    ]
    content = [put_collapse(t(g_key + '_title') if t(g_key + '_title') != (g_key + '_title') else g_title, [put_markdown(t(g_key + '_body') if t(g_key + '_body') != (g_key + '_body') else "Info text here")]) for g_key, g_title in guides]
    put_collapse("❓ CamillaFIR User Guides", content)

@config(theme="dark")
def main():
    set_env(output_max_width='1850px') 
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    put_guide_section(); put_markdown("---")
    d = load_config(); get_val = lambda k, def_v: d.get(k, def_v)
    hc_opts = [t('hc_harman'), t('hc_harman8'), t('hc_toole'), t('hc_bk'), t('hc_flat'), t('hc_cinema'), t('hc_mode_upload')]
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]; taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]; slope_opts = [6, 12, 18, 24, 36, 48]
    
#--- #1 Tiedostot
    
    tab_files = [
        put_markdown(f"### 📂 {t('tab_files')}"),
        put_file_upload('file_l', label=t('upload_l'), accept='.txt'), 
        put_input('local_path_l', label=t('path_l'), value=get_val('local_path_l', ''), help_text=t('path_help')),
        put_file_upload('file_r', label=t('upload_r'), accept='.txt'), 
        put_input('local_path_r', label=t('path_r'), value=get_val('local_path_r', ''), help_text=t('path_help')),
        put_select('fmt', label=t('fmt'), options=['WAV', 'TXT'], value=get_val('fmt', 'WAV'), help_text=t('fmt_help')),
        put_radio('layout', label=t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=get_val('layout', t('layout_stereo')), inline=True),
        put_checkbox('multi_rate_opt', options=[{'label': t('multi_rate'), 'value': True}], value=[True] if get_val('multi_rate_opt', False) else [], help_text=t('multi_rate_help'))
    ]
    
#--- #2 Perus

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        put_row([
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')), 
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),
        put_row([
            put_radio('filter_type', label=t('filter_type'), 
                      options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), 'Asymmetric Linear'], 
                      value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')), 
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help'))
        ]),
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        put_row([
            put_select('lvl_mode', label=t('lvl_mode'), options=[t('lvl_auto'), t('lvl_man')], value=get_val('lvl_mode', t('lvl_auto')), help_text=t('lvl_mode_help')), 
            put_select('lvl_algo', label=t('lvl_algo'), options=[t('algo_mean'), t('algo_median')], value=get_val('lvl_algo', t('algo_median')), help_text=t('lvl_algo_help'))
        ]),
        put_row([
            put_input('lvl_min', label=t('lvl_min'), type=FLOAT, value=get_val('lvl_min', 500.0), help_text=t('lvl_help')), 
            put_input('lvl_max', label=t('lvl_max'), type=FLOAT, value=get_val('lvl_max', 2000.0), help_text=t('lvl_help'))
        ]),
        put_input('lvl_manual_db', label=t('lvl_target_db'), type=FLOAT, value=get_val('lvl_manual_db', 75.0), help_text=t('lvl_manual_help'))
    ]
    
#--- #3 Target
    
    tab_target = [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_select('hc_mode', label=t('hc_mode'), options=hc_opts, value=get_val('hc_mode', t('hc_harman')), help_text=t('hc_mode_help')),
        
        
        put_file_upload('hc_custom_file', label=t('hc_custom'), accept='.txt', help_text=t('hc_custom_help')),
        put_checkbox('mag_correct', options=[{'label': t('enable_corr'), 'value': True}], value=[True] if get_val('mag_correct', True) else []),
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=t('max_boost_help')),
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 12000.0), help_text=t('phase_limit_help'))
        
    ]
#--- #4 Edistyneet
    tab_adv = [
        put_markdown(f"### 🛠️ {t('tab_adv')}"),
        
        put_markdown("#### ⏱️ Asymmetric Linear -ikkunointi"),
        put_row([
            put_input('ir_window_left', label="Left Window (ms)", type=FLOAT, value=get_val('ir_window_left', 100.0), help_text=t('ir_matala')),
            put_input('ir_window', label="Right Window (ms)", type=FLOAT, value=get_val('ir_window', 500.0), help_text=t('ir_korkea'))
        ]),
        put_markdown("---"),

        put_row([
            put_input('hc_min', label=t('min_freq'), type=FLOAT, value=get_val('hc_min', 10.0), help_text=t('hc_range_help')), 
            put_input('hc_max', label=t('max_freq'), type=FLOAT, value=get_val('hc_max', 200.0), help_text=t('hc_range_help'))
        ]),
        put_row([
            put_select('smoothing_type', label=t('smooth_type'), options=[t('smooth_std'), t('smooth_psy')], value=get_val('smoothing_type', t('smooth_psy')), help_text=t('smooth_help')), 
            put_input('fdw_cycles', label=t('fdw'), type=FLOAT, value=get_val('fdw_cycles', 15.0), help_text=t('fdw_help'))
        ]),
        # TDC aka Trinov-mode
        put_markdown("---"),
        put_row([
            put_checkbox('enable_tdc', options=[{'label': t('enable_tdc'), 'value': True}], 
                         value=[True] if get_val('enable_tdc', False) else [], help_text=t('tdc_help')),
            put_input('tdc_strength', label=t('tdc_strength'), type=FLOAT, value=get_val('tdc_strength', 50.0))
        ]),
        put_markdown("---"),
        # Afdw
        put_checkbox('enable_afdw', options=[{'label': t('enable_afdw'), 'value': True}], 
             value=[True] if get_val('enable_afdw', True) else [], help_text=t('afdw_help')),
        put_input('reg_strength', label=t('reg_strength'), type=FLOAT, value=get_val('reg_strength', 30.0), help_text=t('reg_help')),
        put_markdown("---"),

        put_row([
            put_checkbox('normalize_opt', options=[{'label': t('enable_norm'), 'value': True}], value=[True] if get_val('normalize_opt', True) else [], help_text=t('norm_help')), 
            put_checkbox('align_opt', options=[{'label': t('enable_align'), 'value': True}], value=[True] if get_val('align_opt', True) else [], help_text=t('align_help')), 
            put_checkbox('stereo_link', options=[{'label': t('enable_link'), 'value': True}], value=[True] if get_val('stereo_link', False) else [], help_text=t('link_help'))
        ]),
        put_row([
            put_checkbox('exc_prot', options=[{'label': t('enable_exc'), 'value': True}], value=[True] if get_val('exc_prot', False) else [], help_text=t('exc_help')), 
            put_input('exc_freq', label=t('exc_freq'), type=FLOAT, value=get_val('exc_freq', 25.0), help_text=t('exc_help'))
        ]),
        put_row([
            put_checkbox('hpf_enable', options=[{'label': t('hpf_enable'), 'value': True}], value=[True] if get_val('hpf_enable', False) else []), 
            put_input('hpf_freq', label=t('hpf_freq'), type=FLOAT, value=get_val('hpf_freq', 20.0), help_text=t('hpf_freq_help')), 
            put_select('hpf_slope', label=t('hpf_slope'), options=slope_opts, value=get_val('hpf_slope', 24))
        ])
    ]

#--- #5 XO
    tab_xo = [
        put_markdown(f"### ❌ {t('tab_xo')}"), 
        put_grid([[
            put_input(f'xo{i}_f', label=f"XO {i} Hz", type=FLOAT, value=get_val(f'xo{i}_f', None), help_text=t('xo_freq_help')), 
            put_select(f'xo{i}_s', label="dB/oct", options=slope_opts, value=get_val(f'xo{i}_s', 12), help_text=t('xo_slope_help'))
        ] for i in range(1, 6)])
    ]

    # Piirretään välilehdet
    put_tabs([
        {'title': t('tab_files'), 'content': tab_files}, 
        {'title': t('tab_basic'), 'content': tab_basic}, 
        {'title': t('tab_target'), 'content': tab_target}, 
        {'title': t('tab_adv'), 'content': tab_adv}, 
        {'title': t('tab_xo'), 'content': tab_xo}
    ])
    put_markdown("---")
    def process_run():
        # Varmista, että kukin näistä on täällä vain kerran!
        p_keys = [
            'fs', 'taps', 'filter_type', 'mixed_freq', 'gain', 'hc_mode', 
            'hc_min', 'hc_max', 'max_boost', 'phase_limit', 'mag_correct', 
            'lvl_mode', 'reg_strength', 'normalize_opt', 'align_opt', 
            'stereo_link', 'exc_prot', 'exc_freq', 'hpf_enable', 'hpf_freq', 
            'hpf_slope', 'multi_rate_opt', 'ir_window', 'ir_window_left', 
            'local_path_l', 'local_path_r', 'fmt', 'lvl_manual_db', 
            'lvl_min', 'lvl_max', 'lvl_algo', 'smoothing_type', 'fdw_cycles'
        ]
        data = {k: pin[k] for k in p_keys}
        for i in range(1, 6): data[f'xo{i}_f'] = pin[f'xo{i}_f']; data[f'xo{i}_s'] = pin[f'xo{i}_s']
        save_config(data)
        f_l, m_l, p_l = parse_measurements_from_path(data['local_path_l']) if data['local_path_l'] else (None, None, None)
        f_r, m_r, p_r = parse_measurements_from_path(data['local_path_r']) if data['local_path_r'] else (None, None, None)
        if f_l is None and pin.file_l: f_l, m_l, p_l = parse_measurements_from_bytes(pin.file_l['content'])
        if f_r is None and pin.file_r: f_r, m_r, p_r = parse_measurements_from_bytes(pin.file_r['content'])
        if f_l is None or f_r is None: toast("Mittaustiedostot puuttuvat!", color='red'); return
        hc_f, hc_m = get_house_curve_by_name(data['hc_mode'])
        put_processbar('bar'); put_scope('status_area'); update_status(t('stat_reading')); set_processbar('bar', 0.2)
        xos = [{'freq': data[f'xo{i}_f'], 'order': data[f'xo{i}_s']//6} for i in range(1, 6) if data[f'xo{i}_f']]
        hpf = {'enabled': data['hpf_enable'], 'freq': data['hpf_freq'], 'order': data['hpf_slope']//6} if data['hpf_enable'] else None
        target_rates = [44100, 48000, 88200, 96000, 176400, 192000] if data['multi_rate_opt'] else [data['fs']]
        zip_buffer = io.BytesIO(); ts = datetime.now().strftime('%d%m%y_%H%M'); file_ts = datetime.now().strftime('%H%M_%d%m%y')
        ft_short = "Asymmetric" if "Asymmetric" in data['filter_type'] else ("Minimum" if "Min" in data['filter_type'] else ("Mixed" if "Mixed" in data['filter_type'] else "Linear"))
        split, zoom = data['mixed_freq'], t('zoom_hint'); l_st_f, r_st_f = None, None
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, fs_v in enumerate(target_rates):
                taps_v = int(data['taps'] * (fs_v / data['fs'])); taps_v = taps_v + (taps_v % 2)
                update_status(f"Lasketaan {fs_v}Hz..."); set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))
                # args-tuple: Nyt 34 arvoa + f_l, m_l, p_l = 37 yhteensä
                args = (
                    xos, 20, data['phase_limit'], data['hc_min'], data['hc_max'], 
                    hc_f, hc_m, fs_v, taps_v, FINE_TUNE_LIMIT, 
                    data['max_boost'], data['gain'], hpf, data['mag_correct'], 
                    data['smoothing_type'], data['fdw_cycles'], 'Min' in ft_short, 
                    data['filter_type'], data['lvl_mode'], data['lvl_min'], 
                    data['lvl_max'], data['lvl_manual_db'], data['lvl_algo'], 
                    data['normalize_opt'], data['reg_strength'], data['exc_prot'], 
                    data['exc_freq'], data['ir_window'], data['mixed_freq'], 
                    data['ir_window_left'], data['phase_limit'], 
                    'enable_afdw' in data,
                    'enable_tdc' in data, # Tämä tarkistaa onko checkbox valittu
                    data.get('tdc_strength', 50.0) # .get estää KeyErrorin jos kenttää ei löydy
                )
                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, *args)
                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, *args)
                if data['align_opt']:
                    d_s = np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp))
                    if d_s > 0: r_imp = np.roll(r_imp, d_s)
                    else: l_imp = np.roll(l_imp, -d_s)
                wav_l, wav_r = io.BytesIO(), io.BytesIO(); scipy.io.wavfile.write(wav_l, fs_v, l_imp.astype(np.float32)); scipy.io.wavfile.write(wav_r, fs_v, r_imp.astype(np.float32))
                zf.writestr(f"L_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_l.getvalue()); zf.writestr(f"R_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_r.getvalue())
                if fs_v == data['fs']:
                    l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp
                    zf.writestr("Summary.txt", plots.format_summary_content(data, l_st, r_st))
                    zf.writestr("L_Dashboard.html", plots.generate_prediction_plot(f_l, m_l, p_l, l_imp, fs_v, "Left", None, l_st, split, zoom))
                    zf.writestr("R_Dashboard.html", plots.generate_prediction_plot(f_r, m_r, p_r, r_imp, fs_v, "Right", None, r_st, split, zoom))
                    zf.writestr("L_Plot.png", plots.generate_combined_plot_mpl(f_l, m_l, p_l, l_imp, fs_v, "Left", l_st))
                    zf.writestr("R_Plot.png", plots.generate_combined_plot_mpl(f_r, m_r, p_r, r_imp, fs_v, "Right", r_st))
                    yaml_content = generate_raspberry_yaml(data['fs'], ft_short, file_ts)
                    zf.writestr("camilladsp.yml", yaml_content)
                    fname = f"CamillaFIR_{ft_short}_{ts}.zip"
        try:
            with open(fname, "wb") as f: f.write(zip_buffer.getvalue())
            save_msg = f"Tallennettu: {os.path.abspath(fname)}"
        except: save_msg = "Tallennus epäonnistui."
        update_status(t('stat_plot')); set_processbar('bar', 1.0)
        with use_scope('results', clear=True):
            put_success(t('done_msg')); put_text(save_msg).style('font-style: italic; color: #888;')
            put_table([['Suure', 'L', 'R'], ['Tavoite', f"{l_st_f['eff_target_db']:.1f} dB", f"{r_st_f['eff_target_db']:.1f} dB"], ['Huippu', f"{l_st_f['peak_before_norm']:.1f} dB", f"{r_st_f['peak_before_norm']:.1f} dB"], ['Offset', f"{l_st_f['offset_db']:.1f} dB", f"{r_st_f['offset_db']:.1f} dB"], ['Luottamus', f"{l_st_f['avg_confidence']:.1f}%", f"{r_st_f['avg_confidence']:.1f}%"]])
            put_tabs([{'title': 'L Dashboard', 'content': put_html(plots.generate_prediction_plot(f_l, m_l, p_l, l_imp_f, data['fs'], "Left", None, l_st_f, split, zoom))}, {'title': 'R Dashboard', 'content': put_html(plots.generate_prediction_plot(f_r, m_r, p_r, r_imp_f, data['fs'], "Right", None, r_st_f, split, zoom))}])
            put_file(fname, zip_buffer.getvalue(), label="⬇️ ZIP")
    # Napin päivitys: Täysin puhdas teksti ilman taustaa tai kehyksiä
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
#snipet
def generate_raspberry_yaml(fs, ft_short, file_ts):

    import textwrap
    return textwrap.dedent(f"""\
        devices:
          samplerate: {int(fs)}
          chunksize: 4096
          queuesize: 10
          capture:
            type: Alsa
            channels: 2
            device: "hw:1,0"
            format: S32LE
          playback:
            type: Alsa
            channels: 2
            device: "hw:1,0"
            format: S32LE

        filters:
          ir_l:
            type: File
            filename: L_{ft_short}_$samplerate$Hz_{file_ts}.wav
            format: S32LE
            setting: 0
          ir_r:
            type: File
            filename: R_{ft_short}_$samplerate$Hz_{file_ts}.wav
            format: S32LE
            setting: 0

        pipeline:
          - type: Filter
            channel: 0
            names:
              - ir_l
          - type: Filter
            channel: 1
            names:
              - ir_r
    """).strip()

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
