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
from pywebio.pin import put_select, put_input, pin_update, pin
from pywebio.output import put_scope, put_row, put_markdown, put_widget
from models import FilterConfig
from textwrap import dedent
# IMPORT LOCAL MODULES
import camillafir_dsp as dsp
import camillafir_plot as plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

CONFIG_FILE = 'config.json'
TRANS_FILE = 'translations.json'

VERSION = "v2.7.2"    #kaikki toimii edition
PROGRAM_NAME = "CamillaFIR"
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0

def get_resource_path(relative_path):
    """ Palauttaa polun resurssiin, oli se EXE-paketin sisällä tai kehityskoneella. """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
TRANS_FILE = get_resource_path('translations.json')

def parse_measurements_from_path(path):
    """Lukee mittausdatan paikallisesta tiedostopolusta (REW .txt export) robustisti."""
    try:
        if not path: return None, None, None
        
        # 1. Siivotaan polku (poistetaan lainausmerkit ja välilyönnit)
        p = path.strip().strip('"').strip("'")
        
        if not os.path.exists(p):
            logger.error(f"Tiedostoa ei löydy: {p}")
            return None, None, None
            
        # 2. Avataan tiedosto
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        freqs, mags, phases = [], [], []

        for line in lines:
            line = line.strip()
            
            # Ohitetaan kommentit ja tyhjät
            if not line or line.startswith(('*', '#', ';')):
                continue
            
            # Ohitetaan rivit, jotka eivät ala numerolla
            if not line[0].isdigit() and line[0] != '-':
                continue

            # --- ÄLYKÄS EROTTIMEN TUNNISTUS ---
            # Jos rivillä on sekä piste että pilkku (esim. "0.36, 41.8"), pilkku on erotin -> vaihdetaan väliksi
            if ',' in line and '.' in line:
                line = line.replace(',', ' ')
            else:
                # Jos rivillä on vain pilkkuja, ne ovat todennäköisesti desimaaleja (Suomi) -> vaihdetaan pisteeksi
                line = line.replace(',', '.')
            # ----------------------------------

            parts = line.split()

            if len(parts) >= 2:
                try:
                    f_val = float(parts[0])
                    m_val = float(parts[1])
                    p_val = float(parts[2]) if len(parts) > 2 else 0.0
                    
                    freqs.append(f_val)
                    mags.append(m_val)
                    phases.append(p_val)
                except ValueError:
                    continue 
        
        if len(freqs) == 0:
            logger.warning(f"Tiedostosta {p} ei löytynyt dataa.")
            return None, None, None
            
        return np.array(freqs), np.array(mags), np.array(phases)

    except Exception as e:
        logger.error(f"Kriittinen virhe polun luvussa ({path}): {e}")
        return None, None, None


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
    """Lukee mittausdatan (REW export) tavuista robustisti."""
    try:
        # Dekoodataan tavut tekstiksi
        content_str = file_content.decode('utf-8', errors='ignore')
        lines = content_str.split('\n')
        
        freqs, mags, phases = [], [], []
        
        for line in lines:
            line = line.strip()
            # Ohitetaan tyhjät rivit ja kommentit (* tai #)
            if not line or line.startswith(('*', '#')):
                continue
            
            # Yritetään tunnistaa onko rivi dataa (alkaa numerolla)
            if not line[0].isdigit() and line[0] != '-':
                continue
                
            # Korvataan pilkku pisteellä (Suomi-yhteensopivuus)
            line = line.replace(',', '.')
            
            # Pilkotaan välilyöntien/tabulaattorien perusteella
            parts = line.split()
            
            if len(parts) >= 3:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    p = float(parts[2])
                    
                    freqs.append(f)
                    mags.append(m)
                    phases.append(p)
                except ValueError:
                    continue
        
        if len(freqs) == 0:
            return None, None, None
            
        return np.array(freqs), np.array(mags), np.array(phases)

    except Exception as e:
        print(f"Error parsing file: {e}")
        return None, None, None

def update_lvl_ui(mode):
    # Päivitetään ohjetekstit dynaamisesti kielen ja tilan mukaan
    if mode == 'Manual':
        pin_update('lvl_min', help_text=t('lvl_min_help_manual'))
        pin_update('lvl_max', help_text=t('lvl_max_help_manual'))
    else:
        pin_update('lvl_min', help_text=t('lvl_min_help_auto'))
        pin_update('lvl_max', help_text=t('lvl_max_help_auto'))



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

def load_target_curve(file_content):
    """Lukee tavoitekäyrän tekstitiedostosta ja varmistaa järjestyksen."""
    try:
        content_str = file_content.decode('utf-8')
        lines = content_str.split('\n')
        freqs, mags = [], []
        for line in lines:
            # Poistetaan kommentit ja tyhjät
            line = line.split('#')[0].strip()
            if not line: continue
            
            # Tuetaan pilkkua ja pistettä, sekä tabulaattoria ja välilyöntiä
            parts = line.replace(',', '.').split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    # Estetään nollataajuudet ja negatiiviset taajuudet
                    if f > 0:
                        freqs.append(f)
                        mags.append(m)
                except ValueError:
                    continue

        if len(freqs) < 2:
            return None, None

        # --- TÄRKEÄ KORJAUS: LAJITTELU ---
        # Varmistetaan, että taajuudet ovat nousevassa järjestyksessä.
        # Jos eivät ole, np.interp tekee "sahalaitaa" tai monttuja.
        freqs = np.array(freqs)
        mags = np.array(mags)
        if np.mean(mags) > 30:
            mags -= np.mean(mags)

        # Lajittelu taajuuden mukaan (tärkeää interpoloinnille)
        sort_idx = np.argsort(freqs)
        
        return freqs[sort_idx], mags[sort_idx]
        
    except Exception as e:
        print(f"Error loading target curve: {e}")
        return None, None


def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Mono', 'fs': 44100, 'taps': 65536,
        'filter_type': 'Linear Phase', 'gain': 0.0, 
        'hc_mode': 'Harman (Standard +6dB)', 'mag_correct': True,
        'smoothing_type': 'smooth_psy', 'fdw_cycles': 15.0,
        'mag_c_min': 10.0, 'mag_c_max': 200.0, 'max_boost': 5.0,
        'lvl_mode': 'Automatic', 'lvl_algo': 'algo_median', 
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
        'ir_window_left': 0.0,  # Vasen ikkuna (Left) - UUSI
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
    
#--- #2 Perusasetukset

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        
        # Rivi 1: Näytteenottotaajuus ja Tapit          
        put_row([
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')), 
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),
        
        # Rivi 2: Suodintyyppi ja Mixed-taajuus
        put_row([
            put_radio('filter_type', label=t('filter_type'), 
                    options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), t('ft_asymmetric')], 
                    value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')), 
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help'))
        ]),
        
        # Yksittäinen kenttä välissä
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        
        put_select('lvl_algo', label="Algo", options=['Median', 'Average'], value='Median'),
        put_select('smoothing_type', label=t('smooth_type'), options=[
            {'label': t('smooth_std'), 'value': 'Standard'},
            {'label': t('smooth_psy'), 'value': 'Psychoacoustic'}
            ], value='Psychoacoustic', help_text=t('smooth_help')),
        # Rivi 3: Tilan valinta ja tavoitetaso (jaettu kahteen osaan luettavuuden vuoksi)
        put_row([
            put_select('lvl_mode', label=t('lvl_mode'), 
                    options=[
                        {'label': t('lvl_mode_auto'), 'value': 'Auto'},
                       # {'label': t('lvl_mode_manual'), 'value': 'Manual'},
                    ], 
                    value=get_val('lvl_mode', 'Auto')), # Poistettu onchange tästä
           # put_input('lvl_manual_db', label=t('lvl_target_db'), type=FLOAT, 
                  #  value=get_val('lvl_manual_db', 75.0), help_text=t('lvl_manual_help'))
        ]),

        # Rivi 4: Rajat (Min/Max) omalla rivillään
        put_row([
            put_input('lvl_min', 
                    label=t('lvl_min'), 
                    type=FLOAT, 
                    value=get_val('lvl_min', 500.0),
                    # Lisätty 'Auto' oletusarvoksi get_val-kutsuun:
                    help_text=t('lvl_min_help_auto') if get_val('lvl_mode', 'Auto') == 'Auto' else t('lvl_min_help_manual')),

            put_input('lvl_max', 
                    label=t('lvl_max'), 
                    type=FLOAT, 
                    value=get_val('lvl_max', 2500.0),
                    # Lisätty 'Auto' oletusarvoksi get_val-kutsuun:
                    help_text=t('lvl_max_help_auto') if get_val('lvl_mode', 'Auto') == 'Auto' else t('lvl_max_help_manual')),
        ])
            ]
#--- #3 Target
    
    tab_target = [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_select('hc_mode', label=t('hc_mode'), options=hc_opts, value=get_val('hc_mode', t('hc_harman')), help_text=t('hc_mode_help')),
        
        
        put_file_upload('hc_custom_file', label=t('hc_custom'), accept='.txt', help_text=t('hc_custom_help')),
        put_checkbox('mag_correct', options=[{'label': t('enable_corr'), 'value': True}], value=[True] if get_val('mag_correct', True) else []),
        
        put_row([
            put_input('mag_c_min', label=t('min_freq'), type=FLOAT, value=get_val('mag_c_min', 10.0), help_text=t('hc_range_help')), 
            put_input('mag_c_max', label=t('max_freq'), type=FLOAT, value=get_val('mag_c_max', 200.0), help_text=t('hc_range_help'))
        ]),
        
        
        put_select(('smoothing_level'), label=t('filter_smooth'),
                   options=[
            {'label': '1/1 Octave', 'value': 1},
            {'label': '1/3 Octave', 'value': 3},
            {'label': '1/6 Octave', 'value': 6},
            {'label': '1/12 Octave (Standard)', 'value': 12},
            {'label': '1/24 Octave (Fine)', 'value': 24},
            {'label': '1/48 Octave (Ultra)', 'value': 48},
            {'label': '1/96 Octave (HC)', 'value': 96},
            ], value=12,),
              
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=t('max_boost_help')),
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 1000.0), help_text=t('phase_limit_help')),
        put_input('trans_width', type=NUMBER, label="1/1 Transition Width (Hz)", value=100, help_text=t('trans_width'))
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

#        put_row([
#            put_input('mag_c_min', label=t('min_freq'), type=FLOAT, value=get_val('mag_c_min', 10.0), help_text=t('hc_range_help')), 
#            put_input('mag_c_max', label=t('max_freq'), type=FLOAT, value=get_val('mag_c_max', 200.0), help_text=t('hc_range_help'))
#        ]),
        put_row([
        #    put_select('smoothing_type', label=t('smooth_type'), options=[
        #    {'label': t('smooth_std'), 'value': 'Standard'},
        #    {'label': t('smooth_psy'), 'value': 'Psychoacoustic'}
        #    ], value='Psychoacoustic', help_text=t('smooth_help')),
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
    pin_on_change('lvl_mode', onchange=update_lvl_ui)
    put_markdown("---")

    
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
    
def process_run():
        # --- 1. ASETUSTEN KERÄYS ---
        p_keys = [
            'fs', 'taps', 'filter_type', 'mixed_freq', 'gain', 'hc_mode', 
            'mag_c_min', 'mag_c_max', 'max_boost', 'phase_limit', 'mag_correct', 
            'lvl_mode', 'reg_strength', 'normalize_opt', 'align_opt', 
            'stereo_link', 'exc_prot', 'exc_freq', 'hpf_enable', 'hpf_freq', 
            'hpf_slope', 'multi_rate_opt', 'ir_window', 'ir_window_left', 
            'local_path_l', 'local_path_r', 'fmt', 'lvl_manual_db', 
            'lvl_min', 'lvl_max', 'lvl_algo', 'smoothing_type', 'fdw_cycles',
            'trans_width', 'smoothing_level','enable_tdc', 'tdc_strength', 'enable_afdw'
        ]
        
        data = {k: pin[k] for k in p_keys}
        for i in range(1, 6): 
            data[f'xo{i}_f'] = pin[f'xo{i}_f']
            data[f'xo{i}_s'] = pin[f'xo{i}_s']
        save_config(data)

        # --- 2. MITTAUSTEN LATAUS ---
        l_st_sum, r_st_sum = None, None
        f_l, m_l, p_l = parse_measurements_from_path(data['local_path_l']) if data['local_path_l'] else (None, None, None)
        f_r, m_r, p_r = parse_measurements_from_path(data['local_path_r']) if data['local_path_r'] else (None, None, None)
        
        if f_l is None and pin.file_l: 
            f_l, m_l, p_l = parse_measurements_from_bytes(pin.file_l['content'])
        if f_r is None and pin.file_r: 
            f_r, m_r, p_r = parse_measurements_from_bytes(pin.file_r['content'])
        
        if f_l is None or f_r is None: 
            toast("Measurements missing! Check paths or upload files.", color='red')
            return
        
        # --- 3. HOUSE CURVE LATAUS ---
        # --- 3. HOUSE CURVE LATAUS ---
        hc_f, hc_m = None, None
        hc_source = "Preset"

        # Prioriteetti 1: Käyttäjän lataama tiedosto (Browser Upload)
        if pin.hc_custom_file:
            hc_f, hc_m = load_target_curve(pin.hc_custom_file['content'])
            hc_source = "Upload"
        
        # Prioriteetti 2: Paikallinen polku (Local Path)
        # Ladataan vain, jos Uploadia ei ollut JA polku on määritelty
        if hc_f is None and data.get('local_path_house'):
            # Käytetään samaa järeää jäsennintä kuin mittauksille
            try:
                hc_f, hc_m, _ = parse_measurements_from_path(data['local_path_house'])
                # Varmistetaan lajittelu myös tässä
                if hc_f is not None:
                    s_idx = np.argsort(hc_f)
                    hc_f, hc_m = hc_f[s_idx], hc_m[s_idx]
                    hc_source = "LocalFile"
            except:
                hc_f, hc_m = None, None

        # Prioriteetti 3: Preset (Valikko)
        # Jos edelliset epäonnistuivat tai niitä ei ollut
        if hc_f is None:
            # Jos käyttäjä valitsi "Custom" mutta ei ladannut tiedostoa -> Käytä Flattia (ettei tule virhettä)
            preset_name = data['hc_mode']
            if "Custom" in preset_name or "Lataa" in preset_name:
                preset_name = "Flat" 
            
            hc_f, hc_m = get_house_curve_by_name(preset_name)
            hc_source = f"Preset ({preset_name})"

        # Debug-tulostus konsoliin (valinnainen)
        print(f"Loaded Target Curve from: {hc_source}, Points: {len(hc_f) if hc_f is not None else 0}")
        
        put_processbar('bar'); put_scope('status_area'); update_status(t('stat_reading')); set_processbar('bar', 0.2)
        
        # --- 4. RAKENTEET ---
        xos = [{'freq': data[f'xo{i}_f'], 'order': data[f'xo{i}_s']//6} for i in range(1, 6) if data[f'xo{i}_f']]
        hpf = {'enabled': data['hpf_enable'], 'freq': data['hpf_freq'], 'order': data['hpf_slope']//6} if data['hpf_enable'] else None
        
        target_rates = [44100, 48000, 88200, 96000, 176400, 192000] if data['multi_rate_opt'] else [data['fs']]
        zip_buffer = io.BytesIO(); ts = datetime.now().strftime('%d%m%y_%H%M'); file_ts = datetime.now().strftime('%H%M_%d%m%y')
        ft_short = "Asymmetric" if "Asymmetric" in data['filter_type'] else ("Minimum" if "Min" in data['filter_type'] else ("Mixed" if "Mixed" in data['filter_type'] else "Linear"))
        split, zoom = data['mixed_freq'], t('zoom_hint'); l_st_f, r_st_f = None, None
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, fs_v in enumerate(target_rates):
                taps_v = int(data['taps'] * (fs_v / data['fs']))
                taps_v = taps_v + (taps_v % 2)
                update_status(f"Lasketaan {fs_v}Hz..."); set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))
                
                # --- KORJAUS ALKAA TÄSTÄ: Luodaan FilterConfig-olio ---
                # UI:n data mäpätään FilterConfig-kenttiin
                cfg = FilterConfig(
                    fs=fs_v,
                    num_taps=taps_v,
                    filter_type_str=data['filter_type'],
                    mixed_split_freq=data['mixed_freq'],
                    global_gain_db=data['gain'],
                    mag_c_min=data['mag_c_min'],
                    mag_c_max=data['mag_c_max'],
                    max_boost_db=data['max_boost'],
                    phase_limit=data['phase_limit'],
                    enable_mag_correction=bool(data['mag_correct']),
                    lvl_mode=data['lvl_mode'],
                    reg_strength=float(data.get('reg_strength', 30.0)),
                    do_normalize=bool(data['normalize_opt']),
                    exc_prot=bool(data['exc_prot']),
                    exc_freq=data['exc_freq'],
                    ir_window_ms=data['ir_window'],
                    ir_window_ms_left=data.get('ir_window_left', 100.0),
                    enable_afdw=bool(pin.enable_afdw), 
                    enable_tdc=bool(pin.enable_tdc),   
                    tdc_strength=data.get('tdc_strength', 50.0),
                    smoothing_type=data['smoothing_type'],
                    fdw_cycles=data['fdw_cycles'],
                    lvl_manual_db=data['lvl_manual_db'],
                    lvl_min=data['lvl_min'],
                    lvl_max=data['lvl_max'],
                    lvl_algo=data['lvl_algo'],
                    smoothing_level=int(pin.smoothing_level),
                    crossovers=xos,
                    hpf_settings=hpf,
                    house_freqs=hc_f,
                    house_mags=hc_m,
                    trans_width=data.get('trans_width', 100.0)
                )
                
                # Kutsutaan DSP:tä oliolla
                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)
                # --- KORJAUS PÄÄTTYY ---



                if data['align_opt']:
                    d_s = np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp))
                    if d_s > 0: r_imp = np.roll(r_imp, d_s)
                    else: l_imp = np.roll(l_imp, -d_s)
                
                if fs_v == data['fs']:
                    l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp

                # --- AUTO-ALIGN ---
                if 'delay_samples' in l_st and 'delay_samples' in r_st:
                    diff_samples = r_st['delay_samples'] - l_st['delay_samples']
                    delay_ms = round((diff_samples / fs_v) * 1000, 3)
                    distance_cm = round((delay_ms / 1000) * 34300, 2)
                    gain_diff = round(l_st['offset_db'] - r_st['offset_db'], 2)
                    l_st['auto_align'] = {'delay_ms': delay_ms, 'distance_cm': distance_cm, 'gain_diff_db': gain_diff}

                # Tallennus
                wav_l, wav_r = io.BytesIO(), io.BytesIO()
                scipy.io.wavfile.write(wav_l, fs_v, l_imp.astype(np.float32))
                scipy.io.wavfile.write(wav_r, fs_v, r_imp.astype(np.float32))
                zf.writestr(f"L_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_l.getvalue())
                zf.writestr(f"R_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_r.getvalue())

              #  if fs_v == data['fs']: l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp
                
                sum_name = f"Summary_{ft_short}_{fs_v}Hz.txt"
                l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.html"
                r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.html"
                
                summary_content = plots.format_summary_content(data, l_st, r_st)
                
                for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
                    if 'reflections' in st and st['reflections']:
                        summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
                        summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'Error (ms)':<12} {'Dist (m)':<10}\n"
                        summary_content += "-" * 50 + "\n"
                        for rev in st['reflections']:
                            summary_content += f"{rev['freq']:<10} {rev['type']:<12} {rev['gd_error']:<12} {rev['dist']:<10}\n"
                            summary_content += f"\n=== HEADROOM MANAGEMENT ===\n"
                            summary_content += f"Peak Gain: {l_st['peak_gain_db']:.2f} dB\n"
                            summary_content += f"Applied Headroom: {l_st['auto_headroom_db']:.2f} dB (to prevent clipping)\n"
                
                if 'auto_align' in l_st:
                    res = l_st['auto_align']
                    summary_content += f"\n=== AUTO-ALIGN ===\nDelay: {res['delay_ms']} ms\nDistance Diff: {res['distance_cm']} cm\nGain Diff: {res['gain_diff_db']} dB\n"

                zf.writestr(sum_name, summary_content)
                zf.writestr(l_dash_name, plots.generate_prediction_plot(
                    f_l, m_l, p_l, l_imp, fs_v, "Left", None, l_st, data['mixed_freq'], "low"
                ))
                zf.writestr(r_dash_name, plots.generate_prediction_plot(
                    f_r, m_r, p_r, r_imp, fs_v, "Right", None, r_st, data['mixed_freq'], "low"
                ))
                hlc_cfg = generate_hlc_config(fs_v, ft_short, file_ts)
                zf.writestr(f"Config_{ft_short}_{fs_v}Hz.cfg", hlc_cfg)
                yaml_content = generate_raspberry_yaml(data['fs'], ft_short, file_ts)
                zf.writestr(f"camilladsp_{ft_short}_{fs_v}Hz.yml", yaml_content)

        
        fname = f"CamillaFIR_{ft_short}_{ts}.zip"
        try:
            with open(fname, "wb") as f: f.write(zip_buffer.getvalue())
            save_msg = f"Tallennettu: {os.path.abspath(fname)}"
        except: save_msg = "Tallennus epäonnistui."
        
        

        update_status(t('stat_plot')); set_processbar('bar', 1.0)
    
        with use_scope('results', clear=True):
            if l_st_f is None or r_st_f is None:
                put_error("Error: No results captured.")
                return

            put_success(t('done_msg'))

            # --- 1. LASKENNALLISET TULOKSET ---
            l_score_orig = calculate_score(l_st_f, is_predicted=False)
            l_score_pred = calculate_score(l_st_f, is_predicted=True)
            r_score_orig = calculate_score(r_st_f, is_predicted=False)
            r_score_pred = calculate_score(r_st_f, is_predicted=True)
            
            avg_orig = (l_score_orig + r_score_orig) / 2
            avg_pred = (l_score_pred + r_score_pred) / 2
            improvement = avg_pred - avg_orig

            # Target Match laskenta
            l_match = calculate_target_match(l_st_f)
            r_match = calculate_target_match(r_st_f)
            avg_match = (l_match + r_match) / 2

            # --- KAKSOISMITTARI-KORTTI ---
            score_color = "#4CAF50" if avg_pred > 75 else "#FFC107" if avg_pred > 50 else "#F44336"
            match_color = "#4bafff" # Sininen tavoitteen seurannalle

            put_row([
                put_html(f"""
                    <div style="background: #1e1e1e; padding: 25px; border-radius: 15px; border: 1px solid #333; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 15px;">
                            <div>
                                <h2 style="margin:0; color: {score_color};">Acoustic Score: {avg_pred:.0f}%</h2>
                                <p style="margin:5px 0 0 0; color: #888;">Improvement: <b style="color:#4CAF50;">+{improvement:.0f}%</b></p>
                            </div>
                            <div style="text-align: right; color: #666;">
                                Measured: {avg_orig:.0f}% | Filtered: {avg_pred:.0f}%
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h3 style="margin:0; color: {match_color};">Target Curve Match: {avg_match:.0f}%</h3>
                                <p style="margin:5px 0 0 0; color: #888;">Precision of the FIR correction</p>
                            </div>
                            <div style="width: 200px; background: #333; height: 10px; border-radius: 5px;">
                                <div style="width: {avg_match}%; background: {match_color}; height: 10px; border-radius: 5px;"></div>
                            </div>
                        </div>
                    </div>
                """)
            ])

            # --- 3. PÄÄTIEDOT TAULUKOSSA ---
            put_table([
                ['Speaker', 'L', 'R'],
                ['Target Level', f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"],
                ['Smart Scan Range', 
                 f"{l_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz",
                 f"{r_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz"],
                ['Offset to Meas.', f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"],
                ['Acoustic Confidence', f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"],
                ['Estimated RT60', f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"] # KORJATTU
            ])

            put_markdown(f"### 📊 {t('rep_header')}")
            with put_collapse("📋 DSP info"):
                put_markdown(dedent(f"""
                - **Lenght:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
                - **Resolution:** {data['fs']/data['taps']:.2f} Hz
                - **IR window:** {data['ir_window']} ms
                - **FDW:** {data['fdw_cycles']}
                - **House curve:** {data['hc_mode']} ({data['mag_c_min']}-{data['mag_c_max']} Hz)
                - **Filter type:** {data['filter_type']}
                - **Smoothing:** {data['lvl_algo']}
                """))

            # --- 4. AKUSTISTEN TAPAHTUMIEN ANALYYSI ---
            event_cols = []
            for side, st in [("Left", l_st_f), ("Right", r_st_f)]:
                events = st.get('reflections', [])
                table_rows = [['Type', 'Freq', 'Impact', 'Dist']]
                
                for ev in events:
                    impact = "High" if ev['gd_error'] > 5 else "Medium"
                    color = "#ff4b4b" if ev['type'] == "Resonance" else "#4bafff"
                    table_rows.append([
                        put_text(ev['type']).style(f'color: {color}; font-weight: bold'),
                        f"{ev['freq']} Hz",
                        put_text(impact).style(f'color: {"#ff4b4b" if impact=="High" else "#ccc"}'),
                        f"{ev['dist']} m"
                    ])
                
                event_cols.append(put_scope(f'ev_{side}', [
                    put_markdown(f"#### {side} Channel"),
                    put_table(table_rows) if events else put_text("No significant reflections detected.")
                ]))

            put_collapse("🔍 Detailed Acoustic Intelligence Analysis", [
                put_row(event_cols),
                put_markdown("---"),
                put_markdown(f"""
                **Analysis Summary:**
                * **Improvement:** The filter improves accuracy by **{improvement:.0f}** percentage points.
                * **Resonances:** Identified as peaks in Group Delay. These cause bass "boominess".
                * **Reflections:** Delayed arrivals that smear the stereo image and clarity.
                """)
            ])

            

            put_tabs([
                {'title': 'Left Channel', 'content': put_html(plots.generate_prediction_plot(f_l, m_l, p_l, l_imp_f, data['fs'], "Left", None, l_st_f, data['mixed_freq'], "low", create_full_html=False))},
                {'title': 'Right Channel', 'content': put_html(plots.generate_prediction_plot(f_r, m_r, p_r, r_imp_f, data['fs'], "Right", None, r_st_f, data['mixed_freq'], "low", create_full_html=False))}
            ]) 
            put_file(fname, zip_buffer.getvalue(), label="⬇️ DOWNLOAD FILTER ZIP")
        try:
            with open(fname, "wb") as f: f.write(zip_buffer.getvalue())
            save_msg = f"Tallennettu: {os.path.abspath(fname)}"
        except: save_msg = "Tallennus epäonnistui."
        update_status(t('stat_plot')); set_processbar('bar', 1.0)
    

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

# camillafir.py

def calculate_target_match(st):
    """Laskee kuinka hyvin korjattu vaste seuraa tavoitekäyrää (0-100%)."""
    if not st: return 0
    meas = np.array(st.get('measured_mags', []))
    target = np.array(st.get('target_mags', []))
    filt = np.array(st.get('filter_mags', []))
    
    if len(meas) == 0 or len(target) == 0: return 0
    
    # Korjattu vaste = Mittaus + Suodin
    final_error = np.std((meas + filt) - target)
    
    # 0dB virhe = 100%, 6dB virhe = 0%
    match_score = max(0, 100 - (final_error * 16.6))
    return min(100, match_score)

def calculate_score(st, is_predicted=False):
    if not st: return 0
    
    # 1. PERUSPISTEET: Vasteen tasaisuus
    meas = np.array(st.get('measured_mags', []))
    target = np.array(st.get('target_mags', []))
    filt = np.array(st.get('filter_mags', []))
    
    if len(meas) > 0 and len(target) > 0:
        error_orig = np.std(meas - target)
        error_pred = np.std((meas + filt) - target)
        error = error_pred if is_predicted else error_orig
        flatness_score = max(0, 100 - (error * 12))
    else:
        flatness_score = st.get('avg_confidence', 75)

    # 2. RANGAISTUKSET
    rt60 = st.get('rt60_val', 0.4)
    rt_penalty = min(30, max(0, (rt60 - 0.4) * 25)) if rt60 > 0 else 0
    
    events = st.get('reflections', [])
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(40, (len(events) * 4) * penalty_mult)
    
    final_score = flatness_score - rt_penalty - event_penalty
    return max(15, min(99, final_score))

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)


