import sys
import os
import io
import math
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
from models import FilterConfig
from textwrap import dedent
# IMPORT LOCAL MODULES
import camillafir_dsp as dsp
import camillafir_plot as plots
import models, camillafir_dsp, camillafir_plot
print("USING models.py      =", models.__file__)
print("USING camillafir_dsp =", camillafir_dsp.__file__)
print("USING camillafir_plot=", camillafir_plot.__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

CONFIG_FILE = 'config.json'
TRANS_FILE = 'translations.json'

VERSION = "v2.7.5"    #kaikki toimii edition
PROGRAM_NAME = "CamillaFIR"
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0

def scale_taps_with_fs(
    fs: int,
    base_fs: int = 44100,
    base_taps: int = 65536,
    allowed_taps=(
        512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288
   ),
) -> int:
    """Scale FIR taps with sample rate so that filter *time length* stays constant.

    Reference: 44.1 kHz -> 65 536 taps.
    For other rates, choose the next allowed taps value >= scaled target.
    """
    try:
        fs_i = int(fs)
        if fs_i <= 0:
            return int(base_taps)
        target = float(base_taps) * (float(fs_i) / float(base_fs))
        for t in allowed_taps:
            if int(t) >= target:
                return int(t)
        return int(allowed_taps[-1])
    except Exception:
        return int(base_taps)
    
def update_taps_auto_info(_=None):
    """
    UI helper: show Auto-taps mapping when multi-rate is enabled.
    Uses reference 44.1kHz -> 65536 taps (constant time-length).
    """
    # put_checkbox() -> pin['multi_rate_opt'] on lista valituista arvoista:
    # [] = off, [True] = on (tässä projektissa)
    try:
        mr = bool(pin['multi_rate_opt'])
    except Exception:
        mr = False

    for scope_name in ('taps_auto_info_scope_files', 'taps_auto_info_scope_basic'):
        with use_scope(scope_name, clear=True):
            if not mr:
                # Näytä jotain myös OFF-tilassa, jotta käyttäjä näkee että UI oikeasti päivittyy
                put_markdown(f"_{t('auto_taps_title')}: OFF_")
                continue

            rates = [44100, 48000, 88200, 96000, 176400, 192000]
            lines = [f"- **{r/1000:.1f} kHz** → **{scale_taps_with_fs(r)}** taps" for r in rates]

            put_markdown(
                f"### {t('auto_taps_title')}\n"
                f"{t('auto_taps_body')}\n\n"
                f"{t('auto_taps_ref')}\n\n"
                + "\n".join(lines)
            )


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

# clear() voi olla output.clear, mutta use_scope(clear=True) hoitaa sen

def update_lvl_ui(_=None):
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p('lvl_mode', 'Auto') or 'Auto')
        is_manual = ('Manual' in mode)

        # Sanity: lvl_min <= lvl_max
        vmin = float(_p('lvl_min', 500.0) or 500.0)
        vmax = float(_p('lvl_max', 2000.0) or 2000.0)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
            pin_update('lvl_min', value=vmin)
            pin_update('lvl_max', value=vmax)
        
        # lvl_manual_db: näytä AINA, mutta Auto-tilassa harmaana + ei-interaktiivinen
        with use_scope('lvl_manual_scope', clear=True):
            w = put_input(
                'lvl_manual_db',
                label=t('lvl_target_db'),
                type=FLOAT,
                value=float(_p('lvl_manual_db', 75.0) or 75.0),
                help_text=(
                    t('lvl_manual_help')
                    if is_manual
                    else (t('lvl_manual_help') + " (Auto ei käytä tätä arvoa)")
                )
            )
            if not is_manual:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")

    except Exception:
        pass

def apply_tdc_preset(name: str):
    """
    PyWebIO preset buttons for TDC knobs.
    Note: put_checkbox stores list values ([] / [True]) in this project.
    """
    presets = {
        "Safe":       {"enable": True, "strength": 35.0, "max_red": 6.0,  "slope": 3.0},
        "Normal":     {"enable": True, "strength": 50.0, "max_red": 9.0,  "slope": 6.0},
        "Aggressive": {"enable": True, "strength": 70.0, "max_red": 12.0, "slope": 0.0},
    }
    p = presets.get(name)
    if not p:
        return

    # enable_tdc is a checkbox -> list convention in this UI: [] / [True]
    pin_update("enable_tdc", value=[True] if p["enable"] else [])
    pin_update("tdc_strength", value=float(p["strength"]))
    pin_update("tdc_max_reduction_db", value=float(p["max_red"]))
    pin_update("tdc_slope_db_per_oct", value=float(p["slope"]))

    # small feedback
    toast(f"TDC preset applied: {name}", color="success", duration=1.5)


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
        'lvl_mode': 'Auto', 'lvl_algo': 'Median', 
        'lvl_manual_db': 75.0, 'lvl_min': 300.0, 'lvl_max': 3000.0,
        'normalize_opt': False, 'align_opt': True, 'multi_rate_opt': False,
        'reg_strength': 30.0, 'stereo_link': False, 
        'exc_prot': True, 'exc_freq': 20.0, 
        'low_bass_cut_hz': 40.0,    # alle tämän taajuuden sallitaan vain leikkaus (ei boostia)
        'hpf_enable': False, 'hpf_freq': 20.0, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12,
        'mixed_freq': 300.0, 'phase_limit': 1000.0,
        'phase_safe_2058': False, # TUPE-mode
        'ir_window': 500.0,       # Oikea ikkuna (Right)
        'ir_window_left': 50.0,  # Vasen ikkuna (Left) - UUSI
        'enable_tdc': True,       # TDC oletuksena päälle
        'tdc_strength': 50.0,     # TDC voimakkuus 50%
        'enable_afdw': True,      # Adaptiivinen FDW oletuksena päälle
        'max_cut_db': 15.0,              # max vaimennus (dB)
        'max_slope_db_per_oct': 12.0,    # max jyrkkyys (dB/okt), 0 = pois
        'df_smoothing': False,
        'comparison_mode': True,         # LOCK score/match analysis to 44.1k reference grid
        'tdc_max_reduction_db': 9.0,
        'tdc_slope_db_per_oct': 6.0,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                for k in ['mag_correct', 'normalize_opt', 'align_opt', 'multi_rate_opt', 'stereo_link', 'exc_prot', 'hpf_enable', 'df_smoothing', 'comparison_mode', 'phase_safe_2058']:
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
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]; taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; slope_opts = [6, 12, 18, 24, 36, 48]
    
#--- #1 Tiedostot
    
    tab_files = [
        put_markdown(f"### 📂 {t('tab_files')}"),
        put_file_upload('file_l', label=t('upload_l'), accept='.txt'), 
        put_input('local_path_l', label=t('path_l'), value=get_val('local_path_l', ''), help_text=t('path_help')),
        put_file_upload('file_r', label=t('upload_r'), accept='.txt'), 
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
    
#--- #2 Perusasetukset

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        
        # Rivi 1: Näytteenottotaajuus ja Tapit          
        put_row([
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')), 
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),


        # Auto-taps info (näkyy vain kun multi_rate_opt päällä)
        put_scope('taps_auto_info_scope_basic'),
        
        # Rivi 2: Suodintyyppi ja Mixed-taajuus
        put_row([
            put_radio('filter_type', label=t('filter_type'), 
                    options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), t('ft_asymmetric')], 
                    value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')), 
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help'))
        ]),
        
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        
        put_select('lvl_algo', label="Algo", options=['Median', 'Average'], value=get_val('lvl_algo', 'Median')),
        put_select('smoothing_type', label=t('smooth_type'), options=[
            {'label': t('smooth_std'), 'value': 'Standard'},
            {'label': t('smooth_psy'), 'value': 'Psychoacoustic'}
            ], value='Psychoacoustic', help_text=t('smooth_help')),
        # Rivi 3: Tilan valinta ja tavoitetaso (jaettu kahteen osaan luettavuuden vuoksi)
        # Level match range (help_text tulee oikeaan paikkaan suoraan kenttien alle)
        put_row([
            put_input(
                'lvl_min',
                label=t('lvl_min'),
                type=FLOAT,
                value=get_val('lvl_min', 500.0),
                help_text=t('lvl_min_help_auto')  # default Auto-tila
            ),
            put_input(
                'lvl_max',
                label=t('lvl_max'),
                type=FLOAT,
                value=get_val('lvl_max', 2000.0),
                help_text=t('lvl_max_help_auto')  # default Auto-tila
            ),
        ]),

        # lvl_mode + lvl_manual_db (näytetään aina, mutta Auto-tilassa lukittu)
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
        put_select('hc_mode', label=t('hc_mode'), options=hc_opts, value=get_val('hc_mode', t('hc_harman')), help_text=t('hc_mode_help')),
        
        
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
            put_input('max_cut_db', label="Max cut (dB)", type=FLOAT, value=get_val('max_cut_db', 15.0),
                      help_text="Rajoittaa vaimennuksen syvyyttä (esim. -15 dB maksimi)."),
            put_input('max_slope_db_per_oct', label="Max slope (dB/oct)", type=FLOAT, value=get_val('max_slope_db_per_oct', 12.0),
                      help_text="Rajoittaa gain-käyrän jyrkkyyttä per oktaavi. 0 = pois.")
        ]),
        put_input('trans_width', type=NUMBER, label="1/1 Transition Width (Hz)", value=100, help_text=t('trans_width')),
        put_markdown("---"),
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
              
        
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 1000.0), help_text=t('phase_limit_help')),
        
        put_checkbox(
            'phase_safe_2058',
            options=[{'label': t('phase_safe_2058'), 'value': True}],
            value=[True] if get_val('phase_safe_2058', False) else [],
            help_text=t('phase_safe_2058_help')
        ),
        
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

        # Afdw
        put_checkbox('enable_afdw', options=[{'label': t('enable_afdw'), 'value': True}], 
             value=[True] if get_val('enable_afdw', True) else [], help_text=t('afdw_help')),
        put_row([
            put_input('fdw_cycles', label=t('fdw'), type=FLOAT, value=get_val('fdw_cycles', 15.0), help_text=t('fdw_help'))
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
    pin_on_change('lvl_min', onchange=update_lvl_ui)
    pin_on_change('lvl_max', onchange=update_lvl_ui)
    update_lvl_ui()


    # Auto-taps UI updater: react when multi-rate toggles (tab_files) or basic changes
    pin_on_change('multi_rate_opt', onchange=update_taps_auto_info)
    pin_on_change('fs', onchange=update_taps_auto_info)
    pin_on_change('taps', onchange=update_taps_auto_info)
    update_taps_auto_info()

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
            'mag_c_min', 'mag_c_max', 'max_boost','max_cut_db', 'max_slope_db_per_oct','phase_limit', 'phase_safe_2058', 'mag_correct', 
            'lvl_mode', 'reg_strength', 'normalize_opt', 'align_opt', 
            'stereo_link', 'exc_prot', 'exc_freq','low_bass_cut_hz', 'hpf_enable', 'hpf_freq', 
            'hpf_slope', 'multi_rate_opt', 'ir_window', 'ir_window_left', 
            'local_path_l', 'local_path_r', 'fmt', 'lvl_manual_db', 
            'lvl_min', 'lvl_max', 'lvl_algo', 'smoothing_type', 'fdw_cycles',
            'trans_width', 'smoothing_level','enable_tdc', 'tdc_strength', 'tdc_max_reduction_db',
            'tdc_slope_db_per_oct', 'enable_afdw', 'df_smoothing', 'comparison_mode'
        ]
        
        data = {k: pin[k] for k in p_keys}

       
        for k in ['mag_correct','normalize_opt','align_opt','multi_rate_opt','stereo_link','exc_prot','hpf_enable','df_smoothing','comparison_mode']:
            try:
                if isinstance(data.get(k, None), list):
                    data[k] = bool(data[k])
            except Exception:
                pass

        for i in range(1, 6): 
            data[f'xo{i}_f'] = pin[f'xo{i}_f']
            data[f'xo{i}_s'] = pin[f'xo{i}_s']
            data['max_cut_db'] = abs(float(data.get('max_cut_db', 15.0) or 15.0))
            data['max_slope_db_per_oct'] = max(0.0, float(data.get('max_slope_db_per_oct', 12.0) or 12.0))

            # --- Manual level default (estää None) ---
            data['lvl_manual_db'] = float(data.get('lvl_manual_db', 75.0) or 75.0)
        save_config(data)
        
        logger.info(f"UI: phase_safe_2058 = {bool(data.get('phase_safe_2058', False))}")


        # --- LOG: DF-based frequency smoothing (A/B toggle) ---
        try:
            df_on = bool(pin['df_smoothing'])
        except Exception:
            df_on = False
        logger.info(f"DF smoothing: {'ON' if df_on else 'OFF'}")

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
                
                # Multi-rate: scale taps automatically with sample rate.
                # Reference is always 44.1 kHz -> 65 536 taps (keeps time-length constant).
                # Single-rate: use the user's selected taps as-is.
                if data['multi_rate_opt']:
                    taps_v = scale_taps_with_fs(fs_v)
                    logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> 65536 taps)")
                else:
                    taps_v = int(data['taps'])
                update_status(f"Lasketaan {fs_v}Hz..."); set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))
                
                # --- KORJAUS ALKAA TÄSTÄ:
                # UI:n data mäpätään FilterConfig-kenttiin
                cfg = FilterConfig(
                    fs=fs_v,
                    num_taps=taps_v,
                    df_smoothing=bool(pin['df_smoothing']),
                    **({"comparison_mode": True} if hasattr(FilterConfig, "comparison_mode") else {}),
                    filter_type_str=data['filter_type'],
                    mixed_split_freq=data['mixed_freq'],
                    global_gain_db=data['gain'],
                    mag_c_min=data['mag_c_min'],
                    mag_c_max=data['mag_c_max'],
                    max_boost_db=data['max_boost'],
                    max_cut_db=data.get('max_cut_db', 15.0),
                    max_slope_db_per_oct=data.get('max_slope_db_per_oct', 12.0),
                    phase_limit=data['phase_limit'],
                    phase_safe_2058=bool(data.get('phase_safe_2058', False)),
                    enable_mag_correction=bool(data['mag_correct']),
                    lvl_mode=data['lvl_mode'],
                    reg_strength=float(data.get('reg_strength', 30.0)),
                    do_normalize=bool(data['normalize_opt']),
                    exc_prot=bool(data['exc_prot']),
                    exc_freq=data['exc_freq'],
                    low_bass_cut_hz=float(data.get('low_bass_cut_hz', 40.0) or 40.0),
                    ir_window_ms=data['ir_window'],
                    ir_window_ms_left=data.get('ir_window_left', 100.0),
                    enable_afdw=bool(pin.enable_afdw), 
                    enable_tdc=bool(pin.enable_tdc),   
                    tdc_strength=data.get('tdc_strength', 50.0),
                    tdc_max_reduction_db=float(pin['tdc_max_reduction_db']),
                    tdc_slope_db_per_oct=float(pin['tdc_slope_db_per_oct']),
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
                
                    # --- LOG: DF smoothing per fs ---
                if bool(pin['df_smoothing']):
                    try:
                        # sama logiikka kuin DSP:ssä
                        base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)
                        df_ref = 44100.0 / 65536.0
                        sigma_hz = base_sigma * df_ref

                        # arvioi nykyinen df Hz/bin
                        df_cur = (fs_v / cfg.num_taps)
                        sigma_bins = sigma_hz / df_cur if df_cur > 0 else base_sigma

                        logger.info(
                            f"{fs_v//1000} kHz → DF smoothing ON "
                            f"(sigma = {sigma_bins:.1f} bins ≈ {sigma_hz:.1f} Hz)"
                        )
                    except Exception:
                        logger.info(f"{fs_v//1000} kHz → DF smoothing ON")
                else:
                    logger.info(f"{fs_v//1000} kHz → DF smoothing OFF")

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
                # ==========================================================
                # FORCE UI to use comparison-grid stats (44.1 kHz locked)
                # ==========================================================
                if bool(data.get("comparison_mode", False)):
                    try:
                        l_st_f = plots._make_comparison_stats(l_st_f, 44100, 65536)
                        r_st_f = plots._make_comparison_stats(r_st_f, 44100, 65536)
                    except Exception as e:
                        logger.warning(f"Comparison-mode UI stats failed: {e}")


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

                # nimet
                sum_name = f"Summary_{ft_short}_{fs_v}Hz.txt"
                l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.html"
                r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.html"
                
                summary_content = plots.format_summary_content(data, l_st, r_st)

                # ==========================================================
                # DSP EFFECTIVE PARAMS (per fs) - helps A/B and reproducibility
                # ==========================================================
                try:
                    # Read toggles from pin safely (PyWebIO: checkbox -> list)
                    enable_afdw = bool(pin['enable_afdw']) if 'enable_afdw' in pin else bool(data.get('enable_afdw', False))
                    enable_tdc  = bool(pin['enable_tdc'])  if 'enable_tdc'  in pin else bool(data.get('enable_tdc', False))
                    tdc_strength = float(data.get('tdc_strength', 0.0) or 0.0)
                    fdw_cycles = float(data.get('fdw_cycles', 15.0) or 15.0)
                    # Derived FDW params (these match DSP-side intent; A-FDW runtime adapts, so this is "configured baseline")
                    fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
                    afdw_min = max(3.0, fdw_cycles / 3.0)
                    afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0

                    # DF smoothing (A/B) sigma estimate (only meaningful if you enabled df_smoothing patch earlier)
                    df_on = bool(pin['df_smoothing']) if 'df_smoothing' in pin else bool(data.get('df_smoothing', False))
                    df_ref = 44100.0 / 65536.0
                    base_sigma = 60 // (data.get('smoothing_level', 12) / 12 if (data.get('smoothing_level', 12) or 0) > 0 else 1)
                    sigma_hz = float(base_sigma) * df_ref
                    df_cur = (float(fs_v) / float(data.get('taps', 65536) or 65536))
                    sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

                    summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
                    summary_content += f"Sample rate: {int(fs_v)} Hz\n"

                    # FDW / A-FDW
                    if enable_afdw:
                        summary_content += "FDW mode: Adaptive (A-FDW)\n"
                        summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width ≈ {fdw_oct_width:.3f})\n"
                        summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width ≈ {afdw_min_oct_width:.3f})\n"
                        summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
                    else:
                        summary_content += "FDW mode: Fixed\n"
                        summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width ≈ {fdw_oct_width:.3f})\n"

                    # TDC
                    summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
                    if enable_tdc:
                        summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength/100.0:.3f})\n"

                    # DF smoothing (optional)
                    summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
                    if df_on:
                        summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins ≈ {sigma_hz:.2f} Hz\n"
                except Exception:
                    summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
                    summary_content += "Could not compute effective params (unexpected data/pin state).\n"

                
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
                            summary_content += f"\n=== HEADROOM MANAGEMENT ===\n"
                            summary_content += f"Peak Gain: {float(l_st.get('peak_gain_db', 0.0)):.2f} dB\n"
                            summary_content += f"Applied Headroom: {float(l_st.get('auto_headroom_db', 0.0)):.2f} dB (to prevent clipping)\n"
                
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
                yaml_content = generate_raspberry_yaml(
                    fs_v,
                    ft_short,
                    file_ts,
                    master_gain_db=float(data.get('gain', 0.0) or 0.0)
                )
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
                # Use the same analysis context as scoring/summary:
                # If comparison mode produced coherent cmp_* stats, prefer cmp_reflections.
                mode = str((st or {}).get('analysis_mode', 'native') or 'native').lower()
                if mode == "comparison":
                    events = st.get('cmp_reflections') or []
                else:
                    mode = str((st or {}).get('analysis_mode', 'native') or 'native').lower()
                if mode == "comparison":
                    events = st.get('cmp_reflections') or st.get('reflections') or []
                else:
                    events = st.get('reflections') or []
                table_rows = [['Type', 'Freq', 'Impact', 'Dist']]
                
                for ev in events:
                    gd_error = float(ev.get('gd_error', 0) or 0)
                    ev_type = str(ev.get('type', 'Event') or 'Event')
                    ev_freq = float(ev.get('freq', 0) or 0)
                    ev_dist = float(ev.get('dist', 0) or 0)
                    impact = "High" if gd_error > 5 else "Medium"
                    color = "#ff4b4b" if ev_type == "Resonance" else "#4bafff"
                    table_rows.append([
                        put_text(ev_type).style(f'color: {color}; font-weight: bold'),
                        f"{ev_freq} Hz",
                        put_text(impact).style(f'color: {"#ff4b4b" if impact=="High" else "#ccc"}'),
                        f"{ev_dist} m"
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
        update_status(t('stat_plot')); set_processbar('bar', 1.0)
    

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



def calculate_target_match(st):
    """Laskee kuinka hyvin korjattu vaste seuraa tavoitekäyrää (0-100%)."""
    if not st:
        return 0.0

    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if meas.size == 0 or target.size == 0 or filt.size == 0:
        return 0.0

    # Varmista samat pituudet
    n = min(meas.size, target.size, filt.size)
    meas, target, filt = meas[:n], target[:n], filt[:n]

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


def calculate_score(st, is_predicted=False):
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    # Match korjatulle tai alkuperäiselle sen mukaan mitä UI tarvitsee:
    # - is_predicted=False -> "Measured"
    # - is_predicted=True  -> "Filtered"
    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if meas.size == 0 or target.size == 0:
        return max(15.0, min(99.0, conf))

    n = min(meas.size, target.size, filt.size if filt.size else meas.size)
    meas, target = meas[:n], target[:n]
    filt = filt[:n] if filt.size else np.zeros(n, dtype=float)

    if is_predicted:
        diff = (meas + filt) - target
    else:
        diff = meas - target

    rms = float(np.sqrt(np.mean(diff * diff)))

    # Sigmoid-match kuten Summary
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    match_pct = float(np.clip(match_pct, 0.0, 100.0))

    # Acoustic Score = 60% match + 40% confidence
    score = 0.60 * match_pct + 0.40 * conf

    # Rangaistukset (pidä sun nykyinen logiikka)
    rt60 = float(st.get('rt60_val', 0.4) or 0.4)
    rt_penalty = min(30.0, max(0.0, (rt60 - 0.4) * 25.0)) if rt60 > 0 else 0.0

    events = st.get('cmp_reflections', st.get('reflections', [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(40.0, (len(events) * 4.0) * penalty_mult)

    final_score = score - rt_penalty - event_penalty
    return float(max(15.0, min(99.0, final_score)))


if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
