import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.fft
import scipy.ndimage
import sys
import os
import io
import json
import locale
import zipfile 
from datetime import datetime

# --- MATPLOTLIB SETUP ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- GUI LIBRARY ---
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server, config

# --- PLOTLY SETUP ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONSTANTS ---
CONFIG_FILE = 'config.json'
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
VERSION = "v1.28.0 (PNG in ZIP)"
PROGRAM_NAME = "CamillaFIR"

# --- LANGUAGE DETECTION & TRANSLATIONS ---
def detect_language():
    try:
        sys_lang = locale.getdefaultlocale()[0]
        if sys_lang and 'fi' in sys_lang.lower():
            return 'fi'
    except:
        pass
    return 'en'

CURRENT_LANG = detect_language()

TRANSLATIONS = {
    'en': {
        'title': "CamillaFIR",
        'subtitle': "By VilhoValittu & GeminiPro",
        'guide_title': "❓ Guide: Sample Rate & Taps",
        'guide_rule': "Rule: Double Sample Rate -> Double Taps.",
        'guide_formula': "Resolution depends on ratio: Sample Rate / Taps.",
        'guide_desc': "High Sample Rate (192k) with low Taps (65k) = Poor bass correction.",
        'guide_rec': "Recommended:",
        'guide_note': "Note: Higher Taps = More latency.",
        'guide_ft_title': "❓ Guide: Linear vs. Minimum Phase",
        'guide_ft_lin_h': "1. Linear Phase (Default)",
        'guide_ft_lin_desc': "• Corrects timing/phase.\n• High Latency.\n• Heavy CPU.",
        'guide_ft_min_h': "2. Minimum Phase (Zero Latency)",
        'guide_ft_min_desc': "• Zero Latency (Gaming/TV).\n• No Pre-ringing.\n• Low CPU.",
        'guide_ft_rec': "Rec: Linear for Music. Minimum for Gaming/DSD.",
        'guide_fdw_title': "❓ Guide: FDW Cycles",
        'guide_fdw_desc': "Controls how much room sound is included.",
        'guide_fdw_auto': "NOTE: In Minimum Phase mode, this is AUTOMATICALLY set to 5 to prevent bass boosting artifacts.",
        'guide_fdw_low': "• 3-6: Aggressive (Nearfield, Dry).",
        'guide_fdw_mid': "• 15: Standard (Balanced, Linear Phase only).",
        'guide_fdw_high': "• 30-100: Gentle (Room EQ).",
        'guide_reg_title': "❓ Guide: Regularization (Dip Limit)",
        'guide_reg_desc': "Prevents over-correction of deep, narrow dips (nulls).",
        'guide_reg_why': "Why? Trying to boost a room null requires massive energy and sounds bad (ringing).",
        'guide_reg_how': "This setting detects sharp correction spikes and softens them automatically.",
        'guide_reg_rec': "Recommendation: 30-50%. (0% = Off, hard boost limit only).",
        'guide_lvl_title': "❓ Guide: Target Level Calculation",
        'guide_lvl_desc': "How the target curve is aligned to your measurement.",
        'guide_lvl_std': "• Automatic (Default): Calculates average volume from measurement (using 1/1 oct smoothing) within the defined range.",
        'guide_lvl_sub': "• Manual: Allows you to specify the exact target dB level (e.g. 75.0 dB). Useful if automatic detection fails or for gain staging.",
        'grp_files': "Input Files",
        'upload_l': "Measurement L (.txt)",
        'upload_r': "Measurement R (.txt)",
        'path_l': "OR Local Path L",
        'path_r': "OR Local Path R",
        'path_help': "Paste full path (e.g. C:\\Audio\\L.txt).",
        'ph_l': "Select Left .txt",
        'ph_r': "Select Right .txt",
        'fmt': "Format",
        'layout': "Layout",
        'layout_mono': "Mono",
        'layout_stereo': "Stereo",
        'fs': "Base Sample Rate",
        'taps': "Base Taps",
        'taps_help': "Taps for the selected Base FS. Will scale automatically for higher rates.",
        'filter_type': "Filter Type",
        'ft_linear': "Linear Phase",
        'ft_min': "Minimum Phase",
        'ft_help': "Linear=Delay, Min=Fast.",
        'gain': "Gain (dB)",
        'gain_help': "Output volume adjustment.",
        'smooth_type': "Smoothing",
        'smooth_std': "Standard 1/48",
        'smooth_psy': "Psychoacoustic",
        'fdw': "FDW Cycles",
        'fdw_help': "Linear: 15 (Default). Min Phase: Forced to 5.",
        'hc_mode': "House Curve",
        'hc_mode_upload': "Upload Custom...",
        'hc_custom': "Custom HC File",
        'hc_custom_help': "Required if Upload selected.",
        'corr_mode': "Correction",
        'enable_corr': "Enable Correction",
        'min_freq': "Correction Min (Hz)",
        'max_freq': "Correction Max (Hz)",
        'multi_rate': "Generate All Rates (44.1k - 192k)",
        'reg_strength': "Regularization (Dip Limit) %",
        'hc_harman': "Harman (Bass boost, treble roll-off)",
        'hc_bk': "B&K 1974 (Classic Hi-Fi)",
        'hc_flat': "Flat (Studio/Reference)",
        'hc_cinema': "Cinema X-Curve (Heavy roll-off)",
        'lvl_mode': "Level Match Mode",
        'lvl_auto': "Automatic",
        'lvl_man': "Manual dB",
        'lvl_target_db': "Target Level (dB)",
        'lvl_algo': "Calculation Algo",
        'algo_mean': "Average",
        'algo_median': "Median (Robust)",
        'lvl_min': "Calc Range Min (Hz)",
        'lvl_max': "Calc Range Max (Hz)",
        'lvl_help': "Freq range for Auto-calculation.",
        'norm_opt': "Normalize Output (Prevent Clipping)", 
        'enable_norm': "Normalize to -0.1 dB",
        'max_boost': "Max Boost (dB)",
        'boost_help': f"Limit: {MAX_SAFE_BOOST} dB",
        'hpf': "HPF (Subsonic)",
        'hpf_enable': "Enable HPF",
        'hpf_freq': "HPF Freq (Hz)",
        'hpf_freq_help': "e.g. 20Hz protection.",
        'hpf_slope': "HPF Slope",
        'xo_freq': "Freq (Hz)",
        'xo_slope': "Slope (dB/oct)",
        'xo_help': "Linearize existing crossovers.",
        'phase_lin': "(Phase Lin)",
        'grp_settings': "Settings",
        'stat_reading': "Reading...",
        'err_missing_file': "Error: Files missing.",
        'err_file_not_found': "File not found:",
        'err_parse': "Parse error.",
        'err_upload_custom': "Missing Custom HC file.",
        'err_inv_custom': "Invalid HC file.",
        'stat_calc': "Calculating...",
        'saved': "Saved:",
        'saved_zip': "Download All (ZIP):", 
        'saved_local': "Saved locally:",
        'saved_plot': "Saved Plot:",
        'stat_plot': "Plotting...",
        'err_plot': "Plot error.",
        'err_plot_fail': "Plot failed:",
        'stat_done': "Done!",
        'done_msg': "Analysis Complete.",
        'tab_l': "Left",
        'tab_r': "Right",
        'title_plot': "Frequency Response",
        'legend_orig_var': "Original (VAR)",
        'legend_pred': "Predicted (VAR)",
        'legend_target': "Target (House Curve)",
        'title_phase': "Phase Response",
        'legend_orig_sm': "Orig (Smooth)",
        'legend_pred_dr': "Pred (DelayRem)",
        'title_filt': "Filter Response (Correction)",
        'rep_header': "--- CALCULATION REPORT ---",
        'rep_corr_status': "Correction Enabled:",
        'rep_corr_range': "Correction Range:",
        'rep_lvl_mode': "Level Mode:",
        'rep_lvl_range': "Auto-Calc Range:",
        'rep_target_lvl': "Target Level:",
        'rep_offset': "Applied Offset:",
        'rep_norm': "Normalization:",
        'rep_peak': "Peak before norm:",
        'rep_fdw': "FDW Used:", 
        'rep_yes': "YES",
        'rep_no': "NO",
        'rep_norm_done': "DONE (Signal Reduced)",
        'rep_norm_skip': "NO (Signal was Safe)",
        'rep_disabled': "DISABLED",
        'rep_multi': "Multi-Rate Generation:",
        'rep_reg': "Regularization Strength:"
    },
    'fi': {
        'title': "CamillaFIR",
        'subtitle': "Tekijät: VilhoValittu & GeminiPro",
        'guide_title': "❓ Opas: Sample Rate & Taps",
        'guide_rule': "Sääntö: Sample Rate tuplaantuu -> Taps tuplaantuu.",
        'guide_formula': "Bassoresoluutio riippuu suhteesta: Sample Rate / Taps.",
        'guide_desc': "Korkea SR (192k) + matala Taps (65k) = Huono bassokorjaus.",
        'guide_rec': "Suositus:",
        'guide_note': "Huom: Suurempi Taps = enemmän viivettä.",
        'guide_ft_title': "❓ Opas: Linear vs. Minimum Phase",
        'guide_ft_lin_h': "1. Linear Phase (Oletus)",
        'guide_ft_lin_desc': "• Korjaa vaiheen.\n• Suuri viive.\n• Raskas.",
        'guide_ft_min_h': "2. Minimum Phase (Nollaviive)",
        'guide_ft_min_desc': "• Nollaviive (Peli/TV).\n• Ei Pre-ringing.\n• Kevyt.",
        'guide_ft_rec': "Suositus: Linear musiikille. Minimum peleille/DSD.",
        'guide_fdw_title': "❓ Opas: FDW Cycles",
        'guide_fdw_desc': "Kuinka paljon huonetta korjataan.",
        'guide_fdw_auto': "⚠️ HUOM: Minimum Phase -tilassa FDW pakotetaan automaattisesti arvoon 5. Tämä on välttämätöntä basson ylikorostumisen estämiseksi.",
        'guide_fdw_low': "• 3-6: Aggressiivinen (Kuiva, Lähikenttä).",
        'guide_fdw_mid': "• 15: Vakio (Vain Linear Phase).",
        'guide_fdw_high': "• 30-100: Hellävarainen (Huonekorjaus).",
        'guide_reg_title': "❓ Opas: Regularization (Dip Limit)",
        'guide_reg_desc': "Estää syvien ja kapeiden kuoppien (nollakohtien) ylikorjaamisen.",
        'guide_reg_why': "Miksi? Huonemoodin nollakohdan buustaaminen vaatii valtavasti energiaa ja kuulostaa huonolta (soiminen/ringing).",
        'guide_reg_how': "Tämä asetus tunnistaa piikikkäät korjausyritykset ja pehmentää niitä automaattisesti.",
        'guide_reg_rec': "Suositus: 30-50%. (0% = Pois, pelkkä Max Boost rajoittaa).",
        'guide_lvl_title': "❓ Opas: Miten korjauksen taso (Level Match) lasketaan?",
        'guide_lvl_desc': "Miten tavoitekäyrä (House Curve) kohdistetaan mittaukseen.",
        'guide_lvl_algo': "• Laskutapa: 'Mediaani' on suositus, koska se jättää huomiotta isot piikit/kuopat. 'Keskiarvo' ottaa kaiken mukaan.",
        'guide_lvl_std': "• Alue: Määrittää taajuusikkunan laskennalle (näkyy harmaana graafissa).",
        'grp_files': "Tiedostot",
        'upload_l': "Mittaus L (.txt)",
        'upload_r': "Mittaus R (.txt)",
        'path_l': "TAI Polku L",
        'path_r': "TAI Polku R",
        'path_help': "Liitä koko polku (esim. C:\\Mittaukset\\L.txt).",
        'ph_l': "Valitse L",
        'ph_r': "Valitse R",
        'fmt': "Muoto",
        'layout': "Asettelu",
        'layout_mono': "Mono",
        'layout_stereo': "Stereo",
        'fs': "Perus Sample Rate",
        'taps': "Perus Taps",
        'taps_help': "Taps valitulle SR:lle. Skaalautuu automaattisesti ylemmille taajuuksille.",
        'filter_type': "Suotimen Tyyppi",
        'ft_linear': "Linear Phase",
        'ft_min': "Minimum Phase",
        'ft_help': "Linear=Viive, Min=Nopea.",
        'gain': "Vahvistus (dB)",
        'gain_help': "Säätää tasoa. Käytä esim -3.0dB.",
        'smooth_type': "Silotus",
        'smooth_std': "Vakio 1/48",
        'smooth_psy': "Psykoakustinen",
        'fdw': "FDW Cycles",
        'fdw_help': "Linear: 15 (Vakio). Min Phase: Pakotettu 5.",
        'hc_mode': "Tavoitevaste",
        'hc_harman': "Harman (Vakio, basson korostus)",
        'hc_bk': "B&K 1974 (Loiva lasku, Hi-Fi)",
        'hc_flat': "Flat (Suora viiva, Studio)",
        'hc_cinema': "Cinema X-Curve (Leffateatteri)",
        'hc_mode_upload': "Lataa oma...",
        'hc_custom': "Oma HC tiedosto",
        'hc_custom_help': "Pakollinen jos valittu 'Lataa oma'.",
        'corr_mode': "Korjaus",
        'enable_corr': "Ota korjaus käyttöön",
        'min_freq': "Korjaus Alaraja (Hz)",
        'max_freq': "Korjaus Yläraja (Hz)",
        'multi_rate': "Generoi kaikki (44.1k - 192k)",
        'reg_strength': "Regularization (Dip Limit) %",
        'lvl_mode': "Tason sovitus (Mode)",
        'lvl_auto': "Automaattinen",
        'lvl_man': "Manuaalinen dB",
        'lvl_target_db': "Tavoitetaso (dB)",
        'lvl_algo': "Laskenta-algoritmi",
        'algo_mean': "Keskiarvo (Average)",
        'algo_median': "Mediaani (Vakaa)",
        'lvl_min': "Laskenta-alue Min (Hz)",
        'lvl_max': "Laskenta-alue Max (Hz)",
        'lvl_help': "Alue näkyy harmaana graafissa.",
        'norm_opt': "Normalisoi lähtö (Estä Särö)", 
        'enable_norm': "Normalisoi huippu -0.1 dB",
        'max_boost': "Max Boost (dB)",
        'boost_help': f"Raja: {MAX_SAFE_BOOST} dB",
        'hpf': "Ylipäästö (HPF)",
        'hpf_enable': "Käytä HPF",
        'hpf_freq': "HPF Taajuus (Hz)",
        'hpf_freq_help': "Suojaa bassoja (esim. 20Hz).",
        'hpf_slope': "HPF Jyrkkyys",
        'xo_freq': "Taajuus (Hz)",
        'xo_slope': "Jyrkkyys",
        'xo_help': "Oikaisee passiivijaon vaiheen.",
        'phase_lin': "(Vaihelin.)",
        'grp_settings': "Asetukset",
        'stat_reading': "Luetaan...",
        'err_missing_file': "Virhe: Tiedostot puuttuu.",
        'err_file_not_found': "Tiedostoa ei löydy:",
        'err_parse': "Lukuvirhe.",
        'err_upload_custom': "HC-tiedosto puuttuu.",
        'err_inv_custom': "Viallinen HC-tiedosto.",
        'stat_calc': "Lasketaan...",
        'saved': "Tallennettu:",
        'saved_zip': "Lataa Kaikki (ZIP):", 
        'saved_local': "Tallennettu kansioon:",
        'saved_plot': "Kuva:",
        'stat_plot': "Piirretään...",
        'err_plot': "Kuvaajavirhe.",
        'err_plot_fail': "Virhe piirrossa:",
        'stat_done': "Valmis!",
        'done_msg': "Analyysi valmis.",
        'tab_l': "Vasen",
        'tab_r': "Oikea",
        'title_plot': "Taajuusvaste",
        'legend_orig_var': "Alkuperäinen (VAR)",
        'legend_pred': "Ennuste (VAR)", 
        'legend_target': "Tavoite (House Curve)",
        'title_phase': "Vaihevaste",
        'legend_orig_sm': "Alkyp (Silotettu)",
        'legend_pred_dr': "Ennuste",
        'title_filt': "Suotimen Vaste (Filtteri)",
        'rep_header': "--- LASKENTARAPORTTI ---",
        'rep_corr_status': "Korjaus päällä:",
        'rep_corr_range': "Korjausalue:",
        'rep_lvl_mode': "Tason laskenta:",
        'rep_lvl_range': "Laskenta-alue (Auto):",
        'rep_target_lvl': "Tavoitetaso:",
        'rep_offset': "Tason muutos (Offset):",
        'rep_norm': "Normalisointi:",
        'rep_peak': "Huippu ennen norm:",
        'rep_fdw': "FDW Käytössä:", 
        'rep_yes': "KYLLÄ",
        'rep_no': "EI",
        'rep_norm_done': "TEHTY (Vaimennettiin)",
        'rep_norm_skip': "EI (Taso oli turvallinen)",
        'rep_disabled': "POIS PÄÄLTÄ",
        'rep_multi': "Multi-Rate Vienti:",
        'rep_reg': "Regularization (Dip):"
    }
}

def t(key):
    return TRANSLATIONS[CURRENT_LANG].get(key, key)

def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50;')

def put_guide():
    content_taps = [
        put_markdown(f"**{t('guide_rule')}**"),
        put_markdown(f"`{t('guide_formula')}`"),
        put_text(t('guide_desc')),
        put_markdown(f"**{t('guide_rec')}**"),
        put_markdown("""
        * 44.1 / 48 kHz: **65 536 taps**
        * 88.2 / 96 kHz: **131 072 taps**
        * 176.4 / 192 kHz: **262 144 taps**
        * 352.8 / 384 kHz: **524 288 taps**
        """),
        put_text(t('guide_note')).style('font-style: italic;')
    ]
    put_collapse(t('guide_title'), content_taps)

    content_ft = [
        put_markdown(f"**{t('guide_ft_lin_h')}**"),
        put_text(t('guide_ft_lin_desc')),
        put_markdown("---"),
        put_markdown(f"**{t('guide_ft_min_h')}**"),
        put_text(t('guide_ft_min_desc')),
        put_markdown("---"),
        put_markdown(f"_{t('guide_ft_rec')}_").style('font-weight: bold;')
    ]
    put_collapse(t('guide_ft_title'), content_ft)

    content_fdw = [
        put_text(t('guide_fdw_desc')),
        put_markdown(f"**{t('guide_fdw_auto')}**").style('color: #FF5722; font-weight: bold;'), 
        put_markdown(f"**{t('guide_fdw_mid')}**").style('color: #4CAF50;'), 
        put_text(t('guide_fdw_low')),
        put_text(t('guide_fdw_high'))
    ]
    put_collapse(t('guide_fdw_title'), content_fdw)
    
    content_reg = [
        put_text(t('guide_reg_desc')),
        put_markdown(f"**{t('guide_reg_why')}**"),
        put_text(t('guide_reg_how')),
        put_markdown(f"_{t('guide_reg_rec')}_").style('font-weight: bold;')
    ]
    put_collapse(t('guide_reg_title'), content_reg)

    content_lvl = [
        put_text(t('guide_lvl_desc')),
        put_text(t('guide_lvl_algo')),
        put_text(t('guide_lvl_std'))
    ]
    put_collapse(t('guide_lvl_title'), content_lvl)

def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Stereo (Single file)', 'fs': 44100, 'taps': 65536,
        'filter_type': t('ft_linear'),
        'gain': 0.0, 
        'hc_mode': t('hc_harman'),
        'mag_correct': [t('enable_corr')],
        'smoothing_type': t('smooth_psy'),
        'fdw_cycles': 15,
        'hc_min': 10, 
        'hc_max': 200, 
        'max_boost': 5.0,
        'lvl_mode': t('lvl_auto'),
        'lvl_algo': t('algo_median'), 
        'lvl_manual_db': 75.0,
        'lvl_min': 500, 'lvl_max': 2000,
        'normalize_opt': [t('enable_norm')],
        'multi_rate_opt': [],
        'reg_strength': 30.0,
        'hpf_enable': [], 'hpf_freq': None, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                if saved_conf.get('mag_correct') is None: saved_conf['mag_correct'] = []
                if saved_conf.get('hpf_enable') is None: saved_conf['hpf_enable'] = []
                if saved_conf.get('normalize_opt') is None: saved_conf['normalize_opt'] = [t('enable_norm')]
                if saved_conf.get('multi_rate_opt') is None: saved_conf['multi_rate_opt'] = []
                default_conf.update(saved_conf)
        except: pass
    return default_conf

def save_config(data):
    save_data = data.copy()
    for key in ['file_l', 'file_r', 'hc_custom_file']:
        if key in save_data: del save_data[key]
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(save_data, f, indent=4)
    except Exception as e: print(f"Failed to save config: {e}")

def get_house_curve_by_name(name):
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0])
    if name == t('hc_harman') or 'Harman' in name:
        mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    elif name == t('hc_bk') or 'B&K' in name:
        mags = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    elif name == t('hc_flat') or 'Flat' in name:
        mags = np.zeros_like(freqs)
    elif name == t('hc_cinema') or 'Cinema' in name:
        c_freqs = np.array([20, 2000, 4000, 8000, 16000, 20000])
        c_mags = np.array([0.0, 0.0, -3.0, -9.0, -15.0, -18.0])
        return c_freqs, c_mags
    else: 
        mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8, -4.0, -5.5, -6.0])
    return freqs, mags

def parse_measurements_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2] 
    except: return None, None, None

def parse_measurements_from_path(filepath):
    try:
        if not os.path.exists(filepath): return None, None, None
        data = np.loadtxt(filepath, comments=['#', '*'])
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2]
    except: return None, None, None

def parse_house_curve_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        return data[:, 0], data[:, 1]
    except: return None, None

def calculate_minimum_phase(mags_lin_fft):
    n_fft = (len(mags_lin_fft) - 1) * 2
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)
    return min_phase_rad[:len(mags_lin_fft)]

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    safe_cycles = max(cycles, 1.0)
    phase_u = np.unwrap(np.deg2rad(phases))
    oct_width = 2.0 / safe_cycles
    dummy_mags = np.zeros_like(freqs)
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    points_per_octave = 96
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    if num_points < 10: num_points = 10
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    window_size = int(points_per_octave * octave_fraction)
    if window_size < 1: window_size = 1
    window = np.ones(window_size) / window_size
    pad_len = window_size // 2
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    if pad_len > 0:
        sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
        sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
    else:
        sm_mags = np.convolve(m_padded, window, mode='same')
        sm_phases = np.convolve(p_padded, window, mode='same')
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def calculate_theoretical_phase(freq_axis, crossovers):
    if not crossovers: return np.zeros_like(freq_axis)
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

def find_zero_crossing_raw(freqs, phases, search_min=20, search_max=1000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) < 2: return search_min
    for i in range(len(p_sub) - 1):
        if np.sign(p_sub[i]) != np.sign(p_sub[i+1]):
            if abs(p_sub[i] - p_sub[i+1]) < 90.0: return f_sub[i]
    return f_sub[np.argmin(np.abs(p_sub))]

def find_closest_to_zero_raw(freqs, phases, search_min=800, search_max=2000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) == 0: return search_max
    return f_sub[np.argmin(np.abs(p_sub))]

def generate_filter(freqs, raw_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Standard', fdw_cycles=15, is_min_phase=False,
                    lvl_mode='Auto', l_match_min=500, l_match_max=2000, lvl_manual_db=75.0, lvl_algo='Average',
                    do_normalize=True, reg_strength=0.0):
    
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    nyquist = fs / 2.0
    freq_axis = np.linspace(0, nyquist, n_fft // 2 + 1)
    
    if smoothing_type == t('smooth_psy'):
        smoothed_mags = psychoacoustic_smoothing(freqs, raw_mags)
    else:
        smoothed_mags, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1/48.0)
    
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    
    target_mags = np.zeros_like(freq_axis)
    use_house_curve = (house_freqs is not None and house_mags is not None)
    calc_offset_db = 0.0 
    
    if use_house_curve:
        hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
        
        if lvl_mode == t('lvl_man') or lvl_mode == 'Manual dB':
            # Manual Mode
            a_start = max(l_match_min, 10.0)
            a_end = l_match_max
            mask_align = (freq_axis >= a_start) & (freq_axis <= a_end)
            if np.any(mask_align):
                avg_hc = np.mean(hc_interp[mask_align])
                shift = lvl_manual_db - avg_hc
                target_mags = hc_interp + shift
                calc_offset_db = shift 
            else:
                target_mags = hc_interp + (lvl_manual_db - np.mean(hc_interp))
        else:
            # Auto Mode
            mags_1_1, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1.0)
            meas_mags_1_1 = interpolate_response(freqs, mags_1_1, freq_axis)
            
            a_start = max(l_match_min, 10.0)
            a_end = l_match_max
            mask_align = (freq_axis >= a_start) & (freq_axis <= a_end)
            
            if np.any(mask_align):
                diffs = meas_mags_1_1[mask_align] - hc_interp[mask_align]
                if lvl_algo == t('algo_median') or lvl_algo == 'Median (Robust)':
                    calc_offset_db = np.median(diffs)
                else:
                    calc_offset_db = np.mean(diffs)
                target_mags = hc_interp + calc_offset_db
            else:
                idx_align = np.argmin(np.abs(freq_axis - mag_c_max))
                calc_offset_db = meas_mags[idx_align] - hc_interp[idx_align]
                target_mags = hc_interp + calc_offset_db
    
    eff_target_db = 0.0
    if use_house_curve:
         a_start = max(l_match_min, 10.0)
         a_end = l_match_max
         mask_calc = (freq_axis >= a_start) & (freq_axis <= a_end)
         if np.any(mask_calc):
             eff_target_db = np.mean(target_mags[mask_calc])
         else:
             eff_target_db = np.mean(target_mags)
    
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings and hpf_settings['enabled']:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h
        hpf_complex[0] = 0.0

    gain_linear = np.ones_like(freq_axis)
    
    raw_gain_db = target_mags - meas_mags
    smooth_gain_db = scipy.ndimage.gaussian_filter1d(raw_gain_db, sigma=50)
    reg_factor = reg_strength / 100.0
    
    for i, f in enumerate(freq_axis):
        if f > 0:
            g_db = 0.0
            if enable_mag_correction and use_house_curve and (mag_c_min <= f <= mag_c_max):
                
                req_g = raw_gain_db[i]
                
                if reg_factor > 0 and req_g > 0:
                    diff = req_g - smooth_gain_db[i]
                    if diff > 0: 
                        req_g = req_g - (diff * reg_factor)
                
                safe_max_boost = min(max_boost_db, MAX_SAFE_BOOST)
                g_db = np.clip(req_g, -15.0, safe_max_boost)
                
            g_db += global_gain_db
            gain_linear[i] = 10.0 ** (g_db / 20.0)

    stats = {
        'offset_db': calc_offset_db,
        'eff_target_db': eff_target_db, 
        'correction_enabled': enable_mag_correction,
        'target_mags': target_mags, 
        'freq_axis': freq_axis,     
        'has_target': use_house_curve,
        'l_match_min': l_match_min,
        'l_match_max': l_match_max,
        'peak_before_norm': 0.0,
        'normalized': False
    }

    if is_min_phase:
        total_mag_response = gain_linear * np.abs(hpf_complex)
        filt_min_phase_rad = calculate_minimum_phase(total_mag_response)
        final_complex = total_mag_response * np.exp(1j * filt_min_phase_rad)
        impulse = scipy.fft.irfft(final_complex, n=n_fft)
        window = np.ones(n_fft)
        fade_len = int(n_fft * 0.01) 
        if fade_len > 0:
            window[-fade_len:] = scipy.signal.windows.hann(2*fade_len)[fade_len:]
        impulse = impulse * window
    else:
        meas_min_phase_rad = calculate_minimum_phase(10**(meas_mags/20.0))
        meas_phase_rad_raw = np.deg2rad(interpolate_response(freqs, raw_phases, freq_axis))
        meas_phase_rad_unwrapped = np.unwrap(meas_phase_rad_raw)
        excess_phase_rad = meas_phase_rad_unwrapped - meas_min_phase_rad
        excess_phase_deg = np.rad2deg(excess_phase_rad)
        excess_phase_fdw_rad = apply_fdw_smoothing(freq_axis, excess_phase_deg, fdw_cycles)
        theoretical_xo_phase = calculate_theoretical_phase(freq_axis, crossovers)
        phase_corr_rad = np.zeros_like(freq_axis)
        limit_rad = np.deg2rad(fine_phase_limit)
        
        for i, f in enumerate(freq_axis):
            if f > 0:
                fine_correction = 0.0
                if phase_c_min <= f <= phase_c_max:
                    fine_correction = np.clip(-excess_phase_fdw_rad[i], -limit_rad, limit_rad)
                phase_corr_rad[i] = -theoretical_xo_phase[i] + fine_correction

        correction_complex = gain_linear * np.exp(1j * phase_corr_rad)
        final_complex = correction_complex * hpf_complex
        impulse = scipy.fft.irfft(final_complex, n=n_fft)
        window = scipy.signal.windows.tukey(n_fft, alpha=0.1)
        impulse = np.roll(impulse, n_fft // 2) * window

    max_peak = np.max(np.abs(impulse))
    stats['peak_before_norm'] = 20 * np.log10(max_peak + 1e-12)
    
    if do_normalize and max_peak > 0:
        target_amp = 0.9885
        if max_peak > target_amp:
            scaler = target_amp / max_peak
            impulse = impulse * scaler
            stats['normalized'] = True
    
    return impulse, 0.0, 0.0, stats

def save_summary(filename, settings, l_stats, r_stats):
    with open(filename, 'w') as f:
        f.write(f"=== {PROGRAM_NAME} - Filter Generation Summary ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- Settings ---\n")
        for key, val in settings.items():
            if 'file' not in key:
                f.write(f"{key}: {val}\n")
        f.write("\n--- Analysis (20Hz - 20kHz) ---\n")
        f.write(f"Left Channel GD: {l_stats['gd_min']:.2f} ms to {l_stats['gd_max']:.2f} ms\n")
        f.write(f"Right Channel GD: {r_stats['gd_min']:.2f} ms to {r_stats['gd_max']:.2f} ms\n")
        
        f.write(f"Left Target Level: {l_stats['eff_target_db']:.2f} dB\n")
        f.write(f"Right Target Level: {r_stats['eff_target_db']:.2f} dB\n")
        
        f.write(f"Left: Peak={l_stats['peak_before_norm']:.2f}dB, Norm={l_stats['normalized']}\n")
        f.write(f"Right: Peak={r_stats['peak_before_norm']:.2f}dB, Norm={r_stats['normalized']}\n")

def format_summary_content(settings, l_stats, r_stats):
    lines = []
    lines.append(f"=== {PROGRAM_NAME} - Filter Generation Summary ===")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("--- Settings ---")
    for key, val in settings.items():
        if 'file' not in key:
            lines.append(f"{key}: {val}")
    lines.append("\n--- Analysis (20Hz - 20kHz) ---")
    lines.append(f"Left Channel GD: {l_stats['gd_min']:.2f} ms to {l_stats['gd_max']:.2f} ms")
    lines.append(f"Right Channel GD: {r_stats['gd_min']:.2f} ms to {r_stats['gd_max']:.2f} ms")
    
    lines.append(f"Left Target Level: {l_stats['eff_target_db']:.2f} dB")
    lines.append(f"Right Target Level: {r_stats['eff_target_db']:.2f} dB")
    
    lines.append(f"Left: Peak={l_stats['peak_before_norm']:.2f}dB, Norm={l_stats['normalized']}")
    lines.append(f"Right: Peak={r_stats['peak_before_norm']:.2f}dB, Norm={r_stats['normalized']}")
    return "\n".join(lines)

# NEW: Matplotlib plotting function for PNG generation (replaces Plotly HTML in ZIP)
def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    try:
        # 1. Prediction Calculation
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_deg = np.rad2deg(np.unwrap(final_phases_rad))
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg)
        
        plot_orig_var = psychoacoustic_smoothing(orig_freqs, orig_mags)
        plot_pred_var = psychoacoustic_smoothing(orig_freqs, final_mags_plot)
        
        _, plot_phase_orig = apply_smoothing_std(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing_std(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        plot_phase_orig = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred = (plot_phase_pred + 180) % 360 - 180
        
        # 2. Filter Calculation
        w, h = scipy.signal.freqz(filt_ir, 1, worN=max(n_fft, 65536), fs=fs)
        filt_freqs = w
        filt_mags_db = 20 * np.log10(np.abs(h) + 1e-12)

        # 3. Create Matplotlib Figure (3 subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Row 1: Magnitude
        ax1.semilogx(orig_freqs, plot_orig_var, label='Original', color='blue', alpha=0.6)
        ax1.semilogx(orig_freqs, plot_pred_var, label='Predicted', color='orange')
        if target_stats and target_stats.get('has_target'):
            t_freqs = target_stats['freq_axis']
            t_mags = target_stats['target_mags']
            ax1.semilogx(t_freqs, t_mags, label='Target', color='green', linestyle='--')
        ax1.set_title(f"{title} - Magnitude")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        
        # Row 2: Phase
        ax2.semilogx(orig_freqs, plot_phase_orig, label='Orig Phase', color='blue', alpha=0.6, linestyle=':')
        ax2.semilogx(orig_freqs, plot_phase_pred, label='Pred Phase', color='orange')
        ax2.set_title("Phase")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_ylim(-180, 180)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()

        # Row 3: Filter
        ax3.semilogx(filt_freqs, filt_mags_db, label='Filter', color='red')
        ax3.set_title("Filter Response")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Gain (dB)")
        ax3.set_xlim(20, 20000)
        ax3.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"MPL Plot Error: {e}")
        return None

def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None, target_stats=None):
    try:
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_unwrap = np.unwrap(final_phases_rad) 
        final_phases_deg = np.rad2deg(final_phases_unwrap)
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg) 
        
        plot_orig_var = psychoacoustic_smoothing(orig_freqs, orig_mags)
        plot_pred_var = psychoacoustic_smoothing(orig_freqs, final_mags_plot)
        
        w, h = scipy.signal.freqz(filt_ir, 1, worN=max(n_fft, 65536), fs=fs)
        filt_freqs = w
        filt_mags_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        fig = make_subplots(rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.15, 
                            subplot_titles=(f"{title} - Magnitude (dB)", f"{title} - Phase (deg)", f"{title} - Filter (dB)"))

        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_orig_var, mode='lines', name='Original', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_pred_var, mode='lines', name='Predicted', line=dict(color='orange', width=2)), row=1, col=1)
        
        if target_stats and target_stats.get('has_target'):
            t_freqs = target_stats['freq_axis']
            t_mags = target_stats['target_mags']
            mask = t_freqs > 10
            fig.add_trace(go.Scatter(x=t_freqs[mask], y=t_mags[mask], mode='lines', name='Target', line=dict(color='green', dash='dash')), row=1, col=1)

        _, plot_phase_orig = apply_smoothing_std(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing_std(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        plot_phase_orig = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred = (plot_phase_pred + 180) % 360 - 180

        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_phase_orig, mode='lines', name='Orig Phase', line=dict(color='blue', width=0.5, dash='dot'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_phase_pred, mode='lines', name='Pred Phase', line=dict(color='orange', width=1), showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=filt_freqs, y=filt_mags_db, mode='lines', name='Filter', line=dict(color='red', width=1.5)), row=3, col=1)

        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=1, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=2, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", range=[np.log10(20), np.log10(20000)], row=3, col=1)
        
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="Deg", row=2, col=1, range=[-180, 180])
        fig.update_yaxes(title_text="dB", row=3, col=1)
        
        fig.update_layout(height=900, showlegend=True, template="plotly_white", width=1100) 
        
        return fig.to_html(include_plotlyjs=True, full_html=True) 
    except Exception as e:
        print(f"Plotly Error: {e}")
        return None

@config(theme="dark")
def main():
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")

    put_guide()

    defaults = load_config()
    
    hc_options = [t('hc_harman'), t('hc_bk'), t('hc_flat'), t('hc_cinema'), t('hc_mode_upload')]
    fs_options = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
    taps_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    val_corr = defaults.get('mag_correct')
    if saved_val_corr := defaults.get('mag_correct'): 
        val_corr = [t('enable_corr')] 
    else:
        val_corr = []
    
    val_hpf = defaults.get('hpf_enable')
    if val_hpf is None: val_hpf = []
    
    val_norm = defaults.get('normalize_opt')
    if not val_norm: val_norm = [t('enable_norm')]
    
    val_multi = defaults.get('multi_rate_opt')
    if not val_multi: val_multi = []

    inputs_list = [
        file_upload(t('upload_l'), name='file_l', accept='.txt', placeholder=t('ph_l')),
        input(t('path_l'), value=defaults.get('local_path_l', ''), name='local_path_l', help_text=t('path_help')),
        
        file_upload(t('upload_r'), name='file_r', accept='.txt', placeholder=t('ph_r')),
        input(t('path_r'), value=defaults.get('local_path_r', ''), name='local_path_r', help_text=t('path_help')),

        select(t('fmt'), options=['WAV', 'CSV'], value=defaults['fmt'], name='fmt'),
        radio(t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=defaults['layout'], name='layout'),
        select(t('fs'), options=fs_options, value=defaults['fs'], name='fs', type=NUMBER),
        select(t('taps'), options=taps_options, value=defaults['taps'], name='taps', type=NUMBER, help_text=t('taps_help')),
        
        checkbox(t('multi_rate'), options=[t('multi_rate')], value=val_multi, name='multi_rate_opt'),
        
        radio(t('filter_type'), options=[t('ft_linear'), t('ft_min')], value=defaults.get('filter_type', t('ft_linear')), name='filter_type', help_text=t('ft_help')),

        input(t('gain'), value=defaults['gain'], type=FLOAT, name='gain', help_text=t('gain_help')),
        
        select(t('smooth_type'), options=[t('smooth_std'), t('smooth_psy')], value=defaults.get('smoothing_type', t('smooth_psy')), name='smoothing_type'),
        input(t('fdw'), value=defaults.get('fdw_cycles', 15), type=FLOAT, name='fdw_cycles', help_text=t('fdw_help')),

        select(t('hc_mode'), options=hc_options, value=defaults.get('hc_mode', t('hc_harman')), name='hc_mode'),
        file_upload(t('hc_custom'), name='hc_custom_file', accept='.txt', help_text=t('hc_custom_help')),
        
        checkbox(t('corr_mode'), options=[t('enable_corr')], value=val_corr, name='mag_correct'),
        
        input(t('reg_strength'), value=defaults.get('reg_strength', 30.0), type=FLOAT, name='reg_strength'),
        
        input(t('max_boost'), value=defaults['max_boost'], type=FLOAT, name='max_boost', help_text=t('boost_help')),
        
        input(t('min_freq'), value=defaults['hc_min'], type=FLOAT, name='hc_min'),
        input(t('max_freq'), value=defaults['hc_max'], type=FLOAT, name='hc_max'),
        
        select(t('lvl_mode'), options=[t('lvl_auto'), t('lvl_man')], value=defaults.get('lvl_mode', t('lvl_auto')), name='lvl_mode'),
        
        select(t('lvl_algo'), options=[t('algo_mean'), t('algo_median')], value=defaults.get('lvl_algo', t('algo_median')), name='lvl_algo'),
        
        input(t('lvl_target_db'), value=defaults.get('lvl_manual_db', 75.0), type=FLOAT, name='lvl_manual_db'),
        
        input(t('lvl_min'), value=defaults.get('lvl_min', 500), type=FLOAT, name='lvl_min'),
        input(t('lvl_max'), value=defaults.get('lvl_max', 2000), type=FLOAT, name='lvl_max', help_text=t('lvl_help')),
        
        checkbox(t('norm_opt'), options=[t('enable_norm')], value=val_norm, name='normalize_opt'),
        
        checkbox(label=t('hpf'), options=[t('hpf_enable')], value=val_hpf, name='hpf_enable'),
        
        input(t('hpf_freq'), value=defaults['hpf_freq'], type=FLOAT, name='hpf_freq', placeholder="e.g. 20", help_text=t('hpf_freq_help')), 
        select(t('hpf_slope'), options=[6, 12, 18, 24, 36, 48], value=defaults['hpf_slope'], type=NUMBER, name='hpf_slope')
    ]

    slope_opts = [6, 12, 18, 24, 36, 48]
    for i in range(1, 6):
        label_txt = f"XO {i} {t('xo_freq')}"
        if i == 1: label_txt += f" {t('phase_lin')}"
        if i == 1:
            inputs_list.append(input(label_txt, value=defaults[f'xo{i}_f'], name=f'xo{i}_f', type=FLOAT, placeholder="Leave empty if unused", help_text=t('xo_help')))
        else:
            inputs_list.append(input(label_txt, value=defaults[f'xo{i}_f'], name=f'xo{i}_f', type=FLOAT, placeholder="Leave empty if unused"))
        inputs_list.append(select(f"XO {i} {t('xo_slope')}", options=slope_opts, value=defaults[f'xo{i}_s'], name=f'xo{i}_s'))

    data = input_group(t('grp_settings'), inputs_list)
    save_config(data)

    clear()
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    put_guide() 

    try:
        put_processbar('bar')
        put_scope('status_area')
        set_processbar('bar', 0.1)
        update_status(t('stat_reading'))

        freqs_l, mags_l, phases_l = None, None, None
        freqs_r, mags_r, phases_r = None, None, None

        if data['file_l']:
            freqs_l, mags_l, phases_l = parse_measurements_from_bytes(data['file_l']['content'])
        elif data['local_path_l']:
            freqs_l, mags_l, phases_l = parse_measurements_from_path(data['local_path_l'])
            if freqs_l is None:
                put_error(f"{t('err_file_not_found')} {data['local_path_l']}")
                return

        if data['file_r']:
            freqs_r, mags_r, phases_r = parse_measurements_from_bytes(data['file_r']['content'])
        elif data['local_path_r']:
            freqs_r, mags_r, phases_r = parse_measurements_from_path(data['local_path_r'])
            if freqs_r is None:
                put_error(f"{t('err_file_not_found')} {data['local_path_r']}")
                return

        if freqs_l is None or freqs_r is None:
            put_error(t('err_missing_file'))
            return

        hc_freqs, hc_mags = None, None
        
        hc_mode_str = data['hc_mode']
        if 'Harman' in hc_mode_str: hc_tag = "Harman"
        elif 'B&K' in hc_mode_str: hc_tag = "BK"
        elif 'Flat' in hc_mode_str: hc_tag = "Flat"
        elif 'Cinema' in hc_mode_str: hc_tag = "Cinema"
        else: hc_tag = "Custom"
        
        if data['hc_mode'] == t('hc_mode_upload'):
            if not data['hc_custom_file']:
                put_error(t('err_upload_custom'))
                return
            hc_freqs, hc_mags = parse_house_curve_from_bytes(data['hc_custom_file']['content'])
            if hc_freqs is None:
                put_error(t('err_inv_custom'))
                return
        else:
            hc_freqs, hc_mags = get_house_curve_by_name(data['hc_mode'])

        if hc_freqs is None:
            data['hc_min'], data['hc_max'] = 0, 0

        set_processbar('bar', 0.2)
        update_status(t('stat_calc'))
        
        do_mag_correct = bool(data['mag_correct']) 
        do_normalize = bool(data['normalize_opt'])
        
        do_multi_rate = bool(data['multi_rate_opt'])
        reg_strength = float(data.get('reg_strength', 30.0))
        
        crossovers = []
        for i in range(1, 6):
            if data[f'xo{i}_f']:
                crossovers.append({'freq': float(data[f'xo{i}_f']), 'order': int(data[f'xo{i}_s']) // 6})

        hpf_settings = None
        if bool(data['hpf_enable']) and data['hpf_freq']:
            hpf_settings = {'enabled': True, 'freq': data['hpf_freq'], 'order': data['hpf_slope'] // 6}

        c_min = (find_zero_crossing_raw(freqs_l, phases_l) + find_zero_crossing_raw(freqs_r, phases_r)) / 2
        c_max = (find_closest_to_zero_raw(freqs_l, phases_l) + find_closest_to_zero_raw(freqs_r, phases_r)) / 2
        
        smoothing_mode = data.get('smoothing_type', t('smooth_psy'))
        if t('smooth_psy') in smoothing_mode: smoothing_mode = t('smooth_psy')
        else: smoothing_mode = 'Standard'
        
        fdw = float(data.get('fdw_cycles', 15))
        is_min_phase = (data.get('filter_type') == t('ft_min'))
        
        if is_min_phase:
            fdw = 5.0
            phase_tag = "MinPhase"
        else:
            phase_tag = "Linear"
        
        lvl_mode = data.get('lvl_mode', t('lvl_auto'))
        lvl_algo = data.get('lvl_algo', t('algo_median'))
        match_min = float(data.get('lvl_min', 500))
        match_max = float(data.get('lvl_max', 2000))
        match_man_db = float(data.get('lvl_manual_db', 75.0))

        now = datetime.now()
        ts = now.strftime('%d%m%y_%H%M')

        target_rates = [data['fs']] 
        if do_multi_rate:
            target_rates = [44100, 48000, 88200, 96000, 176400, 192000]
            
        base_fs = float(data['fs'])
        base_taps = int(data['taps'])
        
        is_stereo = 'Stereo' in data['layout'] 
        
        zip_buffer = io.BytesIO()
        
        plot_l_html = None
        plot_r_html = None
        
        l_stats_final = None
        r_stats_final = None

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            for i, fs_val in enumerate(target_rates):
                scale_factor = fs_val / base_fs
                current_taps = int(base_taps * scale_factor)
                
                if current_taps % 2 != 0: current_taps += 1
                
                progress = 0.2 + (0.6 * (i / len(target_rates)))
                set_processbar('bar', progress)
                update_status(f"Calculating {fs_val}Hz ({current_taps} taps)...")

                l_imp, l_min, l_max, l_stats = generate_filter(
                    freqs_l, mags_l, phases_l, crossovers, c_min, c_max,
                    data['hc_min'], data['hc_max'], hc_freqs, hc_mags, fs_val, current_taps,
                    FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct,
                    smoothing_mode, fdw, is_min_phase, lvl_mode, match_min, match_max, match_man_db, lvl_algo,
                    do_normalize, reg_strength
                )
                r_imp, r_min, r_max, r_stats = generate_filter(
                    freqs_r, mags_r, phases_r, crossovers, c_min, c_max,
                    data['hc_min'], data['hc_max'], hc_freqs, hc_mags, fs_val, current_taps,
                    FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct,
                    smoothing_mode, fdw, is_min_phase, lvl_mode, match_min, match_max, match_man_db, lvl_algo,
                    do_normalize, reg_strength
                )
                
                if fs_val == base_fs or (len(target_rates) > 1 and i == 0):
                    l_stats_final = l_stats
                    r_stats_final = r_stats
                    l_stats_final['gd_min'] = l_min
                    l_stats_final['gd_max'] = l_max
                    r_stats_final['gd_min'] = r_min
                    r_stats_final['gd_max'] = r_max
                    
                    plot_l_html = generate_prediction_plot(freqs_l, mags_l, phases_l, l_imp, fs_val, t('tab_l'), target_stats=l_stats)
                    plot_r_html = generate_prediction_plot(freqs_r, mags_r, phases_r, r_imp, fs_val, t('tab_r'), target_stats=r_stats)
                    
                    img_mpl_l = generate_combined_plot_mpl(freqs_l, mags_l, phases_l, l_imp, fs_val, t('tab_l'), target_stats=l_stats)
                    img_mpl_r = generate_combined_plot_mpl(freqs_r, mags_r, phases_r, r_imp, fs_val, t('tab_r'), target_stats=r_stats)
                    
                    if img_mpl_l: zip_file.writestr(f"L_Plot_{hc_tag}_{phase_tag}_{fs_val}Hz_{ts}.png", img_mpl_l)
                    if img_mpl_r: zip_file.writestr(f"R_Plot_{hc_tag}_{phase_tag}_{fs_val}Hz_{ts}.png", img_mpl_r)

                ext = ".wav" if data['fmt'] == 'WAV' else ".csv"
                fn_base = f"corr_{hc_tag}_{phase_tag}_{fs_val}Hz_{ts}{ext}"
                
                if data['fmt'] == 'WAV':
                    if is_stereo:
                        wav_buf = io.BytesIO()
                        scipy.io.wavfile.write(wav_buf, fs_val, np.column_stack((l_imp, r_imp)).astype(np.float32))
                        zip_file.writestr(f"Stereo_{fn_base}", wav_buf.getvalue())
                    else:
                        wav_buf_l = io.BytesIO()
                        scipy.io.wavfile.write(wav_buf_l, fs_val, l_imp.astype(np.float32))
                        zip_file.writestr(f"L_{fn_base}", wav_buf_l.getvalue())
                        
                        wav_buf_r = io.BytesIO()
                        scipy.io.wavfile.write(wav_buf_r, fs_val, r_imp.astype(np.float32))
                        zip_file.writestr(f"R_{fn_base}", wav_buf_r.getvalue())
                else:
                    if is_stereo:
                        csv_buf = io.BytesIO()
                        np.savetxt(csv_buf, np.column_stack((l_imp, r_imp)), fmt='%.18f', delimiter=' ')
                        zip_file.writestr(f"Stereo_{fn_base}", csv_buf.getvalue())
                    else:
                        csv_buf_l = io.BytesIO()
                        np.savetxt(csv_buf_l, l_imp, fmt='%.18f')
                        zip_file.writestr(f"L_{fn_base}", csv_buf_l.getvalue())
                        
                        csv_buf_r = io.BytesIO()
                        np.savetxt(csv_buf_r, r_imp, fmt='%.18f')
                        zip_file.writestr(f"R_{fn_base}", csv_buf_r.getvalue())

            settings_dict = data.copy()
            settings_dict['Magnitude Correction'] = "Enabled" if do_mag_correct else "Disabled"
            settings_dict['Crossovers'] = str(crossovers)
            settings_dict['Phase Range'] = f"{c_min:.0f}-{c_max:.0f} Hz"
            settings_dict['FDW Cycles (Used)'] = fdw
            
            fn_sum = f"Summary_{hc_tag}_{phase_tag}_{ts}.txt"
            summary_content = format_summary_content(settings_dict, l_stats_final, r_stats_final)
            zip_file.writestr(fn_sum, summary_content)
            
            set_processbar('bar', 0.9)
            update_status(t('stat_plot'))
            
            put_markdown(f"### {t('rep_header')}")
            
            status_txt = t('rep_yes') if l_stats_final['correction_enabled'] else f"{t('rep_no')} {t('rep_disabled')}"
            style = 'color: green; font-weight: bold;' if l_stats_final['correction_enabled'] else 'color: red; font-weight: bold;'
            put_row([put_text(t('rep_corr_status')), put_text(status_txt).style(style)])
            
            put_row([put_text(t('rep_corr_range')), put_text(f"{data['hc_min']} Hz - {data['hc_max']} Hz")])
            
            if lvl_mode == t('lvl_auto'):
                put_row([put_text(t('rep_lvl_range')), put_text(f"{match_min} Hz - {match_max} Hz (Auto)")])
                tgt_l = l_stats_final['eff_target_db']
                tgt_r = r_stats_final['eff_target_db']
                put_row([put_text(t('rep_target_lvl')), put_text(f"L={tgt_l:.1f} dB, R={tgt_r:.1f} dB (Auto)")])
            else:
                put_row([put_text(t('rep_lvl_mode')), put_text(f"Manual ({match_man_db} dB)")])
                put_row([put_text(t('rep_target_lvl')), put_text(f"{match_man_db} dB (Manual)")])
            
            put_text(f"{t('rep_offset')} L={l_stats_final['offset_db']:.2f} dB, R={r_stats_final['offset_db']:.2f} dB")
            
            if do_normalize:
                if l_stats_final['normalized']:
                    norm_txt = f"{t('rep_norm_done')} (Peak was {l_stats_final['peak_before_norm']:.1f} dB)"
                else:
                    norm_txt = f"{t('rep_norm_skip')} (Peak {l_stats_final['peak_before_norm']:.1f} dB)"
            else:
                norm_txt = t('rep_disabled')
                
            put_row([put_text(t('rep_norm')), put_text(norm_txt)])
            
            put_row([put_text(t('rep_fdw')), put_text(str(fdw))])
            
            mr_txt = "YES (44.1k, 48k, 88.2k, 96k, 176.4k, 192k)" if do_multi_rate else f"NO ({base_fs} Hz only)"
            put_row([put_text(t('rep_multi')), put_text(mr_txt)])
            
            put_row([put_text(t('rep_reg')), put_text(f"{reg_strength}%")])

            if plot_l_html and plot_r_html:
                content_l = [put_html(plot_l_html)]
                content_r = [put_html(plot_r_html)]
                
                put_tabs([
                    {'title': t('tab_l'), 'content': put_column(content_l)},
                    {'title': t('tab_r'), 'content': put_column(content_r)}
                ])
            else:
                put_error(t('err_plot'))
                
        zip_filename = f"CamillaFIR_{hc_tag}_{phase_tag}_Result_{ts}.zip"
        
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        put_success(f"{t('saved_local')} {zip_filename}")
        
        put_file(zip_filename, zip_buffer.getvalue(), t('saved_zip'))
            
        set_processbar('bar', 1.0)
        update_status(t('stat_done'))
        put_markdown(f"### {t('done_msg')}")
        
    except Exception as e:
        put_error(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
