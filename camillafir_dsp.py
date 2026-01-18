import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
import logging
logger = logging.getLogger("CamillaFIR.dsp")
from models import FilterConfig
from camillafir_leveling import compute_leveling
#CamillaFIR DSP Engine v1.0.6
#1.0.2 Fix comma mistake at HPF
#1.03 Fix at phase calculation that caused "spikes"
#1.04 All features works at different configurations
#1.05 Multiplier changes
#1.06 added no phase correction mode (2058-safe)

def apply_smart_tdc(
    freq_axis,
    target_mags,
    reflections,
    rt60_info,
    base_strength=0.5,
    max_total_reduction_db: float = 9.0,
    max_slope_db_per_oct: float = 0.0,
):
    """Temporal Decay Control (TDC)

    Idea: Instead of directly subtracting multiple overlapping kernels from the target
    (which can unintentionally stack into a deep, narrow notch), we accumulate a
    *reduction curve* and apply a safety brake:
      - hard cap max total reduction (dB)
      - optional slope limit (dB/oct) for smoothness
    """
    adjusted_target = np.copy(target_mags)
    tdc_reduction_db = np.zeros_like(adjusted_target)

    # rt60_info voi olla:
    #  - float (vanha käytös)
    #  - dict: {center_hz: rt60_s, ...} (uusi: kaistoittain)
    def rt60_at(freq_hz: float) -> float:
        # fallback
        default = 0.4
        if isinstance(rt60_info, (int, float)):
            v = float(rt60_info)
            return v if v > 0.1 else default
        if isinstance(rt60_info, dict) and rt60_info:
            # interpoloidaan log-taajuudessa kaistakeskuksien yli
            c = np.array(sorted(rt60_info.keys()), dtype=float)
            r = np.array([rt60_info[k] for k in c], dtype=float)
            mask = (c > 0) & (r > 0.05) & (r < 5.0)
            if np.count_nonzero(mask) < 2:
                # jos ei tarpeeksi kaistoja, yritä esim. mediaani
                vv = float(np.median(r[mask])) if np.count_nonzero(mask) else 0.0
                return vv if vv > 0.1 else default
            c = c[mask]; r = r[mask]
            x = np.log10(np.clip(freq_hz, c.min(), c.max()))
            return float(np.interp(x, np.log10(c), r))
        return default
    
    for rev in reflections:
        f_res = rev['freq']
        # FIXED: Changed 'error_ms' to 'gd_error' to match analyze_acoustic_confidence
        error_ms = rev['gd_error'] 
        ref_rt60 = rt60_at(f_res)
        # HERKEMPI KYNNYS: Reagoidaan jo 80% kohdalla keskimääräisestä RT60:stä
        excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
        
        if excess_ratio > 0.8: 
            # Dynaaminen kerroin
            dynamic_mult = np.clip(excess_ratio * base_strength, 0.2, 3.0)
            
            # Kapeampi ja kohdistetumpi kaistanleveys (BW)
            bw = f_res / (error_ms / 15.0) 
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw)**2)
            
            reduction_db = dynamic_mult * 4.0
            # Kerätään vaikutus erilliseen käyrään (estää "stacking surprise" -notchit)
            tdc_reduction_db += (kernel * reduction_db)
            
    # --- Safety brakes ---
    # 1) Hard cap total reduction (per bin)
    if max_total_reduction_db and max_total_reduction_db > 0:
        tdc_reduction_db = np.minimum(tdc_reduction_db, float(max_total_reduction_db))

    # 2) Optional slope limiting in dB/oct to keep the curve smooth/predictable
    try:
        if max_slope_db_per_oct and float(max_slope_db_per_oct) > 0:
            tdc_reduction_db = limit_slope_per_octave(
                freq_axis,
                tdc_reduction_db,
                max_db_per_oct=float(max_slope_db_per_oct),
            )
    except Exception:
        # Never let TDC fail the whole pipeline
        pass

    adjusted_target -= tdc_reduction_db
    return adjusted_target

def apply_hpf_to_mags(freqs, mags, cutoff, order):
    """Soveltaa Butterworth-ylipäästösuodatusta magnitudivasteeseen (dB)."""
    if cutoff <= 0 or order <= 0:
        return mags
    f = np.asarray(freqs, dtype=float)
    # Vältä DC-binin (0 Hz) ääretön vaimennus stats/plot -puolella
    if f.size > 1 and f[0] == 0.0:
        f = f.copy()
        f[0] = f[1] if f[1] > 0 else 1e-6
    # Butterworth vaste: 1 / sqrt(1 + (fc/f)^(2*order))
    # Muutetaan desibeleiksi: -10 * log10(1 + (fc/f)^(2*order))
    with np.errstate(divide='ignore'):
        attenuation = -10 * np.log10(1 + (cutoff / (f + 1e-12))**(2 * order))
    return mags + attenuation

def soft_clip_boost(gain_db, max_boost):
    """Pehmentää korostukset tanh-funktiolla, jotta max_boost ei ylity rajusti."""
    if gain_db <= 0: return gain_db
    return max_boost * np.tanh(gain_db / max_boost)

def soft_clip_gain(gain_db, max_boost_db, max_cut_db):
    """
    Pehmeä rajoitin sekä boostille että cutille.
    - boost: +max_boost_db * tanh(g/max_boost_db)
    - cut:   -max_cut_db  * tanh(|g|/max_cut_db)
    """
    g = np.asarray(gain_db, dtype=float)
    out = np.empty_like(g)
    pos = g > 0
    neg = ~pos
    # Boost
    if np.any(pos):
        mb = float(max_boost_db) if max_boost_db > 0 else 0.0
        out[pos] = mb * np.tanh(g[pos] / (mb + 1e-12)) if mb > 0 else 0.0
    # Cut
    if np.any(neg):
        mc = float(max_cut_db) if max_cut_db > 0 else 0.0
        out[neg] = -mc * np.tanh((-g[neg]) / (mc + 1e-12)) if mc > 0 else g[neg]
    return out

def limit_slope_per_octave(freq_axis, gain_db, max_db_per_oct=12.0):
    """
    Rajoittaa gain-käyrän muutoksen (dB) per oktaavi (log2(f)).
    Tekee forward+backward passin, jotta rajoitus toteutuu molempiin suuntiin.
    """
    f = np.asarray(freq_axis, dtype=float)
    g = np.asarray(gain_db, dtype=float).copy()
    max_db_per_oct = float(max_db_per_oct)
    if max_db_per_oct <= 0:
        return g

    # Käytetään vain f>0 alueella (f=0 kohdalla log2 ei toimi)
    idx = np.where(f > 0)[0]
    if idx.size < 3:
        return g

    ii = idx
    x = np.log2(f[ii])
    # Forward: rajoita nousu/lasku edelliseen nähden
    for k in range(1, ii.size):
        dx = x[k] - x[k-1]
        if dx <= 1e-12:
            continue
        lim = max_db_per_oct * dx
        dg = g[ii[k]] - g[ii[k-1]]
        if dg > lim:
            g[ii[k]] = g[ii[k-1]] + lim
        elif dg < -lim:
            g[ii[k]] = g[ii[k-1]] - lim

    # Backward: sama toiseen suuntaan
    for k in range(ii.size - 2, -1, -1):
        dx = x[k+1] - x[k]
        if dx <= 1e-12:
            continue
        lim = max_db_per_oct * dx
        dg = g[ii[k]] - g[ii[k+1]]
        if dg > lim:
            g[ii[k]] = g[ii[k+1]] + lim
        elif dg < -lim:
            g[ii[k]] = g[ii[k+1]] - lim

    return g


def calculate_minimum_phase(mags_lin_fft):
    """Laskee minimivaiheen Hilbert-muunnoksella. 1e-10 suojaus estää NaN-virheet."""
    n_fft = (len(mags_lin_fft) - 1) * 2
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)
    return min_phase_rad[:len(mags_lin_fft)]

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    """Yhdistää raskaan ja kevyen tasoituksen parhaan visualisoinnin saamiseksi."""
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    """Soveltaa taajuusriippuvaista ikkunointia (FDW) vaiheelle."""
    safe_cycles = max(cycles, 1.0)
    phase_u = np.unwrap(np.deg2rad(phases))
    oct_width = 2.0 / safe_cycles
    dummy_mags = np.zeros_like(freqs)
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_adaptive_fdw(freqs, mags, confidence_mask, base_cycles=15.0, min_cycles=5.0):
    """
    Sovelletaan adaptiivista magnitudin tasoitusta luottamuksen perusteella.
    Korkea luottamus = enemmän syklejä (tarkempi korjaus).
    Matala luottamus = vähemmän syklejä (raskaampi tasoitus).
    """
    # Lasketaan taajuuskohtainen syylimäärä confidence_maskin perusteella
    adaptive_cycles = min_cycles + (confidence_mask * (base_cycles - min_cycles))
    oct_widths = 2.0 / np.maximum(adaptive_cycles, 1.0)
    
    smoothed_mags = np.copy(mags)
    
    # Tehdään tasoitus usealla eri tarkkuudella ja yhdistetään ne maskin avulla
    # Tämä on nopeampi ja vakaampi tapa kuin taajuuskohtainen silmukka
    for bw in [1/3, 1/6, 1/12, 1/24, 1/48]:
        sm, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), bw)
        # Valitaan tälle taajuudelle sopiva tasoitusleveys
        mask = (oct_widths >= bw * 0.7) & (oct_widths < bw * 1.5)
        smoothed_mags[mask] = sm[mask]
    
    return smoothed_mags

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    """Vakioitu oktaavitasoitus logaritmisella näytteistyksellä."""
    if octave_fraction <= 0: return mags, phases
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    
    
    points_per_octave = 384 
    
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    num_points = max(num_points, 10)
    
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    
    window_size = int(points_per_octave * octave_fraction)
    window_size = max(window_size, 1) # Varmistetaan vähintään 1
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

def calculate_theoretical_phase(freq_axis, crossovers, hpf_freq=None, hpf_slope=None):
    """Laskee jakosuotimien ja ylipäästön (HPF) aiheuttaman teoreettisen vaihesiirron."""
    total_phase_rad = np.zeros_like(freq_axis)
    
    # 1. Lisätään HPF:n vaihevaikutus analogisella Butterworth-mallilla
    if hpf_freq and hpf_slope and hpf_freq > 0:
        hpf_order = int(hpf_slope / 6)
        b, a = scipy.signal.butter(hpf_order, 2 * np.pi * hpf_freq, btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    # 2. Lisätään jakosuotimien vaihe
    for xo in crossovers:
        if xo.get('freq') is None: continue
        order = xo.get('order', int(xo.get('slope', 12) / 6))
        b, a = scipy.signal.butter(order, 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
        
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    """Interpoloi vasteen lineaarisesti kohdetaajuuksille."""
    return np.interp(target_freqs, input_freqs, input_values)

def _sigma_bins_from_hz(freq_axis, sigma_hz: float, fallback_bins: float = 3.0) -> float:
    """
    Muunna tasoituksen leveys (Hz) gaussian_filter1d:n sigma-arvoksi (binneissä).
    freq_axis oletetaan nousevaksi ja melko tasaväliseksi.
    """
    try:
        f = np.asarray(freq_axis, dtype=float)
        if f.size < 4:
            return float(fallback_bins)
        # Käytä mediaania, robusti pieneen epätasaisuuteen
        df = np.median(np.diff(f))
        if not np.isfinite(df) or df <= 0:
            return float(fallback_bins)
        s = float(sigma_hz) / float(df)
        if not np.isfinite(s) or s <= 0:
            return float(fallback_bins)
        return float(max(1.0, s))
    except Exception:
        return float(fallback_bins)


def calculate_group_delay(freqs, phases_deg):
    """Laskee ryhmäviiveen (ms) vaiheen gradientista."""
    phase_rad = np.unwrap(np.deg2rad(phases_deg))
    d_phi_d_f = np.gradient(phase_rad, freqs)
    gd_ms = -d_phi_d_f / (2 * np.pi) * 1000.0
    # Tasoitus "Hz-leveydellä", ei binneillä -> pysyy samana fs/taps muuttuessa
    sigma_bins = _sigma_bins_from_hz(freqs, sigma_hz=2.0, fallback_bins=3.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)


def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=300):
    """Yhdistää Linear Phase basson ja Minimum Phase diskantin."""
    ntaps = len(ir_lin)
    fir_lp = scipy.signal.firwin(2047, split_freq, fs=fs, pass_zero=True, window='blackman')
    fir_hp = -fir_lp
    fir_hp[1023] += 1.0
    
    idx_lin = np.argmax(np.abs(ir_lin))
    idx_min = np.argmax(np.abs(ir_min))
    shift = idx_lin - idx_min
    ir_min_aligned = np.roll(ir_min, shift)
    
    filt_bass = scipy.signal.fftconvolve(ir_lin, fir_lp, mode='same')
    filt_treble = scipy.signal.fftconvolve(ir_min_aligned, fir_hp, mode='same')
    return filt_bass + filt_treble

def remove_time_of_flight(freq_axis, phase_rad):
    """Etsii ja poistaa linear phase slopen (viiveen) mittauksesta."""
    mask = (freq_axis >= 1000) & (freq_axis <= 10000)
    if not np.any(mask): return phase_rad, 0.0
    poly = np.polyfit(freq_axis[mask], phase_rad[mask], 1)
    return phase_rad - (poly[0] * freq_axis), poly[0]

def analyze_acoustic_confidence(freq_axis, complex_meas, fs):
    """Analysoi akustisen luottamuksen skaalautuvalla resoluutiolla."""
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0
    
    # Tasoitus vakio Hz-leveydellä (vastaa aiempaa ~44.1k/65536-taps käyttäytymistä)
    # 44.1k & ~65537 FFT-bins -> df ~0.67 Hz -> sigma_bins=10 ~ 6.7 Hz
    sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=6.7, fallback_bins=10.0)
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)
    gd_diff = np.abs(gd_ms - gd_smooth)

    # Luottamusmaski
    threshold_ms = 2.5
    confidence_mask = 1.0 / (1.0 + np.exp(1.5 * (gd_diff - threshold_ms)))
    # Varmistetaan, että `peaks` on aina määritelty (turvaa mahdollisia refaktorointeja vastaan)
    peaks = np.array([], dtype=int)

    reflection_nodes = []
    valid_idx = np.where(freq_axis > 20)[0] 
    
    if len(valid_idx) > 0:
        # 2. PIIKKIEN ETSINTÄ (Herkkyyden lisäys)
        # Lasketaan korkeusvaatimus 3.0 -> 2.0, jotta pienemmätkin moodit löytyvät
        # Lyhennetään distance 50 -> 30, jotta lähekkäiset resonanssit tunnistetaan
        peaks, props = scipy.signal.find_peaks(gd_diff[valid_idx], height=2.0, distance=100)
    
    raw_nodes = []
    for p in peaks:
        idx = valid_idx[p]
        raw_nodes.append({
            'freq': round(freq_axis[idx], 1), 
            'gd_error': round(gd_diff[idx], 2),
            'dist': round((gd_diff[idx] / 1000.0 * 343.0) / 2.0, 2), 
            'type': "Resonance" if freq_axis[idx] < 200 else "Reflection"
        })
    
    # POIMITAAN VAIN 15 VOIMAKKAINTA (estää pisteytyksen romahtamisen)
    reflection_nodes = sorted(raw_nodes, key=lambda x: x['gd_error'], reverse=True)[:15]
            
    return confidence_mask, reflection_nodes, gd_ms


def calculate_rt60(impulse, fs):
    """
    Robustimpi RT60-estimaatti:
    - Schroeder EDC
    - noise floor -rajaus
    - EDT, T20, T30 + fallback
    - laatukriteeri (R^2)
    Palauttaa float (sekunteina), muuten 0.0
    """
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return 0.0

        # 1) Aloitus: peak-kohdasta (kuten sinulla)
        peak_idx = int(np.argmax(np.abs(imp)))
        x = imp[peak_idx:]
        if x.size < int(0.05 * fs):
            return 0.0

        # 2) Energiakäyrä
        e = x * x

        # 3) Noise floor tailista (viimeiset 15% tai vähintään 50 ms)
        tail_n = max(int(0.15 * e.size), int(0.05 * fs))
        tail_n = min(tail_n, e.size)
        noise_power = float(np.mean(e[-tail_n:]))

        # 4) Schroeder EDC (integroitu energia)
        E = np.cumsum(e[::-1])[::-1]
        E0 = float(E[0]) + 1e-18

        # 5) Stop-hetki: missä integroitu energia lähestyy melulattiaa
        # noise_mult määrittää kuinka “varhain” lopetetaan (20 = konservatiivinen)
        noise_mult = 20.0
        stop_candidates = np.where(E <= noise_mult * noise_power)[0]
        stop_idx = int(stop_candidates[0]) if stop_candidates.size > 0 else (E.size - 1)
        stop_idx = max(stop_idx, 10)  # ettei mene nollapituudeksi

        t = np.arange(E.size) / fs
        edc_db = 10.0 * np.log10((E / E0) + 1e-30)

        # 6) Pieni tasoitus EDC:hen (vakauttaa indeksejä)
        smooth_ms = 10.0
        win = max(1, int((smooth_ms / 1000.0) * fs))
        if win > 1:
            kernel = np.ones(win) / win
            edc_db = np.convolve(edc_db, kernel, mode="same")

        # Rajataan luotettavalle alueelle
        t_u = t[:stop_idx + 1]
        d_u = edc_db[:stop_idx + 1]

        def fit_rt(lo_db, hi_db):
            # fit d_u ~ a*t + b, kun d_u on välillä [hi_db..lo_db]
            mask = (d_u <= lo_db) & (d_u >= hi_db)
            if np.count_nonzero(mask) < 12:
                return None

            tt = t_u[mask]
            yy = d_u[mask]
            A = np.vstack([tt, np.ones_like(tt)]).T
            a, b = np.linalg.lstsq(A, yy, rcond=None)[0]  # yy = a*t + b

            # R^2
            yhat = a * tt + b
            ss_res = float(np.sum((yy - yhat) ** 2))
            ss_tot = float(np.sum((yy - np.mean(yy)) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot

            if a >= -1e-9:  # ei laske
                return None

            rt60 = -60.0 / a
            return rt60, r2

        candidates = []
        # EDT: 0..-10 dB (sama kaava rt60 = -60/a)
        r = fit_rt(0.0, -10.0)
        if r: candidates.append(("EDT",) + r)

        # T20: -5..-25
        r = fit_rt(-5.0, -25.0)
        if r: candidates.append(("T20",) + r)

        # T30: -5..-35
        r = fit_rt(-5.0, -35.0)
        if r: candidates.append(("T30",) + r)

        if not candidates:
            return 0.0

        # Preferenssi: T30 > T20 > EDT, mutta vaadi kohtuullinen R^2 jos mahdollista
        pref = {"T30": 0, "T20": 1, "EDT": 2}
        candidates.sort(key=lambda x: (pref[x[0]], -x[2]))  # (prefer, best r2)

        # Valitse ensin r2>=0.90 jos löytyy, muuten paras saatavilla
        chosen = None
        for c in candidates:
            name, rt60, r2 = c
            if r2 >= 0.90:
                chosen = (rt60, r2, name)
                break
        if chosen is None:
            name, rt60, r2 = candidates[0]
            chosen = (rt60, r2, name)

        rt60 = float(chosen[0])

        # Järkevyysrajat (voit säätää)
        if 0.05 < rt60 < 5.0:
            return round(rt60, 2)
        return 0.0

    except Exception:
        return 0.0
    

def _third_oct_centers(f_min=31.5, f_max=8000.0):
    """IEC-tyyliset 1/3-oktaavikaistakeskukset (riittää tähän)."""
    centers = []
    f = float(f_min)
    step = 2 ** (1/3)  # 1/3 oktaavi
    while f <= f_max * 1.0001:
        centers.append(float(f))
        f *= step
    return centers

def calculate_rt60_bands(impulse, fs, f_min=31.5, f_max=8000.0, order=4):
    """
    Laskee RT60:n 1/3-oktaavikaistoittain:
      - bandpass (sos) -> suodatettu impulssi
      - RT60 (käyttää olemassa olevaa calculate_rt60:ää)
    Palauttaa dict: {center_hz: rt60_s}
    """
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return {}

        nyq = 0.5 * fs
        centers = _third_oct_centers(f_min, min(f_max, nyq * 0.90))
        out = {}

        for fc in centers:
            # 1/3 oktaavin rajat: fc * 2^(±1/6)
            fl = fc / (2 ** (1/6))
            fh = fc * (2 ** (1/6))
            # clamp
            fl = max(1.0, fl)
            fh = min(nyq * 0.98, fh)
            if fh <= fl * 1.05:
                continue

            sos = scipy.signal.butter(order, [fl/nyq, fh/nyq], btype='bandpass', output='sos')
            # nollavaiheinen suodatus (vakaa, ei lisää group delaytä)
            x = scipy.signal.sosfiltfilt(sos, imp)
            rt = calculate_rt60(x, fs)
            if 0.05 < rt < 5.0:
                out[float(round(fc, 2))] = float(rt)
        return out
    except Exception:
        return {}


def get_min_phase_impulse(mags_db, n_fft):
    """Luo minimivaiheisen impulssivasteen voimakkuusvasteesta."""
    # Muunnetaan dB -> lineaarinen ja luodaan symmetrinen spektri
    amp = 10**(mags_db / 20.0)
    # Hilbert-muunnos vaatii logaritmi-amplitudin
    l_amp = np.log(amp + 1e-12)
    # Lasketaan minimivaihe käyttämällä FFT:tä ja Hilbertin periaatetta
    h = scipy.fft.ifft(l_amp)
    n = len(h)
    window = np.zeros(n)
    window[0] = 1
    window[1:n//2] = 2
    window[n//2] = 1
    # Muodostetaan minimivaiheinen vaste
    min_phase = np.exp(scipy.fft.fft(h * window))
    return np.real(scipy.fft.ifft(min_phase))

        




#----------- Filtteri ---------------------


import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
import logging
from models import FilterConfig

logger = logging.getLogger("CamillaFIR.dsp")

#----- FILTTERI

def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig):
    # --- 1. DATAN TASAUS ---
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in, m_in, p_in = freqs[:min_len], meas_mags[:min_len], raw_phases[:min_len]

    # --- 2. AKSELIT ---
    n_fft = cfg.num_taps if cfg.num_taps % 2 != 0 else cfg.num_taps + 1
    freq_axis = np.linspace(0, cfg.fs/2.0, n_fft // 2 + 1)
    
    # --- 3. TASOITUS (Skaalautuva resoluutio) ---
    oct_frac = 1.0 / float(cfg.smoothing_level) if cfg.smoothing_level > 0 else 1/12.0
    is_psy = 'Psy' in str(cfg.smoothing_type).lower()
    
    # Magnituditasoitus
    m_smooth, _ = apply_smoothing_std(f_in, m_in, np.zeros_like(m_in), oct_frac * (1.5 if is_psy else 1.0))
    
    # KORJAUS: Dynaaminen vaihetasoitus näytetaajuuden mukaan
    # Korkeilla taajuuksilla (>96kHz) interpolointi luo vaiheeseen "kulmia", 
    # jotka siistitään raskaammalla (1/12 oct) tasoituksella.
    p_smooth_oct = 1/12.0 if cfg.fs > 96000 else 1/24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)

    # --- 4. TOF & INTERPOLOINTI ---
    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)

    # --- 5. ANALYYSI (Skaalattu luottamusmaski) ---
    m_anal = np.interp(freq_axis, f_in, m_smooth)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10**(m_anal/20.0) * np.exp(1j * p_anal_rad)
    
    # Confidence mask ja heijastusanalyysi skaalautuvalla sigmalla
    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)

    # --- 5C. COMPARISON MODE (locked 44.1k analysis grid for score/match/report) ---
    cmp = None
    analysis_mode = "native"
    try:
        if bool(getattr(cfg, "comparison_mode", False)):
            ref_fs = int(getattr(cfg, "comparison_ref_fs", 44100) or 44100)
            ref_taps = int(getattr(cfg, "comparison_ref_taps", 65536) or 65536)
            ref_nfft = ref_taps if (ref_taps % 2 != 0) else (ref_taps + 1)
            freq_cmp_full = np.linspace(0, ref_fs / 2.0, ref_nfft // 2 + 1)

            # clamp comparison grid to what we can represent with current freq_axis
            fmax = float(freq_axis[-1]) if freq_axis.size else 0.0
            if fmax > 0:
                freq_cmp = freq_cmp_full[freq_cmp_full <= fmax]
            else:
                freq_cmp = freq_cmp_full

            # resample analysis magnitude/phase to comparison grid
            m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
            p_cmp_rad = np.interp(freq_cmp, freq_axis, p_anal_rad)
            complex_cmp = 10 ** (m_cmp_raw / 20.0) * np.exp(1j * p_cmp_rad)

            # recompute confidence on reference fs (makes GD-based confidence stable across cfg.fs/taps)
            conf_cmp, refl_cmp, _ = analyze_acoustic_confidence(freq_cmp, complex_cmp, ref_fs)

            # resample target and compute leveling on comparison grid (for stable offset + match window)
            target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
            (
                target_level_db_cmp,
                calc_offset_db_cmp,
                meas_level_db_window_cmp,
                target_level_db_window_cmp,
                offset_method_cmp,
                s_min_cmp,
                s_max_cmp,
            ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp)

            # resample filter correction curve to comparison grid
            filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)

            cmp = {
                "cmp_ref_fs": float(ref_fs),
                "cmp_ref_taps": float(ref_taps),
                "cmp_freq_axis": freq_cmp.tolist(),
                "cmp_target_mags": target_cmp.tolist(),
                "cmp_measured_mags": (m_cmp_raw - calc_offset_db_cmp).tolist(),
                "cmp_filter_mags": filt_cmp.tolist(),
                "cmp_confidence_mask": conf_cmp.tolist(),
                "cmp_reflections": refl_cmp,
                "cmp_smart_scan_range": [float(s_min_cmp), float(s_max_cmp)],
                "cmp_eff_target_db": float(target_level_db_cmp),
                "cmp_offset_db": float(calc_offset_db_cmp),
                "cmp_meas_level_db_window": float(meas_level_db_window_cmp),
                "cmp_target_level_db_window": float(target_level_db_window_cmp),
                "cmp_offset_method": str(offset_method_cmp),
                "cmp_avg_confidence": float(np.mean(conf_cmp) * 100.0),
            }
            if (
                isinstance(cmp.get("cmp_freq_axis", None), list)
                and isinstance(cmp.get("cmp_measured_mags", None), list)
                and isinstance(cmp.get("cmp_target_mags", None), list)
                and isinstance(cmp.get("cmp_filter_mags", None), list)
                and isinstance(cmp.get("cmp_confidence_mask", None), list)
                and len(cmp["cmp_freq_axis"]) > 16
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_measured_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_target_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_filter_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_confidence_mask"])
            ):
                analysis_mode = "comparison"
    except Exception:
        cmp = None
        analysis_mode = "native"

    # --- 5B. A-FDW: käytä luottamusmaskia magnitudin "analyysiversion" tasoituksessa 
    
    # - conf_mask ~ 1.0 => tarkempi (enemmän syklejä) => saat enemmän "oikeaa" korjausta
    # - conf_mask ~ 0.0 => raskaampi tasoitus => vältetään aggressiivinen korjaus epäluotettavassa datassa
    #
    # Tämä vaikuttaa suoraan:
    # - leveling (m_anal)
    # - mag-korjaus (raw_g)
    # mutta EI muuta heijastusten/gd-detektiota (se perustuu pääosin vaiheeseen).
    if getattr(cfg, "enable_afdw", False):
        # min_cycles pidetään järkevänä (ettei tule "liian terävää" edes huonolla confidence-alueella)
            base = float(getattr(cfg, "fdw_cycles", 15.0))
            min_c = max(3.0, base / 3.0)
            m_anal = apply_adaptive_fdw(
            freq_axis,
            m_anal,
            conf_mask,
            base_cycles=base,
            min_cycles=min_c
        )


    # --- 6. RT60 & TARGET ---
    m_rt_lin = np.interp(np.linspace(0, cfg.fs/2, 65537), freq_axis, np.interp(freq_axis, f_in, m_in))
    rt_ir = get_min_phase_impulse(m_rt_lin, 131072)
    current_rt60 = calculate_rt60(rt_ir, cfg.fs)
    rt60_bands = calculate_rt60_bands(rt_ir, cfg.fs, f_min=31.5, f_max=8000.0, order=4)
    # “Yksi luku” kaistoista (hyvä score/reportointiin): mediaani 125–4000 Hz jos löytyy
    band_avg = 0.0
    if rt60_bands:
        ks = np.array(sorted(rt60_bands.keys()), dtype=float)
        vs = np.array([rt60_bands[k] for k in ks], dtype=float)
        mid = (ks >= 125.0) & (ks <= 4000.0) & (vs > 0.05) & (vs < 5.0)
        if np.any(mid):
            band_avg = float(np.median(vs[mid]))
        else:
            band_avg = float(np.median(vs))
    
    if cfg.house_freqs is not None and cfg.house_mags is not None and len(cfg.house_freqs) >= 2 and len(cfg.house_mags) >= 2:
        target_mags = interpolate_response(cfg.house_freqs, cfg.house_mags, freq_axis)
    else:
        # fallback: flat 0 dB target
        target_mags = np.zeros_like(freq_axis, dtype=float)
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        target_mags = apply_hpf_to_mags(freq_axis, target_mags, cfg.hpf_settings['freq'], cfg.hpf_settings['order'])
    
    if cfg.enable_tdc:
        # UUSI: TDC saa taajuusriippuvan RT60:n (dict), fallbackaa automaattisesti jos tyhjä
        rt60_for_tdc = rt60_bands if rt60_bands else current_rt60
        # Safety brakes: cap total TDC reduction and keep it smooth (avoid deep, stacked notches)
        # Configurable TDC safety brakes for easy A/B testing
        tdc_max_red = float(getattr(cfg, "tdc_max_reduction_db", 9.0) or 9.0)
        tdc_slope = float(getattr(cfg, "tdc_slope_db_per_oct", 0.0) or 0.0)

        # Clamp to sane values (never explode)
        if tdc_max_red < 0: tdc_max_red = 0.0
        if tdc_max_red > 24: tdc_max_red = 24.0
        if tdc_slope < 0: tdc_slope = 0.0
        if tdc_slope > 24: tdc_slope = 24.0

        target_mags = apply_smart_tdc(
            freq_axis,
            target_mags,
            reflections,
            rt60_for_tdc,
            cfg.tdc_strength / 100.0,
            max_total_reduction_db=tdc_max_red,
            max_slope_db_per_oct=tdc_slope
        )

    # --- HPF params (always defined) ---
    hpf_f = 0.0
    hpf_order = 0
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        hpf_f = float(cfg.hpf_settings.get('freq', 0.0) or 0.0)
        hpf_order = int(cfg.hpf_settings.get('order', 0) or 0)

    # --- 7. TASONSOVITUS ---
    # Huom: tasosovitus on erotettu omaan moduuliin testattavuuden ja edge-case -robustiuden takia.
    target_level_db, calc_offset_db, meas_level_db_window, target_level_db_window, offset_method, s_min, s_max = (
        compute_leveling(cfg, freq_axis, m_anal, target_mags)
    )


    # --- 8. KORJAUS ---
    gain_db = np.zeros_like(freq_axis)
    if cfg.enable_mag_correction:
        afdw_on = bool(getattr(cfg, "enable_afdw", False))
        afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
        afdw_min = max(3.0, afdw_base / 3.0)
        
        raw_g = target_mags - (m_anal - calc_offset_db)

        base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)

        # Raw_g smoothing:
        # - legacy: sigma in bins scales directly with fs (can over-smooth at high fs)
        # - df_smoothing: keep smoothing width constant in Hz (reference: 44.1k/65536 behavior)
        df_mode = bool(getattr(cfg, "df_smoothing", False))
        if df_mode:
            # Reference bin width ~ 44100/65536 Hz; match the old "base_sigma bins" at ref
            df_ref = 44100.0 / 65536.0
            sigma_hz = float(base_sigma) * df_ref
            # Convert Hz -> bins for current axis
            sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
            sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=float(sigma_bins))
        else:
            sigma_scaling = cfg.fs / 44100.0
            sigma = max(2, int(base_sigma * sigma_scaling))
            sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=sigma)

        final_g = raw_g - (raw_g - sm_g) * (cfg.reg_strength / 100.0)
        
        # --- 8B. A-FDW suoraan korjauskäyrään ---
        # Tasoittaa final_g adaptiivisesti confidence-maskin mukaan:
        # - matala confidence => enemmän "syklejä" => pehmeämpi korjaus
        # - korkea confidence => vähemmän tasoitusta => tarkempi korjaus
        if afdw_on:
            final_g = apply_adaptive_fdw(
                freq_axis,
                final_g,
                conf_mask,
                base_cycles=afdw_base,
                min_cycles=afdw_min
            )
        mask_c = (freq_axis >= (0 if cfg.hpf_settings else cfg.mag_c_min)) & (freq_axis <= cfg.mag_c_max)
        # Kun A-FDW on päällä, ei kerrota final_g:ta eff_conf:lla,
        # koska A-FDW jo tekee "varovaisuuden" muotoon (tasoitus).
        # Tämä välttää tuplavarovaisuuden (muoto pehmenee + amplitudi vaimenee).
        if bool(getattr(cfg, "enable_afdw", False)):
            gain_apply = final_g.copy()
        else:
            eff_conf = np.where(freq_axis < 100, np.maximum(conf_mask, 0.6), conf_mask)
            gain_apply = (final_g * eff_conf).copy()

        # --- 8C. Low-bass CUT allowance (täsmäfix 32 Hz -tyyppisille moodeille) ---
        # < low_hz: sallitaan VAIN vaimennus (ei boostia),
        # ja käytetään tarvittaessa "vahvempaa" leikkausta (min(final_g, raw_g)),
        # jotta regularisointi / confidence ei nollaa selviä huonemoodipiikkejä.
        low_hz = float(getattr(cfg, "low_bass_cut_hz", 40.0))
        low_mask = mask_c & (freq_axis > 0) & (freq_axis <= low_hz)
        if np.any(low_mask):
            low_cut = np.minimum(final_g[low_mask], raw_g[low_mask])  # valitse negatiivisempi (vahvempi cut)
            low_cut = np.minimum(low_cut, 0.0)                       # ei koskaan boostia
            gain_apply[low_mask] = low_cut

        # --- 8D. Max cut + max boost (pehmeä) ---
        max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))  # default: sallitaan kohtuullinen leikkaus
        tmp = np.zeros_like(gain_db, dtype=float)
        tmp[mask_c] = gain_apply[mask_c]
        tmp = soft_clip_gain(tmp, cfg.max_boost_db, max_cut_db)
        gain_db[mask_c] = tmp[mask_c]

        # --- 8E. Slope/oktaavi -rajoitin (gain-käyrän “jyrkkyys”) ---
        # Huom: tehdään ennen exc_prot:ia ja ajetaan exc_prot lopuksi uudelleen,
        # jotta slope-limitointi ei voi “vuotaa” boostia suojavyöhykkeille.
        max_slope = float(getattr(cfg, "max_slope_db_per_oct", 12.0))  # 0 = pois
        if max_slope > 0:
            # Rajoitetaan vain korjausalueella
            g2 = gain_db.copy()
            g2 = limit_slope_per_octave(freq_axis, g2, max_db_per_oct=max_slope)
            # Pidetään ulkopuoli koskemattomana
            gain_db[mask_c] = g2[mask_c]
        
        f_start = max(cfg.mag_c_max - cfg.trans_width, cfg.mag_c_min)
        
        f_mask = (freq_axis > f_start) & (freq_axis <= cfg.mag_c_max)
        # Varmistetaan jakolasku (ettei jaeta nollalla, jos trans_width on 0)
        fade_len = cfg.mag_c_max - f_start
        if np.any(f_mask) and fade_len > 0: 
            gain_db[f_mask] *= (cfg.mag_c_max - freq_axis[f_mask]) / fade_len
        if cfg.exc_prot:
            # Määritetään siirtymäalue (n. 1/2 oktaavia, kerroin 1.41)
            f_start = cfg.exc_freq
            f_end = cfg.exc_freq * 1.41
            
            # 1. Täysi suojaus f_start alapuolella: Pakotetaan boost nollaan, sallitaan leikkaus
            prot_mask = freq_axis < f_start
            gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
            
            # 2. Pehmeä siirtymäalue f_start -> f_end
            # Tällä alueella sallittu boost kasvaa lineaarisesti 0 dB -> max_boost_db
            trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
            if np.any(trans_mask):
                # Lasketaan häivytyskerroin (0.0 -> 1.0)
                fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
                # Sallittu maksimiboost tällä taajuudella
                allowed_boost = fade * cfg.max_boost_db
                # Rajoitetaan boost, mutta pidetään kaikki vaimennukset (leikkaukset)
                gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
            
            logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")

        # --- HPF params (always defined) ---
        hpf_f = 0.0
        if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
            hpf_f = float(cfg.hpf_settings.get('freq', 0.0) or 0.0)

        # --- HPF policy: täysi stop + silkkinen fade (asym-safe) ---
        if hpf_f > 0:
                hpf_end = hpf_f * 1.41  # ~1/2 oktaavia

                # 1) Täysi stoppi HPF:n alapuolella
                below = freq_axis < hpf_f
                gain_db[below] = 0.0

                # 2) Pehmeä häivytys HPF -> HPF*1.41 (0..1)
                trans = (freq_axis >= hpf_f) & (freq_axis <= hpf_end)
                if np.any(trans):
                    fade = (freq_axis[trans] - hpf_f) / (hpf_end - hpf_f)
                    gain_db[trans] *= fade
        
        # --- 8F. Final safety clamp (max boost / max cut) ---
        # Varmistetaan että mikään myöhempi operaatio (fade/slope/exc_prot) ei ylitä rajoja.
        max_cut_db = float(getattr(cfg, "max_cut_db", 15.0))
        max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
        gain_db = np.minimum(gain_db, float(cfg.max_boost_db))
        gain_db = np.maximum(gain_db, -max_cut_db)

    # --- 9. VAIHEEN GENERONTI ---
                # --- THEORETICAL PHASE (single source of truth) ---
        hpf_freq = None
        hpf_slope = None
        hs = cfg.hpf_settings

        if isinstance(hs, dict) and hs.get('enabled'):
            hpf_freq = float(hs.get('freq', 0.0) or 0.0)
            hpf_order = int(hs.get('order', 0) or 0)
            if hpf_freq > 0 and hpf_order > 0:
                hpf_slope = float(hpf_order * 6)  # dB/oct

        theo_xo = calculate_theoretical_phase(
            freq_axis,
            cfg.crossovers,
            hpf_freq=hpf_freq,
            hpf_slope=hpf_slope
        )

        # --- LOGGING: HPF inclusion status ---
        if hpf_freq and hpf_slope:
            logger.info(
                f"Theoretical phase includes HPF: f={hpf_freq:.1f} Hz, "
                f"slope={hpf_slope:.0f} dB/oct (order={int(hpf_slope/6)})"
            )
        else:
            logger.info("Theoretical phase: HPF not included")
            
        if getattr(cfg, "phase_safe_2058", False):
            logger.info("Phase mode: 2058-safe (no room phase correction)")
        else:
            logger.info("Phase mode: modern (excess-phase + confidence + FDW)")
    
        # --- 9A. HPF magnitude into FIR (enabled-check) ---
        if isinstance(hs, dict) and hs.get('enabled'):
            hpf_f = float(hs.get('freq', 0.0) or 0.0)
            hpf_order = int(hs.get('order', 0) or 0)
            if hpf_f > 0 and hpf_order > 0:
                hpf_db = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), hpf_f, hpf_order)
                gain_db = gain_db + hpf_db

    # --- 9B. CLIP PREVENTION & HEADROOM (MOVED UP, so it affects the actual FIR) ---
    current_peak_gain = float(np.max(gain_db + cfg.global_gain_db))
    auto_headroom_db = 0.0
    if current_peak_gain > 0:
        auto_headroom_db = -current_peak_gain - 0.1
        logger.info(f"Clip Prevention: Applied {auto_headroom_db:.2f} dB headroom.")

    final_gain_total = gain_db + cfg.global_gain_db + auto_headroom_db
    total_mag = 10**(final_gain_total / 20.0)
    min_p = calculate_minimum_phase(total_mag)

    # --- 9C. PHASE LOGIC (single entry point) ---
    theo_xo = calculate_theoretical_phase(freq_axis, cfg.crossovers)

    if bool(getattr(cfg, "phase_safe_2058", False)):
        # === 2058-SAFE PHASE MODE ===
        # No room phase correction (no excess-phase, FDW, confidence)

        if 'Min' in cfg.filter_type_str:
            final_phase = min_p

        elif 'Mixed' in cfg.filter_type_str:
            f_center = float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0)
            f_center = float(np.clip(
                f_center, 20.0,
                float(freq_axis[-1] if freq_axis.size else 20000.0)
            ))

            safe_freqs = np.maximum(freq_axis, 1.0)
            octave_dist = np.log2(safe_freqs / f_center)
            mask = np.clip((octave_dist + 0.5), 0.0, 1.0)
            sm_mask = 3.0 * mask**2 - 2.0 * mask**3  # smoothstep

            low_phase = -theo_xo
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p

        else:
            # Linear / Asym
            final_phase = -theo_xo

    else:
        # Includes excess-phase, confidence, phase_limit blend, etc.
        # IMPORTANT: compute low_phase HERE before using it.

        # 1) Smooth confidence slightly (avoid sawtooth weighting)
        try:
            conf_s = scipy.ndimage.gaussian_filter1d(conf_mask.astype(float), sigma=2)
            conf_s = np.clip(conf_s, 0.0, 1.0)
        except Exception:
            conf_s = np.clip(conf_mask.astype(float), 0.0, 1.0)

        phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
        phase_mask = (freq_axis > 0) & (freq_axis <= phase_lim_hz)
        bass_f2 = float(np.clip(phase_lim_hz, 20.0, 400.0))

        # Excess phase = measured - theoretical
        excess_phase = (p_rad_interp - theo_xo)

        # Weighting (your smoothstep bass weighting)
        phase_weight = np.zeros_like(freq_axis, dtype=float)
        f0, w0 = 20.0, 0.30
        f2, w2 = bass_f2, 0.00
        f1 = float(np.clip(0.5 * f2, 80.0, 140.0))
        w1 = float(np.clip(0.20 - 0.04 * ((f1 - 100.0) / 40.0), 0.14, 0.20))
        if f2 <= (f1 + 1.0):
            f2 = f1 + 1.0

        def smoothstep01(x):
            x = np.clip(x, 0.0, 1.0)
            return x*x*(3.0 - 2.0*x)

        bass_band = phase_mask & (freq_axis >= f0) & (freq_axis <= f2)
        f = freq_axis[bass_band]
        w = np.empty_like(f, dtype=float)
        seg1 = f <= f1
        x1 = (f[seg1] - f0) / (f1 - f0)
        s1 = smoothstep01(x1)
        w[seg1] = w0 + (w1 - w0) * s1
        seg2 = ~seg1
        x2 = (f[seg2] - f1) / (f2 - f1)
        s2 = smoothstep01(x2)
        w[seg2] = w1 + (w2 - w1) * s2
        phase_weight[bass_band] = np.maximum(phase_weight[bass_band], w)

        extra_phase = -excess_phase * phase_weight
        low_phase = (-theo_xo) + extra_phase

        if 'Mixed' in cfg.filter_type_str:
            f_center = float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0)
        else:
            f_center = float(getattr(cfg, "phase_limit", 1000.0) or 1000.0)

        f_center = float(np.clip(
            f_center, 20.0,
            float(freq_axis[-1] if freq_axis.size else 20000.0)
        ))

        safe_freqs = np.maximum(freq_axis, 1.0)
        octave_dist = np.log2(safe_freqs / f_center)
        mask = np.clip((octave_dist + 0.5), 0.0, 1.0)
        sm_mask = 3.0 * mask**2 - 2.0 * mask**3

        if 'Min' in cfg.filter_type_str:
            final_phase = min_p
        else:
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p

    # --- 10. IMPULSE GENERATION (common path) ---
    h_complex = total_mag * np.exp(1j * final_phase)
    raw_imp = scipy.fft.irfft(h_complex, n=n_fft)

    if 'Asym' in cfg.filter_type_str:
        shift = min(
            int(cfg.ir_window_ms_left * cfg.fs / 1000.0),
            int(n_fft * 0.4)
        )
        impulse = np.roll(raw_imp, shift)
    elif 'Min' in cfg.filter_type_str:
        impulse = raw_imp
    else:
        impulse = np.roll(raw_imp, n_fft // 2)
    
        # Alustava sijoittelu
    if 'Asym' in cfg.filter_type_str:
        shift = min(int(cfg.ir_window_ms_left * cfg.fs / 1000.0), int(n_fft * 0.4))
        impulse = np.roll(raw_imp, shift)
    elif 'Min' in cfg.filter_type_str: impulse = raw_imp
    else: impulse = np.roll(raw_imp, n_fft // 2)

    # --- 10. KORJATTU IKKUNOINTI (PEAK-CENTRIC) ---
    peak_idx = np.argmax(np.abs(impulse))
    n = len(impulse)
    window = np.zeros(n)
    
    # Ikkunan rajojen laskenta (Näytteet)
    s_left = int(cfg.ir_window_ms_left * cfg.fs / 1000.0)
    s_right = int(cfg.ir_window_ms * cfg.fs / 1000.0)
    
    # Logiikka:
    # Asym/Min: Käytä käyttäjän antamia arvoja kirjaimellisesti.
    # Linear/Mixed: Pakota symmetria (ir_window määrää säteen), jotta lineaarinen vaihe säilyy.
    if not ('Asym' in cfg.filter_type_str or 'Min' in cfg.filter_type_str):
        radius = s_right
        s_left = radius
        s_right = radius

    # Rakennetaan ikkuna piikin ympärille
    # 1. Vasen puoli (Nousu nollasta ykköseen)
    if s_left > 0:
        win_rise = np.sin(np.linspace(0, np.pi/2, s_left + 1))[:-1]**2
        start_idx = peak_idx - s_left
        
        # Sijoitus puskuriin (varovasti reunojen kanssa)
        if start_idx >= 0:
            window[start_idx : peak_idx] = win_rise
        else:
            # Jos piikki on liian lähellä alkua, leikataan ikkunan alkupää
            offset = -start_idx
            window[0 : peak_idx] = win_rise[offset:]
            
    # 2. Oikea puoli (Lasku ykkösestä nollaan)
    if s_right > 0:
        win_fall = np.cos(np.linspace(0, np.pi/2, s_right + 1))[1:]**2
        end_idx = peak_idx + 1 + s_right
        
        if end_idx <= n:
            window[peak_idx + 1 : end_idx] = win_fall
        else:
            len_available = n - (peak_idx + 1)
            window[peak_idx + 1 : n] = win_fall[:len_available]
            
    # Piikki on aina 1.0
    if 0 <= peak_idx < n:
        window[peak_idx] = 1.0
        
    # Sovelletaan ikkuna (nollaa kaiken datan ikkunan ulkopuolelta)
    impulse *= window
    impulse -= np.mean(impulse) # DC-poisto
    
    # --- 11. STATS & RETURN ---
    max_peak = np.max(np.abs(impulse))
    if cfg.do_normalize and max_peak > 0: impulse *= (0.89 / max_peak)

    stats = {
        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        'target_mags': target_mags.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'filter_mags': gain_db.tolist(),
        'confidence_mask': conf_mask.tolist(),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'meas_level_db_window': float(meas_level_db_window),
        'target_level_db_window': float(target_level_db_window),
        'offset_method': str(offset_method),
        'rt60_val': float(current_rt60),
        'rt60_band_avg': float(band_avg),
        'rt60_bands': rt60_bands,        
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'auto_headroom_db': float(auto_headroom_db),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total))
    }

    # attach comparison-mode stats (if any)
    if isinstance(cmp, dict) and cmp:
        stats.update(cmp)
        if stats.get('analysis_mode') != "comparison":
            stats['analysis_mode'] = "native"

    return impulse, stats
