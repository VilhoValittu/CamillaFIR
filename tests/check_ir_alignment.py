"""
check_ir_alignment.py  --  tarkistaa kahden WAV-impulssivasteen välisen
ajoituksen, napaisuuden ja tasoeron.

Käyttö:
    python tests/check_ir_alignment.py kaiutin.wav sub.wav
    python tests/check_ir_alignment.py kaiutin.wav sub.wav --xo 80 --channel 0

Argumentit:
    file_a        Ensimmäinen WAV (referenssi, esim. pääkaiutin)
    file_b        Toinen WAV (vertailtava, esim. sub tai toinen kanava)
    --xo HZ       Vertailufrekvenssi ristiinkytkentäalueen tarkistukseen (oletus: 80)
    --channel N   WAV-kanava 0-pohjaisesti (oletus: 0, eli vasen/mono)
    --max-lag MS  Suurin haettava viive-ero ms (oletus: 150)

Huomio REW-tiedostoista:
    REW vie WAV-tiedostot kiinteällä esivisiveellä (tyypillisesti 1 s = 48000
    näytettä 48 kHz:llä), joten suora huippupaikkojen vertailu antaa aina ~0 ms.
    Tämä ohjelma käyttää ristikorrelaatiota todellisen suhteellisen viiveen
    löytämiseen sekä ryhmäviivettä XO-pisteessä lisäinformaationa.
"""

import argparse
import sys
import numpy as np
import scipy.io.wavfile


# ---------------------------------------------------------------------------
# WAV-lataus ja esikäsittely
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> tuple[int, np.ndarray]:
    fs, data = scipy.io.wavfile.read(path)
    x = np.asarray(data)
    if x.ndim == 2:
        x = x[:, 0]
    if x.dtype.kind != "f":
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        else:
            x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
    x -= float(np.mean(x))
    return int(fs), x


def _select_channel(x: np.ndarray, ch: int) -> np.ndarray:
    if x.ndim == 2:
        ch = max(0, min(ch, x.shape[1] - 1))
        return x[:, ch].astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Apufunktiot
# ---------------------------------------------------------------------------

def _peak_index(x: np.ndarray) -> int:
    return int(np.argmax(np.abs(x)))


def _window_around_peak(x: np.ndarray, fs: int, pre_ms: float = 5.0, post_ms: float = 500.0) -> np.ndarray:
    peak = _peak_index(x)
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    start = max(0, peak - pre)
    end = min(len(x), peak + post + 1)
    return x[start:end]


def _freq_response(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Palauttaa (freqs_hz, magnitude_db, phase_deg) ikkunoidusta IR:stä."""
    n = max(len(x), 65536)
    n = 1 << (max(1, n) - 1).bit_length()
    X = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(X) + 1e-12)
    phase_deg = np.angle(X, deg=True)
    return freqs, mag_db, phase_deg


def _interp_at(freqs: np.ndarray, vals: np.ndarray, hz: float) -> float:
    return float(np.interp(hz, freqs, vals))


def _group_delay_ms(freqs: np.ndarray, phase_deg: np.ndarray, hz: float) -> float:
    """Ryhmäviive millisekunteina frekvenssipisteessä hz."""
    phase_rad = np.unwrap(np.deg2rad(phase_deg))
    # gd = -dφ/dω,  df = freqs[1]-freqs[0],  dω = 2π·df
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    gd_s = -np.diff(phase_rad) / (2.0 * np.pi * df)
    freqs_mid = (freqs[:-1] + freqs[1:]) / 2.0
    return float(np.interp(hz, freqs_mid, gd_s)) * 1000.0


# ---------------------------------------------------------------------------
# Ristikorrelaatiopohjainen ajoitusanalyysi
# ---------------------------------------------------------------------------

def _xcorr_timing(
    fs_a: int, x_a: np.ndarray,
    fs_b: int, x_b: np.ndarray,
    max_lag_ms: float = 150.0,
) -> dict:
    """
    Laskee ristikorrelaation impulssiikkunoiden välillä ja palauttaa
    suhteellisen viiveen (positiivinen = B myöhässä A:sta).

    Käyttää 50 ms esi- ja 600 ms jälkiikkunaa molemmista.
    """
    peak_a = _peak_index(x_a)
    peak_b = _peak_index(x_b)

    pre = int(0.050 * fs_a)
    post = int(0.600 * fs_a)
    seg_a = x_a[max(0, peak_a - pre): min(len(x_a), peak_a + post)].copy()
    seg_b = x_b[max(0, peak_b - pre): min(len(x_b), peak_b + post)].copy()

    # Normalisoi huipun mukaan
    seg_a /= (np.max(np.abs(seg_a)) + 1e-12)
    seg_b /= (np.max(np.abs(seg_b)) + 1e-12)

    n = min(len(seg_a), len(seg_b))
    seg_a = seg_a[:n]
    seg_b = seg_b[:n]

    # Jos signaalit ovat anti-korreloituneita (napaisuus käännetty), flipataan B
    # ennen xcorr-laskentaa jotta ajoitus löytyy oikein riippumatta napaisuudesta.
    xcorr_at_zero = float(np.dot(seg_a, seg_b))
    xcorr_polarity_flip = xcorr_at_zero < 0
    if xcorr_polarity_flip:
        seg_b = -seg_b

    # FFT-pohjainen ristikorrelaatio
    n_fft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(seg_a, n=n_fft)
    B = np.fft.rfft(seg_b, n=n_fft)
    xcorr = np.fft.irfft(A * np.conj(B), n=n_fft)

    # Hae paras viive rajoitetulta alueelta ±max_lag_ms
    max_lag_samp = min(int(round(max_lag_ms * fs_a / 1000.0)), n_fft // 2 - 1)
    # Positiiviset viiveet: xcorr[0..max_lag] (B myöhässä)
    # Negatiiviset viiveet: xcorr[n_fft-max_lag..n_fft] (B edellä)
    search = np.concatenate([xcorr[:max_lag_samp + 1], xcorr[n_fft - max_lag_samp:]])
    best_idx = int(np.argmax(np.abs(search)))

    if best_idx <= max_lag_samp:
        lag_samples = best_idx
    else:
        # negatiivinen alue → B edellä
        lag_samples = best_idx - max_lag_samp - 1 - max_lag_samp  # = -(max_lag - offset)
        lag_samples = -(2 * max_lag_samp - (best_idx - 1))

    offset_ms = lag_samples / float(fs_a) * 1000.0
    confidence = float(np.max(np.abs(search)))

    peaks_same = (peak_a == peak_b)

    return {
        "peak_a_sample": peak_a,
        "peak_b_sample": peak_b,
        "peak_a_ms": peak_a / float(fs_a) * 1000.0,
        "peak_b_ms": peak_b / float(fs_b) * 1000.0,
        "xcorr_lag_samples": lag_samples,
        "offset_ms": offset_ms,             # positiivinen = B myöhässä A:sta
        "xcorr_confidence": confidence,
        "xcorr_polarity_flip": xcorr_polarity_flip,   # True = signaalit olivat anti-korreloidut
        "peaks_same_position": peaks_same,
    }


# ---------------------------------------------------------------------------
# Muut tarkistukset
# ---------------------------------------------------------------------------

def _polarity_check(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    sign_a = float(np.sign(x_a[_peak_index(x_a)]))
    sign_b = float(np.sign(x_b[_peak_index(x_b)]))
    return {
        "sign_a": sign_a,
        "sign_b": sign_b,
        "inverted": sign_a != sign_b,
    }


def _level_check(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    peak_a = float(np.max(np.abs(x_a)))
    peak_b = float(np.max(np.abs(x_b)))
    rms_a = float(np.sqrt(np.mean(x_a ** 2)))
    rms_b = float(np.sqrt(np.mean(x_b ** 2)))
    peak_diff_db = 20.0 * np.log10(peak_b / (peak_a + 1e-12))
    rms_diff_db = 20.0 * np.log10(rms_b / (rms_a + 1e-12))
    return {
        "peak_a_dbfs": 20.0 * np.log10(peak_a + 1e-12),
        "peak_b_dbfs": 20.0 * np.log10(peak_b + 1e-12),
        "peak_diff_db": peak_diff_db,
        "rms_diff_db": rms_diff_db,
    }


def _phase_check(
    freqs_a: np.ndarray, phase_a: np.ndarray, gd_a_ms: float,
    freqs_b: np.ndarray, phase_b: np.ndarray, gd_b_ms: float,
    xo_hz: float,
) -> dict:
    ph_a = _interp_at(freqs_a, phase_a, xo_hz)
    ph_b = _interp_at(freqs_b, phase_b, xo_hz)
    diff = ((ph_b - ph_a) + 180.0) % 360.0 - 180.0
    gd_diff_ms = gd_b_ms - gd_a_ms
    # |diff| > 150° on käytännössä napaisuusinversio (ei pelkkä vaihevirhe)
    phase_polarity_inverted = abs(diff) > 150.0
    return {
        "phase_a_deg": ph_a,
        "phase_b_deg": ph_b,
        "phase_diff_deg": diff,
        "gd_a_ms": gd_a_ms,
        "gd_b_ms": gd_b_ms,
        "gd_diff_ms": gd_diff_ms,
        "xo_hz": xo_hz,
        "in_phase": abs(diff) <= 90.0,
        "phase_polarity_inverted": phase_polarity_inverted,
    }


# ---------------------------------------------------------------------------
# Raportti
# ---------------------------------------------------------------------------

PASS = "OK  "
WARN = "WARN"
FAIL = "FAIL"


def _report(file_a: str, file_b: str,
            timing: dict, polarity: dict, level: dict, phase: dict) -> int:
    problems = 0

    print()
    print("=" * 66)
    print("  IR-vertailu")
    print(f"  A: {file_a}")
    print(f"  B: {file_b}")
    print("=" * 66)

    # --- Ajoitus ---
    off = timing["offset_ms"]
    t_status = PASS if abs(off) < 5.0 else (WARN if abs(off) < 20.0 else FAIL)
    if t_status != PASS:
        problems += 1

    print(f"\n[AJOITUS]")
    print(f"  A-huippu         : {timing['peak_a_ms']:.2f} ms  (näyte {timing['peak_a_sample']})")
    print(f"  B-huippu         : {timing['peak_b_ms']:.2f} ms  (näyte {timing['peak_b_sample']})")

    if timing["peaks_same_position"]:
        print(f"  ! Huiput täsmälleen samassa näytteessä – REW on todennäköisesti")
        print(f"    tasannut molemmat mittaukset samaan viitepisteeseen.")
        print(f"    Ristikorrelaatio on tällöin paras saatavilla oleva arvio.")

    if timing["xcorr_polarity_flip"]:
        print(f"  ! Signaalit olivat anti-korreloituneita – napaisuusinversio huomioitu")
        print(f"    ajoituslaskennassa (B käännetty ennen ristikorrelaatiota).")

    if abs(off) < 0.1:
        off_str = "≈0.00 ms  (ei merkittävää eroa)"
    elif off > 0:
        off_str = f"+{off:.2f} ms  (B myöhässä A:sta)"
    else:
        off_str = f"{off:.2f} ms  (B edellä A:ta)"
    print(f"  Ristikorrelaatio : {off_str}")
    print(f"  [{t_status}] {'Ajoitus ok.' if t_status == PASS else 'Ajoituseroa havaittu – tarkista viive.'}")

    # --- Napaisuus ---
    pol = polarity
    # Vaiheesta saatu napaisuustieto on subwooferille luotettavampi kuin IR-huipun etumerkki
    phase_pol_inv = phase["phase_polarity_inverted"]
    peak_pol_inv = pol["inverted"]
    any_pol_inv = peak_pol_inv or phase_pol_inv
    p_status = FAIL if any_pol_inv else PASS
    if p_status != PASS:
        problems += 1
    print(f"\n[NAPAISUUS]")
    print(f"  IR-huipun etumerkki A : {'+' if pol['sign_a'] > 0 else '-'}")
    print(f"  IR-huipun etumerkki B : {'+' if pol['sign_b'] > 0 else '-'}")
    print(f"  Vaihe @ XO            : {phase['phase_diff_deg']:+.1f}°  "
          f"({'käänteinen' if phase_pol_inv else 'normaali'})")
    if peak_pol_inv and not phase_pol_inv:
        print(f"  [FAIL] IR-huipun etumerkki käänteinen (vaihe ok).")
    elif phase_pol_inv and not peak_pol_inv:
        print(f"  [FAIL] NAPAISUUSINVERSIO VAIHEEN PERUSTEELLA – vaihe-ero ≈ 180°.")
        print(f"         IR-huipun etumerkki sama, mutta sub toimii invertoidusti XO-alueella!")
    elif peak_pol_inv and phase_pol_inv:
        print(f"  [FAIL] KÄÄNTEINEN NAPAISUUS – sekä IR-huippu että vaihe invertoidut.")
    else:
        print(f"  [OK  ] Napaisuus ok.")

    # --- Taso ---
    lv = level
    ld = lv["peak_diff_db"]
    # Subeille huippuero voi olla suuri (rajoitettu taajuusalue → pienempi FFT-huippu)
    # Käytetään RMS-eroa pääkriteerinä
    rd = lv["rms_diff_db"]
    l_status = PASS if abs(rd) < 10.0 else WARN
    if l_status != PASS:
        problems += 1
    print(f"\n[TASO]")
    print(f"  Huippu A : {lv['peak_a_dbfs']:.1f} dBFS")
    print(f"  Huippu B : {lv['peak_b_dbfs']:.1f} dBFS")
    print(f"  Ero (B−A): {ld:+.1f} dB  (huippu)  /  {rd:+.1f} dB  (RMS)")
    if l_status == PASS:
        print(f"  [OK  ] Taso ok.")
    else:
        print(f"  [WARN] Suuri RMS-ero – tarkista mittaustaso tai vahvistin.")
        print(f"         (Huippuero voi olla normaali subille: rajoitettu kaistanleveys.)")

    # --- Vaihe ja ryhmäviive XO:ssa ---
    ph = phase
    diff = ph["phase_diff_deg"]
    ph_status = PASS if ph["in_phase"] else FAIL
    if ph_status != PASS:
        problems += 1
    gd_diff = ph["gd_diff_ms"]
    gd_status = PASS if abs(gd_diff) < 5.0 else (WARN if abs(gd_diff) < 15.0 else FAIL)

    print(f"\n[VAIHE & RYHMÄVIIVE @ {ph['xo_hz']:.0f} Hz]")
    print(f"  Vaihe A         : {ph['phase_a_deg']:+.1f}°")
    print(f"  Vaihe B         : {ph['phase_b_deg']:+.1f}°")
    print(f"  Vaihero (B−A)   : {diff:+.1f}°")
    if ph["phase_polarity_inverted"]:
        print(f"  [FAIL] NAPAISUUSINVERSIO – vaihe-ero ≈ ±180°, ei korjattavissa viiveellä.")
        print(f"         Korjaus: vaihda B:n kytkentä tai lisää 180° vaiheenkäännös suotimeen.")
    elif ph["in_phase"]:
        print(f"  [OK  ] Vaihesuhde ok (|ero| ≤ 90°).")
    else:
        print(f"  [FAIL] VAIHEVIRHE – ero > 90°, kaiuttimet summaavat huonosti!")
    print(f"  Ryhmäviive A    : {ph['gd_a_ms']:.2f} ms")
    print(f"  Ryhmäviive B    : {ph['gd_b_ms']:.2f} ms")
    gd_str = f"{gd_diff:+.2f} ms  ({'B myöhässä' if gd_diff > 0 else 'B edellä'})"
    print(f"  GD-ero (B−A)    : {gd_str}")
    print(f"  [{gd_status}] {'Ryhmäviive-ero ok.' if gd_status == PASS else 'Ryhmäviive-eroa – harkitse viivekorjausta.'}")

    # --- Yhteenveto ---
    print()
    print("-" * 66)
    if problems == 0:
        print("  YHTEENVETO: Kaikki tarkistukset läpäisty.")
    else:
        print(f"  YHTEENVETO: {problems} ongelma(a) havaittu – katso WARN/FAIL yllä.")
    print("=" * 66)
    print()

    return 1 if problems > 0 else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vertaa kahden WAV-impulssivasteen ajoitusta, napaisuutta ja tasoa."
    )
    parser.add_argument("file_a", help="Referenssi-WAV (pääkaiutin)")
    parser.add_argument("file_b", help="Vertailtava WAV (sub tai toinen kaiutin)")
    parser.add_argument("--xo", type=float, default=80.0, metavar="HZ",
                        help="Ristiinkytkentäfrekvenssi Hz (oletus: 80)")
    parser.add_argument("--channel", type=int, default=0, metavar="N",
                        help="WAV-kanava 0-pohjaisesti (oletus: 0)")
    parser.add_argument("--max-lag", type=float, default=150.0, dest="max_lag_ms", metavar="MS",
                        help="Suurin haettava viive-ero ms (oletus: 150)")
    args = parser.parse_args()

    try:
        fs_a, raw_a = _load_wav(args.file_a)
        fs_b, raw_b = _load_wav(args.file_b)
    except Exception as e:
        print(f"Virhe WAV-latauksessa: {e}", file=sys.stderr)
        return 2

    x_a = _select_channel(raw_a, args.channel)
    x_b = _select_channel(raw_b, args.channel)

    # Vaihevasteen ikkuna: lyhyt pre (5 ms) riittää vaihe-erolle
    win_a = _window_around_peak(x_a, fs_a)
    win_b = _window_around_peak(x_b, fs_b)
    freqs_a, _, phase_a = _freq_response(win_a, fs_a)
    freqs_b, _, phase_b = _freq_response(win_b, fs_b)

    # Ryhmäviive-ikkuna: pitkä pre (>= 3 jaksoa XO-taajuudella) luotettavuuden vuoksi
    gd_pre_ms = max(50.0, 4.0 / args.xo * 1000.0)
    gd_win_a = _window_around_peak(x_a, fs_a, pre_ms=gd_pre_ms, post_ms=500.0)
    gd_win_b = _window_around_peak(x_b, fs_b, pre_ms=gd_pre_ms, post_ms=500.0)
    gd_freqs_a, _, gd_phase_a = _freq_response(gd_win_a, fs_a)
    gd_freqs_b, _, gd_phase_b = _freq_response(gd_win_b, fs_b)

    gd_a = _group_delay_ms(gd_freqs_a, gd_phase_a, args.xo)
    gd_b = _group_delay_ms(gd_freqs_b, gd_phase_b, args.xo)

    timing = _xcorr_timing(fs_a, x_a, fs_b, x_b, max_lag_ms=args.max_lag_ms)
    polarity = _polarity_check(x_a, x_b)
    level = _level_check(x_a, x_b)
    phase = _phase_check(freqs_a, phase_a, gd_a, freqs_b, phase_b, gd_b, args.xo)

    return _report(args.file_a, args.file_b, timing, polarity, level, phase)


if __name__ == "__main__":
    sys.exit(main())
