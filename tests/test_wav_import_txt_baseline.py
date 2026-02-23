import io

import numpy as np
import scipy.io.wavfile

from src.camillafir.config.camillafir_pipeline import detect_is_wav_source
from src.camillafir.io.measurements_wav import parse_measurements_from_wav_bytes


def _make_test_wav_bytes(fs: int = 48000, n: int = 48000) -> bytes:
    x = np.zeros(n, dtype=np.float32)
    peak = 256
    x[peak] = 1.0
    tail_len = min(12000, n - peak - 1)
    if tail_len > 0:
        k = np.arange(tail_len, dtype=np.float32)
        x[peak + 1 : peak + 1 + tail_len] = 0.4 * np.exp(-k / 3500.0)

    bio = io.BytesIO()
    scipy.io.wavfile.write(bio, fs, x)
    return bio.getvalue()


def test_wav_parse_is_deterministic_vs_ui_knobs():
    wav_bytes = _make_test_wav_bytes()

    f1, m1, p1 = parse_measurements_from_wav_bytes(
        wav_bytes,
        pre_ms=5.0,
        post_ms=1700.0,
        smoothing_level=48,
    )
    f2, m2, p2 = parse_measurements_from_wav_bytes(
        wav_bytes,
        pre_ms=120.0,
        post_ms=300.0,
        smoothing_level=0,
    )

    assert f1 is not None and m1 is not None and p1 is not None
    assert f2 is not None and m2 is not None and p2 is not None
    assert np.allclose(f1, f2, atol=0.0, rtol=0.0)
    assert np.allclose(m1, m2, atol=0.0, rtol=0.0)
    assert np.allclose(p1, p2, atol=0.0, rtol=0.0)
    assert f1.size > 2
    # 48 kHz baseline should stay at fine FFT grid (<= 0.4 Hz/bin).
    assert (f1[1] - f1[0]) <= 0.4


def test_detect_is_wav_source_ignores_fmt_override():
    pin = {}

    data_txt_fmt_wav = {
        "local_path_l": r"C:\REW\uudet\L_subit_mukana.txt",
        "local_path_r": r"C:\REW\uudet\R_subit_mukana.txt",
        "fmt": "WAV",
    }
    assert detect_is_wav_source(data_txt_fmt_wav, pin) is False

    data_wav_fmt_txt = {
        "local_path_l": r"C:\REW\uudet\L.wav",
        "local_path_r": r"C:\REW\uudet\R.wav",
        "fmt": "TXT",
    }
    assert detect_is_wav_source(data_wav_fmt_txt, pin) is True
