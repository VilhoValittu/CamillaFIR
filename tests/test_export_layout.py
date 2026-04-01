import io
import zipfile

import numpy as np
import scipy.io.wavfile

from camillafir.config.results import FilterResult
from camillafir.ui.export_bundle import build_export_zip


def _make_result(fs: int = 48000) -> FilterResult:
    l_ir = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=np.float32)
    r_ir = np.asarray([0.0, -0.5, 0.5, 0.0], dtype=np.float32)
    freq = np.asarray([20.0, 100.0, 1000.0], dtype=float)
    zeros = np.zeros_like(freq)
    st = {
        "offset_method": "Auto",
        "smart_scan_range": [20.0, 200.0],
        "offset_db": 0.0,
        "eff_target_db": 0.0,
    }
    measurements = {
        "f_l": freq,
        "m_l": zeros,
        "p_l": zeros,
        "f_r": freq,
        "m_r": zeros,
        "p_r": zeros,
    }
    return FilterResult(
        fs=fs,
        taps=4,
        l_ir=l_ir,
        r_ir=r_ir,
        l_mag=zeros,
        r_mag=zeros,
        l_phase=zeros,
        r_phase=zeros,
        freq_axis=freq,
        l_st=dict(st),
        r_st=dict(st),
        measurements=measurements,
    )


def _base_data(layout: str) -> dict:
    return {
        "layout": layout,
        "target_curve_tag": "target",
        "multi_rate_opt": False,
        "program_version": "v.0.0.0",
        "filter_type": "Asymmetric",
        "mixed_freq": 200.0,
        "camillafir_automatic_mode": False,
    }


def test_build_export_zip_keeps_dual_mono_layout():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("Mono"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names

        yaml_name = next(name for name in names if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "filename: ../coeffs/L_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav" in yaml_text
        assert "filename: ../coeffs/R_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav" in yaml_text
        assert yaml_text.count("channel: 0") >= 2

        cfg_text = zf.read("Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in cfg_text
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in cfg_text


def test_build_export_zip_writes_single_stereo_wav_when_requested():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("Stereo"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        stereo_name = "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav"
        assert stereo_name in names
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names

        fs, stereo_data = scipy.io.wavfile.read(io.BytesIO(zf.read(stereo_name)))
        assert fs == 48000
        assert stereo_data.shape == (4, 2)
        assert np.allclose(stereo_data[:, 0], result.l_ir)
        assert np.allclose(stereo_data[:, 1], result.r_ir)

        yaml_name = next(name for name in names if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert yaml_text.count("filename: ../coeffs/Stereo_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav") == 2
        assert "channel: 0" in yaml_text
        assert "channel: 1" in yaml_text

        cfg_text = zf.read("Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert cfg_text.count("Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav") == 2
        assert "\n0\n0.0\n0.0\nStereo_Asymmetric_48000Hz_target_1200_290326_auto.wav\n1\n1.0\n1.0" in cfg_text


def test_build_export_zip_accepts_stable_layout_key():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("stereo"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
