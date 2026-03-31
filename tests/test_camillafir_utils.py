import scipy.fft

from camillafir.ui.camillafir_utils import scale_taps_with_fs


def test_scale_taps_with_fs_keeps_power_of_two_reference_points():
    assert scale_taps_with_fs(44100, base_taps=65536) == 65536
    assert scale_taps_with_fs(88200, base_taps=65536) == 131072
    assert scale_taps_with_fs(176400, base_taps=65536) == 262144


def test_scale_taps_with_fs_rounds_multirate_targets_to_real_fft_fast_lengths():
    scaled_48k = int(round(65536 * (48000 / 44100)))
    scaled_96k = int(round(65536 * (96000 / 44100)))
    scaled_192k = int(round(65536 * (192000 / 44100)))

    assert scale_taps_with_fs(48000, base_taps=65536) == int(scipy.fft.next_fast_len(scaled_48k, real=True))
    assert scale_taps_with_fs(96000, base_taps=65536) == int(scipy.fft.next_fast_len(scaled_96k, real=True))
    assert scale_taps_with_fs(192000, base_taps=65536) == int(scipy.fft.next_fast_len(scaled_192k, real=True))
