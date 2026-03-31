
import scipy.fft


def scale_taps_with_fs(
    fs: int,
    base_taps: int = 65536,
    base_fs: int = 44100,
) -> int:
    """Scale FIR tap count proportionally so filter time length stays constant."""
    try:
        fs_i = int(float(fs))
        base_taps_i = int(float(base_taps))
        base_fs_i = int(float(base_fs))
        if fs_i <= 0 or base_taps_i <= 0 or base_fs_i <= 0:
            return int(base_taps_i if base_taps_i > 0 else 65536)

        scaled = int(round(float(base_taps_i) * (float(fs_i) / float(base_fs_i))))
        scaled = max(1, scaled)
        return int(scipy.fft.next_fast_len(scaled, real=True))
    except Exception:
        return int(base_taps)
