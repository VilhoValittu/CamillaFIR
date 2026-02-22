
def scale_taps_with_fs(
    fs: int,
    base_fs: int = 44100,
    base_taps: int = 65536,
    allowed_taps=(
        512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288,
        1048576
    ),
) -> int:
    """Kasittelee signaalia tai dataa: scale taps with fs."""
    try:
        fs_i = int(fs)
        if fs_i <= 0:
            return int(base_taps)

        target = float(base_taps) * (fs_i / float(base_fs))
        for taps in allowed_taps:
            if int(taps) >= target:
                return int(taps)
        return int(allowed_taps[-1])
    except Exception:
        return int(base_taps)
