def _tc_segment(target_curve_tag: str | None) -> str:
    tag = str(target_curve_tag or "").strip()
    return f"_{tag}" if tag else ""


def generate_raspberry_yaml(
    fs,
    ft_short,
    file_ts,
    master_gain_db=0.0,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
):
    import textwrap

    tc = _tc_segment(target_curve_tag)
    l_wav = f'../coeffs/L_{ft_short}_$samplerate$Hz{tc}_{file_ts}_{irw_tag}.wav'
    r_wav = f'../coeffs/R_{ft_short}_$samplerate$Hz{tc}_{file_ts}_{irw_tag}.wav'

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
      enable_rate_adjust: true
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
    title: {ft_short} Window {irw_tag}{tc} {file_ts}
    """).strip()




def generate_hlc_config(fs, ft_short, file_ts, irw_tag: str = "auto", target_curve_tag: str = ""):
    """Rakentaa tai generoi: generate hlc config."""
    tc = _tc_segment(target_curve_tag)
    l_name = f"L_{ft_short}_{fs}Hz{tc}_{file_ts}_{irw_tag}.wav"
    r_name = f"R_{ft_short}_{fs}Hz{tc}_{file_ts}_{irw_tag}.wav"

    config = [
        f"{int(fs)} 2 2 0",
        "0 0",
        "0 0",
        f"{l_name}",
        "0",
        "0.0",
        "0.0",
        f"{r_name}",
        "0",
        "1.0",
        "1.0"
    ]
    return "\n".join(config)
