"""
Convolver config generators (CamillaDSP YAML, HLC/BruteFIR/Convolver VST cfg).

These are intentionally isolated from UI/DSP logic so export paths can import
them without pulling the whole app.
"""

def generate_raspberry_yaml(fs, ft_short, file_ts, master_gain_db=0.0, irw_tag: str = "auto"):
    import textwrap

    # FIR .wav files (CamillaDSP replaces $samplerate$ at runtime)
    l_wav = f'../coeffs/L_{ft_short}_$samplerate$Hz_{file_ts}_{irw_tag}.wav'
    r_wav = f'../coeffs/R_{ft_short}_$samplerate$Hz_{file_ts}_{irw_tag}.wav'

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
    title: {ft_short} Window {irw_tag}
    """).strip()




def generate_hlc_config(fs, ft_short, file_ts, irw_tag: str = "auto"):
    """
    Luo standardin .cfg konfiguraatiotiedoston (HLC, Convolver VST, BruteFIR).
    Generoi tiedostonimet sisäisesti samoilla säännöillä kuin YAML-funktio.
    """
    # Generoidaan tiedostonimet täsmälleen samalla kaavalla kuin tallennuksessa
    l_name = f"L_{ft_short}_{fs}Hz_{file_ts}_{irw_tag}.wav"
    r_name = f"R_{ft_short}_{fs}Hz_{file_ts}_{irw_tag}.wav"

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