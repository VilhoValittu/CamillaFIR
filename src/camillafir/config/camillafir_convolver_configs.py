def _tc_segment(target_curve_tag: str | None) -> str:
    tag = str(target_curve_tag or "").strip()
    return f"_{tag}" if tag else ""


def _title_suffix(program_version: str | None = None, winner_rank_score: float | None = None) -> str:
    parts: list[str] = []
    ver = str(program_version or "").strip()
    if ver:
        parts.append(ver)
    try:
        rank = float(winner_rank_score) if winner_rank_score is not None else float("nan")
    except Exception:
        rank = float("nan")
    if rank == rank and abs(rank) != float("inf"):
        parts.append(f"rank {rank:.3f}")
    return "" if not parts else " " + " ".join(parts)


def _normalize_layout(layout: str | None) -> str:
    try:
        layout_s = str(layout or "").strip().lower()
    except Exception:
        layout_s = ""
    return "Stereo" if "stereo" in layout_s else "Mono"


def filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    prefix: str = "",
):
    tc = _tc_segment(target_curve_tag)
    fs_s = str(fs_token)
    layout_norm = _normalize_layout(layout)

    if layout_norm == "Stereo":
        stereo_name = f"Stereo_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
        stereo_path = f"{prefix}{stereo_name}"
        return {
            "layout": layout_norm,
            "bundle_names": [stereo_name],
            "left_filename": stereo_path,
            "right_filename": stereo_path,
            "left_channel": 0,
            "right_channel": 1,
        }

    l_name = f"L_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
    r_name = f"R_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
    return {
        "layout": layout_norm,
        "bundle_names": [l_name, r_name],
        "left_filename": f"{prefix}{l_name}",
        "right_filename": f"{prefix}{r_name}",
        "left_channel": 0,
        "right_channel": 0,
    }


def sub_filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    prefix: str = "",
):
    fs_s = str(fs_token)
    sub_name = f"Sub_{ft_short}_{fs_s}Hz_{file_ts}_{irw_tag}.wav"
    return {
        "bundle_name": sub_name,
        "filename": f"{prefix}{sub_name}",
        "channel": 0,
    }


def generate_raspberry_yaml(
    fs,
    ft_short,
    file_ts,
    master_gain_db=0.0,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    program_version: str | None = None,
    winner_rank_score: float | None = None,
    include_sub: bool = False,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
):
    tc = _tc_segment(target_curve_tag)
    title_meta = _title_suffix(program_version=program_version, winner_rank_score=winner_rank_score)
    spec = filter_wav_export_spec(
        "$samplerate$",
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
        prefix="../coeffs/",
    )
    l_wav = str(spec["left_filename"])
    r_wav = str(spec["right_filename"])
    l_ch = int(spec["left_channel"])
    r_ch = int(spec["right_channel"])
    sub_spec = sub_filter_wav_export_spec(
        "$samplerate$",
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        prefix="../coeffs/",
    )
    sub_wav = str(sub_spec["filename"])
    sub_ch = int(sub_spec["channel"])

    include_sub = bool(include_sub)
    try:
        ap_freq_hz = float(sub_allpass_freq_hz) if sub_allpass_freq_hz is not None else float("nan")
    except Exception:
        ap_freq_hz = float("nan")
    try:
        ap_q = float(sub_allpass_q) if sub_allpass_q is not None else float("nan")
    except Exception:
        ap_q = float("nan")
    use_sub_allpass = bool(
        include_sub
        and ap_freq_hz == ap_freq_hz
        and ap_q == ap_q
        and abs(ap_freq_hz) != float("inf")
        and abs(ap_q) != float("inf")
        and ap_freq_hz > 0.0
        and ap_q > 0.0
    )

    playback_channels = 3 if include_sub else 2
    sub_filter_names = "mastergain, sub_allpass, ir_sub" if use_sub_allpass else "mastergain, ir_sub"

    lines = [
        "description: null",
        "devices:",
        "  capture:",
        "    type: Stdin",
        "    channels: 2",
        "    format: S32LE",
        "  playback:",
        "    type: Alsa",
        "    device: plughw:0,0",
        f"    channels: {playback_channels}",
        "    format: S32LE",
        f"  samplerate: {int(fs)}",
        "  enable_rate_adjust: true",
        "  chunksize: 2048",
        "  queuelimit: 1",
        "  volume_ramp_time: 150",
        "",
        "filters:",
        "  ir_left:",
        "    type: Conv",
        "    parameters:",
        "      type: Wav",
        f"      filename: {l_wav}",
        f"      channel: {l_ch}",
        "",
        "  ir_right:",
        "    type: Conv",
        "    parameters:",
        "      type: Wav",
        f"      filename: {r_wav}",
        f"      channel: {r_ch}",
    ]
    if include_sub:
        lines.extend(
            [
                "",
                "  ir_sub:",
                "    type: Conv",
                "    parameters:",
                "      type: Wav",
                f"      filename: {sub_wav}",
                f"      channel: {sub_ch}",
            ]
        )
        if use_sub_allpass:
            lines.extend(
                [
                    "",
                    "  sub_allpass:",
                    "    type: Biquad",
                    "    parameters:",
                    "      type: Allpass",
                    f"      freq: {ap_freq_hz:.3f}",
                    f"      q: {ap_q:.6f}",
                ]
            )
    lines.extend(
        [
            "",
            "  mastergain:",
            "    type: Gain",
            "    parameters:",
            "      gain: -4",
            "",
            "mixers:",
            "  stereo:",
            "    channels:",
            "      in: 2",
            f"      out: {playback_channels}",
            "    mapping:",
            "      - dest: 0",
            "        sources:",
            "          - channel: 0",
            "            gain: 0",
            "      - dest: 1",
            "        sources:",
            "          - channel: 1",
            "            gain: 0",
        ]
    )
    if include_sub:
        lines.extend(
            [
                "      - dest: 2",
                "        sources:",
                "          - channel: 0",
                "            gain: 0",
                "          - channel: 1",
                "            gain: 0",
            ]
        )
    lines.extend(
        [
            "",
            "pipeline:",
            "  - type: Mixer",
            "    name: stereo",
            "  - type: Filter",
            "    channels: [0]",
            "    names: [mastergain, ir_left]",
            "  - type: Filter",
            "    channels: [1]",
            "    names: [mastergain, ir_right]",
        ]
    )
    if include_sub:
        lines.extend(
            [
                "  - type: Filter",
                "    channels: [2]",
                f"    names: [{sub_filter_names}]",
            ]
        )
    lines.extend(
        [
            "",
            "processors: null",
            f"title: {ft_short} Window {irw_tag}{tc} {file_ts} {title_meta}",
        ]
    )
    return "\n".join(lines).strip()




def generate_hlc_config(
    fs,
    ft_short,
    file_ts,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
):
    """Rakentaa tai generoi: generate hlc config."""
    spec = filter_wav_export_spec(
        fs,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
    )
    l_name = str(spec["left_filename"])
    r_name = str(spec["right_filename"])
    l_ch = int(spec["left_channel"])
    r_ch = int(spec["right_channel"])

    config = [
        f"{int(fs)} 2 2 0",
        "0 0",
        "0 0",
        f"{l_name}",
        f"{l_ch}",
        "0.0",
        "0.0",
        f"{r_name}",
        f"{r_ch}",
        "1.0",
        "1.0"
    ]
    return "\n".join(config)
