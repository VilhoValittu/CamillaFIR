import io
import logging
import os
import time
import zipfile
from typing import Any

import numpy as np
import scipy.io.wavfile

from ..app_paths import safe_filters_dir
from ..config.camillafir_convolver_configs import (
    filter_wav_export_spec,
    generate_raspberry_yaml,
    sub_filter_wav_export_spec,
)
from ..config.results import FilterResult
from .export_outputs import (
    _camilladsp_yaml_name,
    _direct_dac_yaml_export_settings,
    _export_version_tag,
    _export_winner_rank_score,
    _export_winner_rank_tag,
    _write_fs_outputs,
)
from .export_scoring import _build_export_ranking

logger = logging.getLogger("CamillaFIR")


def build_export_zip(
    *,
    data: dict,
    results: list[FilterResult],
    ft_short: str,
    file_ts: str,
    irw_tag: str = "auto",
    write_dashboards: bool = False,
    dash_fs: int | None = None,
) -> tuple[io.BytesIO, dict, dict]:
    """
    Build full export ZIP from pipeline results.

    Returns:
    - zip_buffer: in-memory ZIP payload
    - ui_dashboards: optional dashboard HTML payloads
    - perf: {"zip_png_s": float, "per_fs_stats": {fs: {"zip_png_s": float}}}
    """
    zip_buffer = io.BytesIO()
    ui_dashboards: dict[str, Any] = {}
    perf = {"zip_png_s": 0.0, "per_fs_stats": {}}
    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()
    multi_rate_on = bool(data.get("multi_rate_opt", False))
    ranking_context = _build_export_ranking(results)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in list(results or []):
            fs_v = int(result.fs)
            t0 = time.perf_counter()
            spec = filter_wav_export_spec(
                fs_v,
                ft_short,
                file_ts,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
                layout=data.get("layout", "Mono"),
            )

            if str(spec.get("layout")) == "Stereo":
                stereo_wav = io.BytesIO()
                stereo_ir = np.column_stack(
                    (
                        result.l_ir.astype("float32"),
                        result.r_ir.astype("float32"),
                    )
                )
                scipy.io.wavfile.write(stereo_wav, fs_v, stereo_ir)
                zf.writestr(str(spec["bundle_names"][0]), stereo_wav.getvalue())
            else:
                wav_l = io.BytesIO()
                wav_r = io.BytesIO()
                scipy.io.wavfile.write(wav_l, fs_v, result.l_ir.astype("float32"))
                scipy.io.wavfile.write(wav_r, fs_v, result.r_ir.astype("float32"))
                zf.writestr(str(spec["bundle_names"][0]), wav_l.getvalue())
                zf.writestr(str(spec["bundle_names"][1]), wav_r.getvalue())

            if getattr(result, "sub_ir", None) is not None and result.sub_ir.size > 0:
                wav_sub = io.BytesIO()
                scipy.io.wavfile.write(wav_sub, fs_v, result.sub_ir.astype("float32"))
                sub_name = str(
                    sub_filter_wav_export_spec(
                        fs_v,
                        ft_short,
                        file_ts,
                        irw_tag=irw_tag,
                    )["bundle_name"]
                )
                zf.writestr(sub_name, wav_sub.getvalue())

            write_dash = bool(
                write_dashboards
                and ((not multi_rate_on) or (dash_fs is not None and int(fs_v) == int(dash_fs)))
            )
            _write_fs_outputs(
                zf,
                data,
                fs_v,
                ft_short,
                file_ts,
                None,
                None,
                None,
                None,
                {},
                None,
                None,
                None,
                None,
                {},
                result=result,
                write_dashboards=write_dash,
                irw_tag=irw_tag,
                ui_dashboards=ui_dashboards if (dash_fs is not None and int(fs_v) == int(dash_fs)) else None,
                ranking_context=ranking_context,
            )

            dt = max(0.0, float(time.perf_counter() - t0))
            perf["zip_png_s"] = float(perf.get("zip_png_s", 0.0)) + dt
            slot = perf["per_fs_stats"].setdefault(int(fs_v), {})
            slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + dt

        if multi_rate_on:
            include_sub = bool(results) and all(
                getattr(getattr(result, "sub_ir", None), "size", 0) > 0
                for result in list(results or [])
            )
            yaml_settings = _direct_dac_yaml_export_settings(data, include_sub=include_sub)
            yaml_content = generate_raspberry_yaml(
                int(data.get("fs") or 44100),
                ft_short,
                file_ts,
                master_gain_db=0.0,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
                layout=data.get("layout", "Mono"),
                program_version=str(data.get("program_version", "") or "").strip(),
                winner_rank_score=_export_winner_rank_score(data),
                include_sub=bool(yaml_settings.get("include_sub", False)),
                sub_allpass_freq_hz=yaml_settings.get("sub_allpass_freq_hz"),
                sub_allpass_q=yaml_settings.get("sub_allpass_q"),
            )
            zf.writestr(
                _camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag),
                yaml_content,
            )

    return zip_buffer, ui_dashboards, perf


def save_export_bundle(
    zip_buffer: io.BytesIO,
    *,
    data: dict | None = None,
    ft_short: str,
    irw_tag: str,
    target_curve_tag: str,
    ts: str,
    output_dir: str | None = None,
    program_version: str | None = None,
) -> tuple[str, str, str]:
    ver_tag = _export_version_tag(data, program_version=program_version)
    rank_tag = _export_winner_rank_tag(data)
    filters_dir = safe_filters_dir(output_dir, program_version=program_version)
    logger.info(f"Export filters directory: {filters_dir}")
    parts = ["CamillaFIR", str(ft_short), str(irw_tag), str(target_curve_tag), str(ver_tag)]
    if rank_tag:
        parts.append(rank_tag)
    parts.append(str(ts))
    fname = "_".join(parts) + ".zip"
    out_path = os.path.join(filters_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Saved: {os.path.abspath(out_path)}"
    except Exception:
        save_msg = "Zip saving failed."
    return fname, os.path.abspath(filters_dir), save_msg
