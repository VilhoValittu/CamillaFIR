# camillafir_io/measurements_loader.py
import os
from pywebio.pin import pin

from .measurements_txt import parse_measurements_from_path as parse_txt_path
from .measurements_txt import parse_measurements_from_bytes as parse_txt_bytes
from .measurements_wav import parse_measurements_from_wav_bytes, parse_measurements_from_wav_path


def parse_measurements_from_upload(
    file_dict,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    logger=None,
):
    try:
        if not file_dict:
            return None, None, None
        name = str(file_dict.get("filename", "") or "")
        content = file_dict.get("content", None)
        if content is None:
            return None, None, None
        ext = os.path.splitext(name)[1].lower()
        if ext == ".wav":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=logger,
            )
        # fallback: try wav by header "RIFF"
        if isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=logger,
            )
        return parse_txt_bytes(content)
    except Exception:
        return None, None, None


def load_measurements_lr(data: dict, *, logger=None):
    """
    Load measurements for Left/Right.
    Priority:
      1) Browser upload (pin.file_l / pin.file_r)
      2) Local path fields (local_path_l / local_path_r)
    """
    # UI-driven IR windows (ms) + smoothing for WAV parsing
    try:
        pre_ms = float(data.get("ir_window_left", 50.0) or 50.0)
    except Exception:
        pre_ms = 50.0
    try:
        post_ms = float(data.get("ir_window", 500.0) or 500.0)
    except Exception:
        post_ms = 500.0
    try:
        sl = int(data.get("smoothing_level", 0) or 0)
    except Exception:
        sl = 0

    # 1) Browser uploads
    try:
        up_l = pin["file_l"]
    except Exception:
        up_l = None
    try:
        up_r = pin["file_r"]
    except Exception:
        up_r = None

    has_up_l = isinstance(up_l, dict) and (up_l.get("content") is not None)
    has_up_r = isinstance(up_r, dict) and (up_r.get("content") is not None)

    if has_up_l and has_up_r:
        f_l, m_l, p_l = parse_measurements_from_upload(up_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
        f_r, m_r, p_r = parse_measurements_from_upload(up_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
        if f_l is not None and f_r is not None:
            return f_l, m_l, p_l, f_r, m_r, p_r

    # 2) Local paths fallback
    lp_l = str(data.get("local_path_l", "") or "").strip()
    lp_r = str(data.get("local_path_r", "") or "").strip()

    if lp_l and lp_r:
        ext_l = os.path.splitext(lp_l.strip().strip('"').strip("'"))[1].lower()
        ext_r = os.path.splitext(lp_r.strip().strip('"').strip("'"))[1].lower()

        # WAV local
        if ext_l == ".wav" and ext_r == ".wav":
            f_l, m_l, p_l = parse_measurements_from_wav_path(lp_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
            f_r, m_r, p_r = parse_measurements_from_wav_path(lp_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
            return f_l, m_l, p_l, f_r, m_r, p_r

        # TXT local (or anything else treated as TXT)
        f_l, m_l, p_l = parse_txt_path(lp_l, logger=logger)
        f_r, m_r, p_r = parse_txt_path(lp_r, logger=logger)
        return f_l, m_l, p_l, f_r, m_r, p_r

    return None, None, None, None, None, None
