import io
import os

import numpy as np
import scipy.io.wavfile

from ..auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
)
from ..dsp.bass_integration import (
    compute_bass_integration_diagnostics,
    compute_direct_dac_bass_integration_diagnostics,
    sum_complex_responses,
)
from .measurement_bundle import BassIntegrationBundle, TransferData
from .measurements_txt import parse_measurements_from_path as parse_txt_path
from .measurements_txt import parse_measurements_from_bytes as parse_txt_bytes
from .measurements_wav import (
    detect_coherent_anchor_sample_from_wav_bytes,
    detect_coherent_anchor_sample_from_wav_path,
    parse_coherent_transfer_from_wav_bytes,
    parse_coherent_transfer_from_wav_path,
    parse_measurements_from_wav_bytes,
    parse_measurements_from_wav_path,
)


def _clean_local_path(p) -> str:
    """Normalisoi kayttajan antaman paikallisen tiedostopolun merkkijonoksi."""
    try:
        return str(p or "").strip().strip('"').strip("'")
    except Exception:
        return ""


def _get_uploaded_file(data: dict, key: str):
    """Palauttaa upload-dictionaryn datasta tai None."""
    try:
        v = data.get(key)
        if isinstance(v, dict) and v.get("content") is not None:
            return v
    except Exception:
        pass
    return None


def _get_local_path(data: dict, key: str) -> str:
    """Palauttaa paikallisen tiedostopolun datasta tai tyhjän merkkijonon."""
    return _clean_local_path(data.get(key, ""))


def _get_wav_window_params(data: dict) -> tuple[float, float, int]:
    try:
        pre_ms = float(data.get("ir_window_left", 85.0) or 85.0)
    except Exception:
        pre_ms = 10.0
    try:
        post_ms = float(data.get("ir_window_right", data.get("ir_window", 500.0)) or 500.0)
    except Exception:
        post_ms = 500.0
    try:
        sl = int(data.get("smoothing_level", 0) or 0)
    except Exception:
        sl = 0
    return float(pre_ms), float(post_ms), int(sl)


def _is_wav_upload(file_dict) -> bool:
    try:
        if not isinstance(file_dict, dict):
            return False
        name = str(file_dict.get("filename", "") or "").strip().lower()
        if name.endswith(".wav"):
            return True
        content = file_dict.get("content", None)
        return isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
    except Exception:
        return False


def _load_coherent_transfer_slot(
    data: dict,
    *,
    file_key: str,
    path_key: str,
    label: str,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    anchor_sample: int | None,
    logger=None,
):
    up = _get_uploaded_file(data, file_key)
    if up is not None:
        if not _is_wav_upload(up):
            if logger:
                logger.error(f"Bass Integration requires WAV uploads: {label}")
            return None
        return parse_coherent_transfer_from_wav_bytes(
            up.get("content", b""),
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            anchor_sample=anchor_sample,
            label=label,
            logger=logger,
        )

    lp = _get_local_path(data, path_key)
    if lp:
        if not str(lp).lower().endswith(".wav"):
            if logger:
                logger.error(f"Bass Integration requires WAV local files: {label}")
            return None
        return parse_coherent_transfer_from_wav_path(
            lp,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            anchor_sample=anchor_sample,
            label=label,
            logger=logger,
        )

    return None


def _detect_coherent_slot_anchor_sample(
    data: dict,
    *,
    file_key: str,
    path_key: str,
    logger=None,
) -> int | None:
    up = _get_uploaded_file(data, file_key)
    if up is not None:
        if not _is_wav_upload(up):
            return None
        return detect_coherent_anchor_sample_from_wav_bytes(
            up.get("content", b""),
            logger=logger,
        )

    lp = _get_local_path(data, path_key)
    if lp and str(lp).lower().endswith(".wav"):
        return detect_coherent_anchor_sample_from_wav_path(lp, logger=logger)

    return None


def _detect_shared_coherent_anchor_sample(data: dict, *, logger=None) -> int | None:
    peaks = []
    for file_key, path_key in (
        ("file_l_main", "local_path_l_main"),
        ("file_r_main", "local_path_r_main"),
        ("file_l_sub", "local_path_l_sub"),
        ("file_r_sub", "local_path_r_sub"),
    ):
        peak = _detect_coherent_slot_anchor_sample(
            data,
            file_key=file_key,
            path_key=path_key,
            logger=logger,
        )
        if peak is not None:
            peaks.append(int(peak))

    if not peaks:
        return None

    shared_anchor = int(round(sum(peaks) / float(len(peaks))))
    if logger:
        try:
            spread = int(max(peaks) - min(peaks))
            logger.info(f"Bass Integration shared WAV anchor sample: {shared_anchor} (spread {spread} samples)")
        except Exception:
            pass
    return shared_anchor


def _silent_transfer_like(template: TransferData, *, label: str) -> TransferData:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    complex_spec = np.zeros(freqs.shape, dtype=np.complex128)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=complex_spec,
        mag_db=np.full(freqs.shape, -240.0, dtype=float),
        phase_deg=np.zeros(freqs.shape, dtype=float),
        sample_rate=int(template.sample_rate),
        label=str(label or ""),
    )


def parse_measurements_from_upload(
    file_dict,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    logger=None,
):
    """
    Jasentaa selaimesta ladatun mittaustiedoston sisallon.

    Valitsee parserin tiedostopaateen tai RIFF-headerin perusteella:
    WAV -> WAV-parseri, muuten TXT-parseri.
    """
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
    Lataa vasemman ja oikean kanavan mittaukset ensisijaisuusjarjestyksessa.

    Jarjestys:
    1) selainlataukset (`data["file_l"]`, `data["file_r"]`)
    2) paikalliset polut (`local_path_l`, `local_path_r`)

    Palauttaa aina 6-arvoisen tuplen:
    `(f_l, m_l, p_l, f_r, m_r, p_r)`.
    """
    pre_ms, post_ms, sl = _get_wav_window_params(data)

    up_l = _get_uploaded_file(data, "file_l")
    up_r = _get_uploaded_file(data, "file_r")

    if up_l is not None and up_r is not None:
        f_l, m_l, p_l = parse_measurements_from_upload(up_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
        f_r, m_r, p_r = parse_measurements_from_upload(up_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
        if f_l is not None and f_r is not None:
            return f_l, m_l, p_l, f_r, m_r, p_r

    lp_l = _get_local_path(data, "local_path_l")
    lp_r = _get_local_path(data, "local_path_r")

    if lp_l and lp_r:
        ext_l = os.path.splitext(lp_l)[1].lower()
        ext_r = os.path.splitext(lp_r)[1].lower()

        if ext_l == ".wav" and ext_r == ".wav":
            f_l, m_l, p_l = parse_measurements_from_wav_path(lp_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
            f_r, m_r, p_r = parse_measurements_from_wav_path(lp_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger)
            return f_l, m_l, p_l, f_r, m_r, p_r

        f_l, m_l, p_l = parse_txt_path(lp_l, logger=logger)
        f_r, m_r, p_r = parse_txt_path(lp_r, logger=logger)
        return f_l, m_l, p_l, f_r, m_r, p_r

    return None, None, None, None, None, None


def _load_raw_wav_from_source(file_dict=None, local_path: str = "") -> tuple:
    """
    Lataa raaka WAV-data (ilman ikkunointia/tasoitusta) ylöslatauksesta tai
    paikallisesta tiedostosta. Palauttaa (ir_array, sample_rate) tai (None, 0).

    ir_array on float32-vektori (kanava 0), DC poistettu.
    """
    try:
        content = None
        if file_dict is not None:
            if not isinstance(file_dict, dict):
                return None, 0
            name = str(file_dict.get("filename", "") or "")
            raw = file_dict.get("content", None)
            if raw is None:
                return None, 0
            ext = os.path.splitext(name)[1].lower()
            if ext != ".wav":
                # Tarkista RIFF-header
                if not (isinstance(raw, (bytes, bytearray)) and len(raw) >= 4 and raw[:4] == b"RIFF"):
                    return None, 0
            content = raw
        elif local_path:
            lp = _clean_local_path(local_path)
            if not lp:
                return None, 0
            ext = os.path.splitext(lp)[1].lower()
            if ext != ".wav":
                return None, 0
            with open(lp, "rb") as f:
                content = f.read()
        else:
            return None, 0

        fs, data_raw = scipy.io.wavfile.read(io.BytesIO(bytes(content)))
        x = np.asarray(data_raw)
        if x.ndim == 2:
            x = x[:, 0]
        if x.dtype.kind != "f":
            if x.dtype == np.int16:
                x = x.astype(np.float32) / 32768.0
            elif x.dtype == np.int32:
                x = x.astype(np.float32) / 2147483648.0
            else:
                x = x.astype(np.float32)
        else:
            x = x.astype(np.float32, copy=False)
        x = x - float(np.mean(x))
        return x, int(fs)
    except Exception:
        return None, 0


def load_raw_irs_lr(
    data: dict,
    *,
    file_key_l: str = "file_l",
    path_key_l: str = "local_path_l",
    file_key_r: str = "file_r",
    path_key_r: str = "local_path_r",
    logger=None,
) -> tuple:
    """
    Lataa vasemman ja oikean kanavan raaka WAV IR -datat (ilman ikkunointia)
    annetuista L/R-sloteista.

    Palauttaa (raw_ir_l, fs_l, raw_ir_r, fs_r).
    Palauttaa (None, 0, None, 0) jos kumpaakaan ei löydy WAV-muodossa.
    """
    up_l = _get_uploaded_file(data, file_key_l)
    up_r = _get_uploaded_file(data, file_key_r)

    if up_l is not None and up_r is not None:
        ir_l, fs_l = _load_raw_wav_from_source(file_dict=up_l)
        ir_r, fs_r = _load_raw_wav_from_source(file_dict=up_r)
        if ir_l is not None and ir_r is not None:
            return ir_l, fs_l, ir_r, fs_r

    lp_l = _get_local_path(data, path_key_l)
    lp_r = _get_local_path(data, path_key_r)
    if lp_l and lp_r:
        ir_l, fs_l = _load_raw_wav_from_source(local_path=lp_l)
        ir_r, fs_r = _load_raw_wav_from_source(local_path=lp_r)
        if ir_l is not None and ir_r is not None:
            return ir_l, fs_l, ir_r, fs_r

    return None, 0, None, 0


def load_raw_ir_sub(data: dict, *, logger=None) -> tuple:
    """
    Lataa sub-kanavan raaka WAV IR -data bass integration -sloteista
    (file_l_sub / local_path_l_sub, fallback file_r_sub / local_path_r_sub).

    Palauttaa (raw_ir_sub, fs_sub) tai (None, 0).
    """
    up_ls = _get_uploaded_file(data, "file_l_sub")
    if up_ls is not None:
        ir, fs = _load_raw_wav_from_source(file_dict=up_ls)
        if ir is not None:
            return ir, fs

    lp_ls = _get_local_path(data, "local_path_l_sub")
    if lp_ls:
        ir, fs = _load_raw_wav_from_source(local_path=lp_ls)
        if ir is not None:
            return ir, fs

    up_rs = _get_uploaded_file(data, "file_r_sub")
    if up_rs is not None:
        ir, fs = _load_raw_wav_from_source(file_dict=up_rs)
        if ir is not None:
            return ir, fs

    lp_rs = _get_local_path(data, "local_path_r_sub")
    if lp_rs:
        ir, fs = _load_raw_wav_from_source(local_path=lp_rs)
        if ir is not None:
            return ir, fs

    return None, 0


def load_bass_integration_measurements(data: dict, *, logger=None):
    """
    Load decomposed AVR LFE+Main WAV measurements and build complex predicted totals.

    Returns a 7-value tuple:
    `(bundle, f_l, m_l, p_l, f_r, m_r, p_r)`.
    """
    bi_mode = str(
        data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
    ).strip().lower()
    is_direct_dac = bi_mode == "direct_dac"

    pre_ms, post_ms, sl = _get_wav_window_params(data)
    anchor_sample = _detect_shared_coherent_anchor_sample(data, logger=logger)
    l_main = _load_coherent_transfer_slot(
        data,
        file_key="file_l_main",
        path_key="local_path_l_main",
        label="L main only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    r_main = _load_coherent_transfer_slot(
        data,
        file_key="file_r_main",
        path_key="local_path_r_main",
        label="R main only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    l_sub = _load_coherent_transfer_slot(
        data,
        file_key="file_l_sub",
        path_key="local_path_l_sub",
        label="L sub only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    r_sub_source_present = bool(
        _get_uploaded_file(data, "file_r_sub") is not None
        or _get_local_path(data, "local_path_r_sub")
    )
    r_sub = _load_coherent_transfer_slot(
        data,
        file_key="file_r_sub",
        path_key="local_path_r_sub",
        label="R sub only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    if any(v is None for v in (l_main, r_main, l_sub)):
        return None, None, None, None, None, None, None
    if r_sub is None:
        if (not is_direct_dac) or r_sub_source_present:
            return None, None, None, None, None, None, None
        r_sub = _silent_transfer_like(l_sub, label="R sub absent")

    if any(v is None for v in (l_main, r_main, l_sub, r_sub)):
        return None, None, None, None, None, None, None

    sample_rates = {
        int(l_main.sample_rate),
        int(r_main.sample_rate),
        int(l_sub.sample_rate),
        int(r_sub.sample_rate),
    }
    if len(sample_rates) != 1:
        if logger:
            logger.error(f"Bass Integration sample rates do not match: {sorted(sample_rates)}")
        return None, None, None, None, None, None, None

    l_total = sum_complex_responses(l_main, l_sub, label="L total predicted")
    r_total = sum_complex_responses(r_main, r_sub, label="R total predicted")

    try:
        fc_hz = float(data.get("avr_crossover_hz", 80.0) or 80.0)
    except Exception:
        fc_hz = 80.0
    profile = str(data.get("bass_integration_profile", "safe") or "safe").strip().lower()
    try:
        guard_lo_ratio = float(
            data.get(
                "bass_integration_guard_lo_ratio",
                AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
            )
            or AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO
        )
    except Exception:
        guard_lo_ratio = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO
    try:
        guard_hi_ratio = float(
            data.get(
                "bass_integration_guard_hi_ratio",
                AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
            )
            or AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO
        )
    except Exception:
        guard_hi_ratio = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO

    bundle_base = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(fc_hz),
        profile=profile,
        diagnostics={},
    )
    if is_direct_dac:
        try:
            xo_order = max(1, int(round(float(data.get("sub_crossover_slope", 24) or 24.0))) // 6)
        except Exception:
            xo_order = 4
        try:
            sub_hpf_hz = float(data.get("sub_hpf_freq", 20.0) or 20.0)
        except Exception:
            sub_hpf_hz = 20.0
        try:
            sub_hpf_order = max(1, int(round(float(data.get("sub_hpf_slope", 12) or 12.0))) // 6)
        except Exception:
            sub_hpf_order = 2
        diagnostics = compute_direct_dac_bass_integration_diagnostics(
            bundle_base,
            float(fc_hz),
            profile,
            main_hpf_order=int(xo_order),
            sub_lpf_order=int(xo_order),
            sub_hpf_hz=float(sub_hpf_hz),
            sub_hpf_order=int(sub_hpf_order),
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
    else:
        diagnostics = compute_bass_integration_diagnostics(
            bundle_base,
            float(fc_hz),
            profile,
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(fc_hz),
        profile=profile,
        diagnostics=dict(diagnostics or {}),
    )
    if is_direct_dac:
        return (
            bundle,
            l_main.freqs_hz,
            l_main.mag_db,
            l_main.phase_deg,
            r_main.freqs_hz,
            r_main.mag_db,
            r_main.phase_deg,
        )
    return (
        bundle,
        l_total.freqs_hz,
        l_total.mag_db,
        l_total.phase_deg,
        r_total.freqs_hz,
        r_total.mag_db,
        r_total.phase_deg,
    )
