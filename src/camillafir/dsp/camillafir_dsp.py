import numpy as np
import scipy.ndimage
import copy
import logging
logger = logging.getLogger("CamillaFIR.dsp")
from camillafir.config.models import FilterConfig
from .dsp_correction import run_correction_stage
from .dsp_phase_ir import run_phase_ir_stage
from .dsp_preprocess import run_preprocess
from .dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero
from .dsp_utils import safe_range as _safe_range



def apply_confidence_weighted_target_pull(
    target_db,
    measured_db,
    confidence_mask,
    *,
    conf_floor: float = 0.07,
    conf_ceil: float = 0.95,
    freq_axis=None,
    freq_limit_hz: float | None = 400.0,
    gamma_cut: float = 0.70,
    gamma_boost: float = 1.20,
    return_telemetry: bool = False,
):
    """
    Sekoittaa target-kayraa kohti mitattua kayraa confidence-maskin perusteella.

    Korkea confidence painottaa targetia ja matala confidence painottaa mittausta.
    Pull voidaan rajata taajuusalueelle `freq_limit_hz` ja painottaa erikseen
    cut/boost-suuntiin `gamma_cut`- ja `gamma_boost`-parametreilla.

    Kun `return_telemetry=True`, palauttaa myos diagnostiikkadatan:
    `w_eff`, `pull_mask` ja `pull_strength`.
    """
    try:
        t = np.asarray(target_db, dtype=float)
        m = np.asarray(measured_db, dtype=float)
        c = np.asarray(confidence_mask, dtype=float) if confidence_mask is not None else None

        if c is None or t.size < 8 or t.shape != m.shape or t.shape != c.shape:
            return t

        if conf_ceil <= conf_floor + 1e-9:
            return t
        if freq_limit_hz is not None and freq_axis is not None:
            f = np.asarray(freq_axis, dtype=float)
            if f.shape == t.shape:
                pull_mask = (f > 0.0) & (f <= float(freq_limit_hz))
            else:
                pull_mask = None
        else:
            pull_mask = None

        c = np.clip(c, conf_floor, conf_ceil)
        w = (c - conf_floor) / (conf_ceil - conf_floor)
        w = np.clip(w, 0.0, 1.0)

        is_cut = (t < 0.0)
        gc = float(gamma_cut) if np.isfinite(float(gamma_cut)) and float(gamma_cut) > 0 else 1.0
        gb = float(gamma_boost) if np.isfinite(float(gamma_boost)) and float(gamma_boost) > 0 else 1.0
        w_eff = np.where(is_cut, w ** gc, w ** gb)
        w_eff = np.clip(w_eff, 0.0, 1.0)

        out = (w_eff * t) + ((1.0 - w_eff) * m)

        if pull_mask is not None:
            out = np.where(pull_mask, out, t)
        if not return_telemetry:
            return out

        try:
            if pull_mask is None:
                pm = np.ones_like(w_eff, dtype=bool)
            else:
                pm = np.asarray(pull_mask, dtype=bool)
            pull_strength = np.clip(1.0 - w_eff, 0.0, 1.0)
            return out, {"w_eff": w_eff, "pull_mask": pm, "pull_strength": pull_strength}
        except Exception:
            return out, {"w_eff": w_eff, "pull_mask": None, "pull_strength": None}
    except Exception:
        out = np.asarray(target_db, dtype=float)
        return (out, None) if return_telemetry else out

def _limit_gd_gradient_ms_per_oct(
    freq_axis,
    phase_rad,
    *,
    mask=None,
    max_grad_ms_per_oct=30.0,
    f_min=20.0,
    f_max=250.0,
    grad_smooth_sigma=0.8,
    soft_limit=True,
):
    """
    Rajoittaa ryhmaviiveen gradienttia (ms/oktaavi) valitulla taajuusalueella.

    Menetelma:
    1) laskee vaiheesta ryhmaviiveen
    2) laskee ryhmaviiveen gradientin log-taajuusakselilla
    3) rajaa gradientin pehmeasti (`tanh`) tai kovalla clipilla
    4) integroi takaisin rajatuksi vaiheeksi

    Tarkoitus on vaimentaa epavakaita bassopiikkeja ilman, etta koko vaihe
    muokkautuu liikaa.
    """
    f = np.asarray(freq_axis, dtype=float)
    ph = np.asarray(phase_rad, dtype=float)
    if f.size < 16 or ph.size != f.size:
        return ph

    m = (np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(f, dtype=bool))
    m = m & (f > 0.0) & (f >= float(f_min)) & (f <= float(f_max))
    if not np.any(m):
        return ph

    idx = np.where(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    ff = f[i0:i1+1]
    pp = ph[i0:i1+1]
    if ff.size < 16:
        return ph

    pp_u = np.unwrap(pp)

    df = np.gradient(ff) + 1e-12
    gd_ms = (-np.gradient(pp_u) / (2.0 * np.pi * df)) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)

    log2f = np.log2(np.maximum(ff, 1e-9))
    dlog = np.gradient(log2f) + 1e-12
    gd_grad = np.gradient(gd_ms) / dlog

    if grad_smooth_sigma and float(grad_smooth_sigma) > 0.0:
        try:
            gd_grad = scipy.ndimage.gaussian_filter1d(gd_grad, sigma=float(grad_smooth_sigma))
        except Exception:
            pass

    lim = float(max(0.1, max_grad_ms_per_oct))

    if soft_limit:
        gd_grad_l = lim * np.tanh(gd_grad / lim)
    else:
        gd_grad_l = np.clip(gd_grad, -lim, lim)

    k0 = int(ff.size // 2)
    gd0 = float(gd_ms[k0])
    gd_l = np.empty_like(gd_ms)
    gd_l[k0] = gd0
    for k in range(k0 + 1, ff.size):
        gd_l[k] = gd_l[k-1] + gd_grad_l[k-1] * (log2f[k] - log2f[k-1])
    for k in range(k0 - 1, -1, -1):
        gd_l[k] = gd_l[k+1] - gd_grad_l[k+1] * (log2f[k+1] - log2f[k])

    dphi_df = -2.0 * np.pi * (gd_l / 1000.0)
    phi_l = np.empty_like(pp_u)
    phi_l[k0] = float(pp_u[k0])
    for k in range(k0 + 1, ff.size):
        phi_l[k] = phi_l[k-1] + 0.5 * (dphi_df[k-1] + dphi_df[k]) * (ff[k] - ff[k-1])
    for k in range(k0 - 1, -1, -1):
        phi_l[k] = phi_l[k+1] - 0.5 * (dphi_df[k+1] + dphi_df[k]) * (ff[k+1] - ff[k])

    out = ph.copy()
    out[i0:i1+1] = phi_l
    return out



def _stage_probe(stage_name, freq_axis, arr_db, mask_c, global_gain_db=0.0, auto_headroom_db=0.0, logger_obj=None):
    """
    Kerää kevyet vaihekohtaiset debug-mittarit korjauskayrasta.

    Palauttaa sanakirjan, jossa on mm. boost/cut-huiput, boost/cut-binien
    lukumaara seka nettopiikki global gain + auto headroom huomioiden.
    Jos logger on annettu, kirjoittaa saman tiedon lokiin.
    """
    try:
        out = {
            "stage": str(stage_name),
            "boost_peak_db": 0.0,
            "cut_peak_db": 0.0,
            "boost_bins": 0,
            "cut_bins": 0,
            "net_boost_peak_db": 0.0
        }
        if arr_db is None or mask_c is None or not np.any(mask_c):
            return out
        v = np.asarray(arr_db, dtype=float)
        m = np.asarray(mask_c, dtype=bool)
        vv = v[m]
        out["boost_peak_db"] = float(np.max(vv)) if vv.size else 0.0
        out["cut_peak_db"] = float(np.min(vv)) if vv.size else 0.0
        out["boost_bins"] = int(np.sum(vv > 1e-6))
        out["cut_bins"] = int(np.sum(vv < -1e-6))
        out["net_boost_peak_db"] = float(out["boost_peak_db"] + float(global_gain_db) + float(auto_headroom_db))
        if logger_obj is not None:
            logger_obj.info(
                f"StageProbe[{out['stage']}]: "
                f"boost_peak={out['boost_peak_db']:.2f} dB, cut_peak={out['cut_peak_db']:.2f} dB, "
                f"boost_bins={out['boost_bins']}, cut_bins={out['cut_bins']}, "
                f"net_boost_peak={out['net_boost_peak_db']:.2f} dB"
            )
        return out
    except Exception:
        return {
            "stage": str(stage_name),
            "boost_peak_db": 0.0,
            "cut_peak_db": 0.0,
            "boost_bins": 0,
            "cut_bins": 0,
            "net_boost_peak_db": 0.0
        }

def apply_hpf_to_mags(freqs, mags, cutoff, order):
    """
    Lisaa Butterworth-HPF:n vaimennuksen dB-kayraan.

    Funktio ei rakenna erillista suodintaikatasossa, vaan laskee taajuuskohtaisen
    attenuaation analyyttisesti ja summaa sen suoraan `mags`-arvoihin.
    """
    if cutoff <= 0 or order <= 0:
        return mags
    f = np.asarray(freqs, dtype=float)
    if f.size > 1 and f[0] == 0.0:
        f = f.copy()
        f[0] = f[1] if f[1] > 0 else 1e-6
    with np.errstate(divide='ignore'):
        attenuation = -10 * np.log10(1 + (cutoff / (f + 1e-12))**(2 * order))
    return mags + attenuation

def interpolate_response(input_freqs, input_values, target_freqs):
    """Interpoloi vasteen lineaarisesti kohde-taajuusakselille."""
    return np.interp(target_freqs, input_freqs, input_values)


def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig):
    prep = run_preprocess(freqs, meas_mags, raw_phases, cfg)
    f_in = prep.f_in
    m_in = prep.m_in
    n_fft = prep.ctx.n_fft
    freq_axis = prep.ctx.freq_axis
    gain_db = prep.ctx.gain_db
    st = prep.ctx.st
    target_mags = prep.ctx.target_mags
    m_interp = prep.m_interp
    p_rad_interp = prep.p_rad_interp
    delay_slope = prep.delay_slope
    m_plot_db = prep.m_plot_db
    complex_meas = prep.complex_meas
    m_anal = prep.m_anal
    conf_mask = prep.conf_mask
    reflections = prep.reflections
    cmp = prep.cmp
    analysis_mode = prep.analysis_mode
    is_psy = prep.is_psy


    corr = run_correction_stage(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=f_in,
        m_in=m_in,
        reflections=reflections,
        st=st,
        m_anal=m_anal,
        m_plot_db=m_plot_db,
        is_psy=is_psy,
        cmp=cmp,
        analysis_mode=analysis_mode,
        gain_db=gain_db,
        conf_mask=conf_mask,
        complex_meas=complex_meas,
        logger=logger,
        interpolate_response_fn=interpolate_response,
        apply_confidence_weighted_target_pull_fn=apply_confidence_weighted_target_pull,
        stage_probe_fn=_stage_probe,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )

    current_rt60 = corr["current_rt60"]
    rt60_bands = corr["rt60_bands"]
    band_avg = corr["band_avg"]
    target_mags = corr["target_mags"]
    hpf_f = corr["hpf_f"]
    hpf_order = corr["hpf_order"]
    target_level_db = corr["target_level_db"]
    calc_offset_db = corr["calc_offset_db"]
    meas_level_db_window = corr["meas_level_db_window"]
    target_level_db_window = corr["target_level_db_window"]
    offset_method = corr["offset_method"]
    s_min = corr["s_min"]
    s_max = corr["s_max"]
    target_shift_db = corr["target_shift_db"]
    cmp = corr["cmp"]
    analysis_mode = corr["analysis_mode"]
    gain_db = corr["gain_db"]
    afdw_on = corr["afdw_on"]
    base_sigma = corr["base_sigma"]
    _filter_smooth = corr["_filter_smooth"]
    df_mode = corr["df_mode"]
    raw_g = corr["raw_g"]
    final_g = corr["final_g"]
    mask_c = corr["mask_c"]
    stage_probes = corr["stage_probes"]
    use_bassfirst = corr["use_bassfirst"]
    bf_room_mode = corr["bf_room_mode"]
    bf_rel = corr["bf_rel"]
    bf_conf_for_smoothing = corr["bf_conf_for_smoothing"]
    boost_peak_db = corr["boost_peak_db"]
    cut_peak_db = corr["cut_peak_db"]
    n_boost = corr["n_boost"]
    boost_cand_peak = corr["boost_cand_peak"]
    n_boost_cand = corr["n_boost_cand"]
    n_boost_cand_low = corr["n_boost_cand_low"]
    n_boost_cand_exc = corr["n_boost_cand_exc"]
    softclip_boost_bins = corr["softclip_boost_bins"]
    softclip_cut_bins = corr["softclip_cut_bins"]
    over_boost = corr["over_boost"]
    over_cut = corr["over_cut"]
    hardclamp_boost_bins = corr["hardclamp_boost_bins"]
    hardclamp_cut_bins = corr["hardclamp_cut_bins"]
    hard_over_boost = corr["hard_over_boost"]
    hard_over_cut = corr["hard_over_cut"]

    phase_ir = run_phase_ir_stage(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        _filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        apply_hpf_to_mags_fn=apply_hpf_to_mags,
        limit_gd_gradient_ms_per_oct_fn=_limit_gd_gradient_ms_per_oct,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )

    impulse = phase_ir["impulse"]
    gain_db = phase_ir["gain_db"]
    auto_global_gain_db = phase_ir.get("auto_global_gain_db", 0.0)
    gain_margin_db = phase_ir.get("gain_margin_db", 0.0)
    auto_headroom_db = phase_ir["auto_headroom_db"]
    current_peak_gain = phase_ir["current_peak_gain"]
    final_gain_total = phase_ir["final_gain_total"]

    max_peak = np.max(np.abs(impulse))
    if cfg.do_normalize and max_peak > 0: impulse *= (0.89 / max_peak)

    stats = {

        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        'mag_c_min': float(getattr(cfg, 'mag_c_min', 0.0) or 0.0),
        'mag_c_max': float(getattr(cfg, 'mag_c_max', 0.0) or 0.0),
        'target_mags': target_mags.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'filter_mags': gain_db.tolist(),
        'confidence_mask': conf_mask.tolist(),
        'afdw_active': bool(afdw_on),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'meas_level_db_window': float(meas_level_db_window),
        'target_level_db_window': float(target_level_db_window),
        'offset_method': str(offset_method),
            'tilt_slope_db_per_oct': (
                float(getattr(cfg, "_lvl_tilt_slope_db_per_oct"))
                if getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) is not None
                else None
            ),
        'rt60_val': float(current_rt60),
        'rt60_band_avg': float(band_avg),
        'rt60_bands': rt60_bands,        
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'do_normalize': bool(getattr(cfg, 'do_normalize', False)),
        'gain_margin_db': float(gain_margin_db),
        'auto_global_gain_db': float(auto_global_gain_db),
        'auto_headroom_db': float(auto_headroom_db),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total)),

        'max_boost_db': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_effective': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_user': float(getattr(cfg, 'max_boost_db_user', getattr(cfg, 'max_boost_db', 0.0)) or 0.0),
        'max_safe_boost_db': float(getattr(cfg, 'max_safe_boost_db', 0.0) or 0.0),
        'max_cut_db': float(abs(float(getattr(cfg, 'max_cut_db', 15.0) or 15.0))),
        'low_bass_cut_hz': _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 40.0),
        'exc_prot': bool(getattr(cfg, 'exc_prot', False)),
        'exc_freq': float(getattr(cfg, 'exc_freq', 0.0) or 0.0),
        'max_slope_db_per_oct': float(getattr(cfg, 'max_slope_db_per_oct', 0.0) or 0.0),
        'max_slope_boost_db_per_oct': float(getattr(cfg, 'max_slope_boost_db_per_oct', 0.0) or 0.0),
        'max_slope_cut_db_per_oct': float(getattr(cfg, 'max_slope_cut_db_per_oct', 0.0) or 0.0),
        

        'boost_peak_db': float(locals().get('boost_peak_db', 0.0)),
        'cut_peak_db': float(locals().get('cut_peak_db', 0.0)),
        'boost_bins': int(locals().get('n_boost', 0)),
        'boost_candidate_peak_db': float(locals().get('boost_cand_peak', 0.0)),
        'boost_candidate_bins': int(locals().get('n_boost_cand', 0)),
        'boost_candidate_bins_lowbass': int(locals().get('n_boost_cand_low', 0)),
        'boost_candidate_bins_excprot': int(locals().get('n_boost_cand_exc', 0)),


        'bass_first_ai': bool(locals().get('use_bassfirst', False)),

        'bass_first_mode_peak_hz': (
            float(freq_axis[int(np.argmax(np.asarray(bf_room_mode)))])
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_mode_peak_score': (
            float(np.max(np.asarray(bf_room_mode)))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_conf_floor_applied': (
            bool(
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and (locals().get('bf_rel', None) is not None)
                and np.any(
                    np.asarray(bf_conf_for_smoothing) > (np.asarray(bf_rel) + 1e-6)
                )
            )
            if (
                locals().get('bf_conf_for_smoothing', None) is not None
                and locals().get('bf_rel', None) is not None
            )
            else False
        ),


        'bass_first_rel_mean_20_200': (
            float(np.mean(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_rel_min_20_200': (
            float(np.min(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_mean_20_200': (
            float(np.mean(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_min_20_200': (
            float(np.min(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_roommode_max_20_200': (
            float(np.max(np.asarray(bf_room_mode)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

    }


    try:
        if isinstance(st, dict) and st:
            stats.update(st)
    except Exception:
        pass

    try:
        if bool(afdw_on) and (afdw_bw_oct is not None):
            stats['afdw_bw_oct'] = np.asarray(afdw_bw_oct, dtype=float).tolist()
            stats['afdw_bw_min_oct'] = float(afdw_bw_min_oct) if afdw_bw_min_oct is not None else None
            stats['afdw_bw_mean_oct'] = float(afdw_bw_mean_oct) if afdw_bw_mean_oct is not None else None
            stats['afdw_bw_max_oct'] = float(afdw_bw_max_oct) if afdw_bw_max_oct is not None else None
            stats['afdw_bw_min_hz'] = float(afdw_bw_min_hz) if afdw_bw_min_hz is not None else None
            stats['afdw_bw_max_hz'] = float(afdw_bw_max_hz) if afdw_bw_max_hz is not None else None
    except Exception:
        pass

    try:
        stats["stage_probes"] = {k: dict(v) for k, v in stage_probes.items()} if isinstance(stage_probes, dict) else {}
    except Exception:
        stats["stage_probes"] = {}

    try:
        stats['softclip_boost_bins'] = int(locals().get('softclip_boost_bins', 0))
        stats['softclip_cut_bins']   = int(locals().get('softclip_cut_bins', 0))
        stats['softclip_worst_over_boost_db'] = float(locals().get('over_boost', 0.0))
        stats['softclip_worst_over_cut_db']   = float(locals().get('over_cut', 0.0))

        stats['hardclamp_boost_bins'] = int(locals().get('hardclamp_boost_bins', 0))
        stats['hardclamp_cut_bins']   = int(locals().get('hardclamp_cut_bins', 0))
        stats['hardclamp_worst_over_boost_db'] = float(locals().get('hard_over_boost', 0.0))
        stats['hardclamp_worst_over_cut_db']   = float(locals().get('hard_over_cut', 0.0))

        stats['clamp_summary'] = (
            f"soft_clip: boost={stats['softclip_boost_bins']} cut={stats['softclip_cut_bins']} "
            f"(worst_over_boost={stats['softclip_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['softclip_worst_over_cut_db']:.2f} dB); "
            f"hard_clamp: boost={stats['hardclamp_boost_bins']} cut={stats['hardclamp_cut_bins']} "
            f"(worst_over_boost={stats['hardclamp_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['hardclamp_worst_over_cut_db']:.2f} dB)"
        )
    except Exception:
        stats['clamp_summary'] = "n/a"


    try:
        max_boost_db_cfg = float(getattr(cfg, 'max_boost_db', 0.0) or 0.0)
        low_hz_cfg = float(getattr(cfg, 'low_bass_cut_hz', 40.0) or 40.0)
        exc_on = bool(getattr(cfg, 'exc_prot', False))
        exc_f_cfg = float(getattr(cfg, 'exc_freq', 0.0) or 0.0)
        do_norm = bool(getattr(cfg, 'do_normalize', False))
        g_global = float(stats.get('auto_global_gain_db', getattr(cfg, 'global_gain_db', 0.0)) or 0.0)

        boost_bins_post = int(stats.get('boost_bins', 0) or 0)
        boost_bins_cand = int(stats.get('boost_candidate_bins', 0) or 0)
        boost_bins_cand_low = int(stats.get('boost_candidate_bins_lowbass', 0) or 0)
        boost_bins_cand_exc = int(stats.get('boost_candidate_bins_excprot', 0) or 0)

        boost_peak_post = float(stats.get('boost_peak_db', 0.0) or 0.0)
        net_boost_peak = boost_peak_post + g_global + float(stats.get('auto_headroom_db', 0.0) or 0.0)
        stats['net_boost_peak_db'] = float(net_boost_peak)

        reasons = []

        if max_boost_db_cfg <= 0.0:
            reasons.append("max_boost_db <= 0 (boost disabled)")

        if boost_bins_cand == 0 and boost_bins_post == 0:
            reasons.append("no boost candidates (algorithm produced only cuts in correction band)")

        if boost_bins_cand > 0 and boost_bins_post == 0:
            if boost_bins_cand_low == boost_bins_cand and low_hz_cfg > 0:
                reasons.append(f"all boost candidates were <= low_bass_cut_hz ({low_hz_cfg:.1f} Hz) where cuts-only policy applies")
            if exc_on and exc_f_cfg > 0 and boost_bins_cand_exc == boost_bins_cand:
                reasons.append(f"all boost candidates were < exc_freq ({exc_f_cfg:.1f} Hz) while exc_prot is ON")
            if not reasons:
                reasons.append("boost candidates existed but were removed by limits/safety clamp (check max_boost_db, slope limits, exc/low-bass policies)")

        if boost_bins_cand > 0 and boost_bins_post > 0 and boost_bins_post < boost_bins_cand:
            if boost_bins_cand_low > 0:
                reasons.append(f"some boost candidates were in low-bass restricted region (<= {low_hz_cfg:.1f} Hz)")
            if exc_on and boost_bins_cand_exc > 0 and exc_f_cfg > 0:
                reasons.append(f"some boost candidates were in exc_prot region (< {exc_f_cfg:.1f} Hz)")
            reasons.append("some boost candidates were reduced by limits/safety clamp")

        if boost_bins_post > 0 and net_boost_peak <= 0.0:
            reasons.append(
                f"net boost peak <= 0.00 dB after global gain/headroom (net_peak={net_boost_peak:.2f} dB, normalize={'ON' if do_norm else 'OFF'})"
            )

        stats['boost_blocked_reason'] = "; ".join(dict.fromkeys(reasons)) if reasons else "no blocking detected"
    except Exception:
        stats['boost_blocked_reason'] = "diagnostic unavailable (exception)"


    if isinstance(cmp, dict) and cmp:
        stats.update(cmp)
        if stats.get('analysis_mode') != "comparison":
            stats['analysis_mode'] = "native"

    return impulse, stats

def generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg: FilterConfig):
    """
    Generoi vasemman ja oikean kanavan FIR-suodattimet.

    Jos `stereo_link` ei ole paalla, kanavat lasketaan itsenaisesti.
    Jos `stereo_link` on paalla, toteutus tekee kaksivaiheisen ajon:
    1) alustava ajo molemmille kanaville (ikkuna + offset-arviot)
    2) yhteisen offsetin ja mahdollisen auto-gain-overriden laskenta
    3) uusi ajo molemmille kanaville samoilla pakotetuilla arvoilla

    Palauttaa `(l_imp, l_stats, r_imp, r_stats)`.
    """
    if not bool(getattr(cfg, "stereo_link", False)):
        l_imp, l_st = generate_filter(f_l, m_l, p_l, cfg)
        r_imp, r_st = generate_filter(f_r, m_r, p_r, cfg)
        return l_imp, l_st, r_imp, r_st

    cfg1 = copy.deepcopy(cfg)
    try:
        cfg1.stereo_link = False
        cfg1.lvl_force_window = None
        cfg1.lvl_force_offset_db = None
    except Exception:
        pass

    l_imp1, l_st1 = generate_filter(f_l, m_l, p_l, cfg1)
    r_imp1, r_st1 = generate_filter(f_r, m_r, p_r, cfg1)

    mode = str(getattr(cfg1, "lvl_mode", "Auto") or "Auto")
    if "Manual" in mode:
        win = [float(getattr(cfg1, "lvl_min", 200.0) or 200.0), float(getattr(cfg1, "lvl_max", 3000.0) or 3000.0)]
    else:
        win = _safe_range((l_st1 or {}).get("smart_scan_range"), getattr(cfg1, "lvl_min", 200.0), getattr(cfg1, "lvl_max", 3000.0))
        if not (win[1] > win[0]):
            win = _safe_range((r_st1 or {}).get("smart_scan_range"), getattr(cfg1, "lvl_min", 200.0), getattr(cfg1, "lvl_max", 3000.0))

    off_l = float((l_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_r = float((r_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_shared = 0.5 * (off_l + off_r)

    shared_auto_gain_db = None
    try:
        margin_db = float(getattr(cfg1, "auto_gain_margin_db", getattr(cfg1, "global_gain_db", 0.0)) or 0.0)
    except Exception:
        margin_db = 0.0
    if (not np.isfinite(margin_db)) or (margin_db < 0.0):
        margin_db = 0.0
    try:
        peak_l = float((l_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
        peak_r = float((r_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
        peak_shared = max(0.0, peak_l, peak_r)
        shared_auto_gain_db = -(peak_shared + margin_db)
    except Exception:
        shared_auto_gain_db = None

    cfg2 = copy.deepcopy(cfg)
    try:
        cfg2.stereo_link = False
        cfg2.lvl_force_window = (float(win[0]), float(win[1]))
        cfg2.lvl_force_offset_db = float(off_shared)
        if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
            cfg2.auto_gain_db_override = float(shared_auto_gain_db)
    except Exception:
        pass

    l_imp2, l_st2 = generate_filter(f_l, m_l, p_l, cfg2)
    r_imp2, r_st2 = generate_filter(f_r, m_r, p_r, cfg2)

    try:
        if isinstance(l_st2, dict):
            l_st2["offset_method"] = str(l_st2.get("offset_method", "")) + " (StereoLinkShared)"
            l_st2["stereo_link_shared_offset_db"] = float(off_shared)
            l_st2["stereo_link_shared_window"] = [float(win[0]), float(win[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                l_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
        if isinstance(r_st2, dict):
            r_st2["offset_method"] = str(r_st2.get("offset_method", "")) + " (StereoLinkShared)"
            r_st2["stereo_link_shared_offset_db"] = float(off_shared)
            r_st2["stereo_link_shared_window"] = [float(win[0]), float(win[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                r_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
    except Exception:
        pass

    return l_imp2, l_st2, r_imp2, r_st2
