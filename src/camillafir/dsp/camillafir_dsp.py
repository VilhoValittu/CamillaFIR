import numpy as np
import scipy.ndimage
import copy
import logging
logger = logging.getLogger("CamillaFIR.dsp")
from camillafir.config.models import FilterConfig
from .dsp_correction import run_correction_stage
from .dsp_phase_ir import run_phase_ir_stage
from .dsp_preprocess import run_preprocess
from .camillafir_leveling import StereoLinkContext, find_shared_stereo_level_window
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

        # For target-pull, "cut direction" means target is BELOW measured
        # (i.e. target requests attenuation at that frequency).
        # Using (t < 0) is wrong here because target SPL is typically positive.
        is_cut = (t < m) #leikkaus

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
    m = m & np.isfinite(f) & np.isfinite(ph) & (f > 0.0) & (f >= float(f_min)) & (f <= float(f_max))
    if not np.any(m):
        return ph

    idx = np.where(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    ff = f[i0:i1 + 1]
    pp = ph[i0:i1 + 1]
    if ff.size < 16:
        return ph
    if not np.all(np.diff(ff) > 0.0):
        # Non-monotonic axis can explode derivative math.
        return ph

    pp_u = np.unwrap(pp)

    # GD in seconds: gd_s = -dphi/d(2*pi*f); convert to ms.
    omega = 2.0 * np.pi * ff
    gd_s = -np.gradient(pp_u, omega)
    gd_ms = np.nan_to_num(gd_s * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)

    log2f = np.log2(np.maximum(ff, 1e-12))
    lim = float(max(0.1, max_grad_ms_per_oct))

    gd_l = gd_ms.copy()
    try:
        sigma = float(grad_smooth_sigma) if float(grad_smooth_sigma) > 0.0 else 0.6
    except Exception:
        sigma = 0.6
    sigma = float(max(0.25, sigma))

    # Iteratively smooth GD until gradient is under limit (or close).
    for _ in range(14):
        gd_grad_now = np.gradient(gd_l, log2f)
        gd_grad_now = np.nan_to_num(gd_grad_now, nan=0.0, posinf=0.0, neginf=0.0)
        if gd_grad_now.size == 0:
            break
        if float(np.max(np.abs(gd_grad_now))) <= (lim * 1.001):
            break
        gd_l = scipy.ndimage.gaussian_filter1d(gd_l, sigma=sigma, mode="nearest")
        sigma *= 1.25

    gd_grad = np.gradient(gd_l, log2f)
    gd_grad = np.nan_to_num(gd_grad, nan=0.0, posinf=0.0, neginf=0.0)
    if soft_limit:
        gd_grad_l = lim * np.tanh(gd_grad / lim)
    else:
        gd_grad_l = np.clip(gd_grad, -lim, lim)

    # Rebuild GD from bounded gradient on log2-frequency axis.
    gd_new = np.empty_like(gd_l)
    gd_new[0] = float(gd_l[0])
    for k in range(1, ff.size):
        gd_new[k] = gd_new[k - 1] + gd_grad_l[k - 1] * (log2f[k] - log2f[k - 1])

    # Integrate back to phase: dphi/df = -2*pi*gd_s = -2*pi*(gd_ms/1000).
    dphi_df = -2.0 * np.pi * (gd_new / 1000.0)
    phi_new = np.empty_like(pp_u)
    phi_new[0] = float(pp_u[0])
    dff = np.diff(ff)
    for k in range(1, ff.size):
        phi_new[k] = phi_new[k - 1] + 0.5 * (dphi_df[k - 1] + dphi_df[k]) * dff[k - 1]

    out = ph.copy()
    out[i0:i1 + 1] = np.nan_to_num(phi_new, nan=0.0, posinf=0.0, neginf=0.0)
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


def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig, *, stereo_link_ctx: StereoLinkContext | None = None):
    prep = run_preprocess(freqs, meas_mags, raw_phases, cfg, stereo_link_ctx=stereo_link_ctx)
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
        stereo_link_ctx=stereo_link_ctx,
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
    boost_cand_min_hz = corr.get("boost_cand_min_hz", float("nan"))
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
    normalize_gain_db_applied = 0.0
    if cfg.do_normalize and max_peak > 0:
        _norm_scale = float(0.89 / max_peak)
        impulse *= _norm_scale
        try:
            normalize_gain_db_applied = float(20.0 * np.log10(max(_norm_scale, 1e-12)))
        except Exception:
            normalize_gain_db_applied = 0.0

    stats = {

        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        'mag_c_min': float(getattr(cfg, 'mag_c_min', 0.0) or 0.0),
        'mag_c_max': float(getattr(cfg, 'mag_c_max', 0.0) or 0.0),
        'target_mags': target_mags.tolist(),
        'measured_mags_raw': m_anal.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'predicted_filter_mags': gain_db.tolist(),
        'predicted_filter_mags_source': "mag_post_limits_pre_ir",
        'filter_mags': gain_db.tolist(),
        'filter_mags_source': "mag_post_limits_pre_ir",
        'mag_mask': np.asarray(mask_c, dtype=float).tolist(),
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
        'normalize_gain_db_applied': float(normalize_gain_db_applied),
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
        'boost_candidate_min_hz': float(locals().get('boost_cand_min_hz', float("nan"))),
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
    # Keep target on a shared absolute reference level for scoring/quality.
    # Preserve measured arrays from `st` when available to keep UI view
    # behavior stable; only fill missing measured fields.
    try:
        stats["target_mags"] = np.asarray(target_mags, dtype=float).tolist()
        f_ref = np.asarray(stats.get("freq_axis", freq_axis), dtype=float).reshape(-1)
        n_ref = int(f_ref.size)

        def _arr_if_valid(v):
            try:
                a = np.asarray(v, dtype=float).reshape(-1)
            except Exception:
                return None
            if a.size < 8:
                return None
            if n_ref >= 8 and a.size != n_ref:
                return None
            return np.asarray(a, dtype=float)

        m_corr_st = _arr_if_valid(stats.get("measured_mags", None))
        if m_corr_st is None:
            m_corr = np.asarray(m_anal, dtype=float) - float(calc_offset_db)
        else:
            m_corr = np.asarray(m_corr_st, dtype=float)

        m_raw_st = _arr_if_valid(stats.get("measured_mags_raw", None))
        if m_raw_st is None:
            m_raw = np.asarray(m_corr, dtype=float) + float(calc_offset_db)
        else:
            m_raw = np.asarray(m_raw_st, dtype=float)

        stats["measured_mags"] = np.asarray(m_corr, dtype=float).tolist()
        stats["measured_mags_raw"] = np.asarray(m_raw, dtype=float).tolist()
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
        low_hz_cfg = float(stats.get('low_bass_cut_hz', getattr(cfg, 'low_bass_cut_hz', 0.0)) or 0.0)
        exc_on_cfg = bool(stats.get('exc_prot', getattr(cfg, 'exc_prot', False)))
        exc_f_cfg = float(stats.get('exc_freq', getattr(cfg, 'exc_freq', 0.0)) or 0.0)
        lf_guard_hz = 0.0
        if np.isfinite(low_hz_cfg) and low_hz_cfg > 0.0:
            lf_guard_hz = max(lf_guard_hz, float(low_hz_cfg))
        if exc_on_cfg and np.isfinite(exc_f_cfg) and exc_f_cfg > 0.0:
            lf_guard_hz = max(lf_guard_hz, float(exc_f_cfg * 1.41))
        lf_mask = (freq_axis > 0.0) & (freq_axis <= lf_guard_hz) if lf_guard_hz > 0.0 else np.zeros_like(freq_axis, dtype=bool)
        if np.any(lf_mask):
            lf_boost_max_db = float(np.max(np.asarray(gain_db, dtype=float)[lf_mask]))
        else:
            lf_boost_max_db = 0.0
        stats['lf_guard_hz'] = float(lf_guard_hz)
        stats['lf_guard_bins'] = int(np.count_nonzero(lf_mask))
        stats['lf_boost_max_db'] = float(lf_boost_max_db)
    except Exception:
        stats['lf_guard_hz'] = 0.0
        stats['lf_guard_bins'] = 0
        stats['lf_boost_max_db'] = 0.0

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
    try:
        cmp_g_pred = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
        cmp_g_cur = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
        if cmp_g_pred.size < 8 and cmp_g_cur.size >= 8:
            stats["cmp_predicted_filter_mags"] = cmp_g_cur.tolist()
            stats["cmp_predicted_filter_mags_source"] = str(
                stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
            )
    except Exception:
        pass
    try:
        cmp_m_raw = np.asarray(stats.get("cmp_measured_mags_raw", []), dtype=float).reshape(-1)
        cmp_m_cur = np.asarray(stats.get("cmp_measured_mags", []), dtype=float).reshape(-1)
        if cmp_m_raw.size < 8 and cmp_m_cur.size >= 8:
            cmp_off = float(stats.get("cmp_offset_db", 0.0) or 0.0)
            stats["cmp_measured_mags_raw"] = (cmp_m_cur + cmp_off).tolist()
    except Exception:
        pass

    # Canonical filter magnitude for reporting: always derive from final IR.
    # This keeps DSP Quality Report aligned with the exported/used filter.
    try:
        ir = np.asarray(impulse, dtype=float).flatten()
        fs_i = int(getattr(cfg, "fs", 0) or 0)
        f_native = np.asarray(stats.get("freq_axis", freq_axis), dtype=float).reshape(-1)
        if ir.size >= 8 and fs_i > 0 and f_native.size >= 4:
            h = np.fft.rfft(ir)
            f_fft = np.fft.rfftfreq(ir.size, d=1.0 / float(fs_i))
            g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
            f_q = np.clip(f_native, float(np.min(f_fft)), float(np.max(f_fft)))
            g_pred_native = np.asarray(stats.get("predicted_filter_mags", []), dtype=float).reshape(-1)
            if g_pred_native.size < 8:
                g_cur_native = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                if g_cur_native.size >= 8:
                    stats["predicted_filter_mags"] = g_cur_native.tolist()
                    stats["predicted_filter_mags_source"] = str(
                        stats.get("filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                    )

            g_real_native = np.interp(f_q, f_fft, g_db)
            stats["realized_filter_mags"] = g_real_native.tolist()
            stats["realized_filter_mags_source"] = "ir_fft_final"
            stats["filter_mags"] = g_real_native.tolist()
            stats["filter_mags_source"] = "ir_fft_final"

            f_cmp = np.asarray(stats.get("cmp_freq_axis", []), dtype=float).reshape(-1)
            if f_cmp.size >= 4:
                f_cmp_q = np.clip(f_cmp, float(np.min(f_fft)), float(np.max(f_fft)))
                g_pred_cmp = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
                if g_pred_cmp.size < 4:
                    g_cur_cmp = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
                    if g_cur_cmp.size >= 4:
                        stats["cmp_predicted_filter_mags"] = g_cur_cmp.tolist()
                        stats["cmp_predicted_filter_mags_source"] = str(
                            stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                        )
                g_real_cmp = np.interp(f_cmp_q, f_fft, g_db)
                stats["cmp_realized_filter_mags"] = g_real_cmp.tolist()
                stats["cmp_realized_filter_mags_source"] = "ir_fft_final"
                stats["cmp_filter_mags"] = g_real_cmp.tolist()
                stats["cmp_filter_mags_source"] = "ir_fft_final"

            # Realization delta diagnostics:
            # compare post-limits mag gain_db against final IR-derived filter_mags.
            try:
                g_post = np.asarray(gain_db, dtype=float).reshape(-1)
                g_ir = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                n = int(min(f_q.size, g_post.size, g_ir.size))
                if n >= 8:
                    f_eval = np.asarray(f_q[:n], dtype=float)
                    d_eval = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_post[:n], dtype=float)
                    valid = np.isfinite(f_eval) & np.isfinite(d_eval) & (f_eval > 0.0)

                    def _band_delta_on(d_arr: np.ndarray, lo_hz: float, hi_hz: float):
                        m = valid & (f_eval >= float(lo_hz)) & (f_eval <= float(hi_hz))
                        if int(np.count_nonzero(m)) < 8:
                            return None, None, None
                        dv = np.asarray(d_arr[m], dtype=float)
                        fv = f_eval[m]
                        idx = int(np.argmax(np.abs(dv)))
                        return float(np.sqrt(np.mean(dv * dv))), float(np.abs(dv[idx])), float(fv[idx])

                    rms_b, max_b, hz_b = _band_delta_on(d_eval, 20.0, 200.0)
                    stats["post_to_ir_delta_rms_20_200_db"] = rms_b
                    stats["post_to_ir_delta_max_20_200_db"] = max_b
                    stats["post_to_ir_delta_max_hz_20_200"] = hz_b

                    m20 = valid & (f_eval >= 20.0) & (f_eval <= 200.0)
                    if int(np.count_nonzero(m20)) >= 8:
                        off20 = float(np.median(np.asarray(d_eval[m20], dtype=float)))
                        d_shape_20 = np.asarray(d_eval, dtype=float) - float(off20)
                        srms20, smax20, shz20 = _band_delta_on(d_shape_20, 20.0, 200.0)
                        stats["post_to_ir_delta_offset_20_200_db"] = float(off20)
                        stats["post_to_ir_shape_delta_rms_20_200_db"] = srms20
                        stats["post_to_ir_shape_delta_max_20_200_db"] = smax20
                        stats["post_to_ir_shape_delta_max_hz_20_200"] = shz20

                        # Same diagnostic, but baseline includes gain staging
                        # (auto gain/headroom + possible final normalize scale).
                        g_stage = (
                            np.asarray(g_post, dtype=float)
                            + float(auto_global_gain_db)
                            + float(auto_headroom_db)
                            + float(normalize_gain_db_applied)
                        )
                        d_stage = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_stage[:n], dtype=float)
                        srms_abs, smax_abs, shz_abs = _band_delta_on(d_stage, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_rms_20_200_db"] = srms_abs
                        stats["post_to_ir_staged_delta_max_20_200_db"] = smax_abs
                        stats["post_to_ir_staged_delta_max_hz_20_200"] = shz_abs

                        off_stage = float(np.median(np.asarray(d_stage[m20], dtype=float)))
                        d_stage_shape = np.asarray(d_stage, dtype=float) - float(off_stage)
                        srms_shape, smax_shape, shz_shape = _band_delta_on(d_stage_shape, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_offset_20_200_db"] = float(off_stage)
                        stats["post_to_ir_staged_shape_delta_rms_20_200_db"] = srms_shape
                        stats["post_to_ir_staged_shape_delta_max_20_200_db"] = smax_shape
                        stats["post_to_ir_staged_shape_delta_max_hz_20_200"] = shz_shape

                    cmin = float(stats.get("mag_c_min", getattr(cfg, "mag_c_min", 20.0)) or 20.0)
                    cmax = float(stats.get("mag_c_max", getattr(cfg, "mag_c_max", 20000.0)) or 20000.0)
                    if (not np.isfinite(cmin)) or (cmin < 0.0):
                        cmin = 20.0
                    if (not np.isfinite(cmax)) or (cmax <= cmin):
                        cmax = max(200.0, cmin + 1.0)
                    rms_c, max_c, hz_c = _band_delta_on(d_eval, cmin, cmax)
                    stats["post_to_ir_delta_rms_magc_db"] = rms_c
                    stats["post_to_ir_delta_max_magc_db"] = max_c
                    stats["post_to_ir_delta_max_hz_magc"] = hz_c
                    m_c = valid & (f_eval >= float(cmin)) & (f_eval <= float(cmax))
                    if int(np.count_nonzero(m_c)) >= 8:
                        off_c = float(np.median(np.asarray(d_eval[m_c], dtype=float)))
                        d_shape_c = np.asarray(d_eval, dtype=float) - float(off_c)
                        srms_c, smax_c, shz_c = _band_delta_on(d_shape_c, cmin, cmax)
                        stats["post_to_ir_delta_offset_magc_db"] = float(off_c)
                        stats["post_to_ir_shape_delta_rms_magc_db"] = srms_c
                        stats["post_to_ir_shape_delta_max_magc_db"] = smax_c
                        stats["post_to_ir_shape_delta_max_hz_magc"] = shz_c
            except Exception:
                pass
    except Exception:
        pass

    return impulse, stats

def generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg: FilterConfig):
    """
    Generoi vasemman ja oikean kanavan FIR-suodattimet.

    Jos `stereo_link` ei ole paalla, kanavat lasketaan itsenaisesti.
    Jos `stereo_link` on paalla, toteutus tekee kaksivaiheisen ajon:
    1) alustava ajo molemmille kanaville (ikkuna + offset-arviot)
    2) yhteisen offsetin ja mahdollisen auto-gain-overriden laskenta
    3) uusi ajo strategian mukaan:
       - shared: sama ikkuna + sama offset molemmille
       - hybrid: kanavakohtainen ikkuna + sama offset molemmille
       - auto: guard-valinta shared/hybrid

    Palauttaa `(l_imp, l_stats, r_imp, r_stats)`.
    """
    if not bool(getattr(cfg, "stereo_link", False)):
        l_imp, l_st = generate_filter(f_l, m_l, p_l, cfg)
        r_imp, r_st = generate_filter(f_r, m_r, p_r, cfg)
        return l_imp, l_st, r_imp, r_st

    l_imp1, l_st1 = generate_filter(f_l, m_l, p_l, cfg)
    r_imp1, r_st1 = generate_filter(f_r, m_r, p_r, cfg)

    def _as_stat_float(st: dict | None, key: str, default=np.nan) -> float:
        try:
            if isinstance(st, dict):
                v = float(st.get(key, default))
                return v if np.isfinite(v) else float(default)
        except Exception:
            pass
        return float(default)

    def _shared_window_from_stats(st_l: dict | None, st_r: dict | None):
        try:
            if not isinstance(st_l, dict) or not isinstance(st_r, dict):
                return None
            freq_l = np.asarray(st_l.get("freq_axis", []) or [], dtype=float)
            meas_l = np.asarray(st_l.get("measured_mags", []) or [], dtype=float)
            targ_l = np.asarray(st_l.get("target_mags", []) or [], dtype=float)
            freq_r = np.asarray(st_r.get("freq_axis", []) or [], dtype=float)
            meas_r = np.asarray(st_r.get("measured_mags", []) or [], dtype=float)
            targ_r = np.asarray(st_r.get("target_mags", []) or [], dtype=float)
            if (
                freq_l.size < 50
                or freq_r.size < 50
                or meas_l.size != freq_l.size
                or targ_l.size != freq_l.size
                or meas_r.size != freq_r.size
                or targ_r.size != freq_r.size
            ):
                return None
            try:
                hpf_settings = getattr(cfg, "hpf_settings", None)
                hpf_freq = float(hpf_settings.get("freq", 0.0)) if hpf_settings else 0.0
            except Exception:
                hpf_freq = 0.0
            win = find_shared_stereo_level_window(
                freq_l,
                meas_l,
                targ_l,
                freq_r,
                meas_r,
                targ_r,
                float(lvl_min),
                float(lvl_max),
                window_size_octaves=1.0,
                hpf_freq=float(hpf_freq),
                tilt_comp=bool(getattr(cfg, "lvl_tilt_comp", True)),
                tilt_max_db_per_oct=float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0) or 2.0),
                perceptual_weighting=bool(getattr(cfg, "lvl_perceptual_weighting", False)),
                perceptual_strength=float(getattr(cfg, "lvl_perceptual_strength", 0.12) or 0.12),
                perceptual_min_hz=float(getattr(cfg, "lvl_perceptual_min_hz", 250.0) or 250.0),
                perceptual_max_hz=float(getattr(cfg, "lvl_perceptual_max_hz", 4000.0) or 4000.0),
                perceptual_tie_only=bool(getattr(cfg, "lvl_perceptual_tie_only", True)),
            )
            return _safe_range(win, lvl_min, lvl_max)
        except Exception:
            return None

    def _pick_quieter_anchor():
        left = {
            "channel": "left",
            "offset_db": _as_stat_float(l_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(l_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(l_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(l_st1, "meas_level_db_window", np.nan),
        }
        right = {
            "channel": "right",
            "offset_db": _as_stat_float(r_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(r_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(r_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(r_st1, "meas_level_db_window", np.nan),
        }

        def _anchor_key(candidate):
            shift = candidate["target_shift_db"]
            meas = candidate["meas_level_db_window"]
            target = candidate["target_level_db"]
            if np.isfinite(shift):
                return (0, float(shift))
            if np.isfinite(meas):
                return (1, float(meas))
            if np.isfinite(target):
                return (2, float(target))
            return (3, float("inf"))

        candidates = [c for c in (left, right) if _anchor_key(c)[0] < 3]
        if not candidates:
            return None
        return min(candidates, key=_anchor_key)

    lvl_min = float(getattr(cfg, "lvl_min", 200.0) or 200.0)
    lvl_max = float(getattr(cfg, "lvl_max", 3000.0) or 3000.0)

    mode = str(getattr(cfg, "lvl_mode", "Auto") or "Auto")
    if "Manual" in mode:
        win_l = [lvl_min, lvl_max]
        win_r = [lvl_min, lvl_max]
    else:
        win_l = _safe_range((l_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)
        win_r = _safe_range((r_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)

    shared_win_from_scan = _shared_window_from_stats(l_st1, r_st1) if "Manual" not in mode else None
    win_shared = list(shared_win_from_scan) if shared_win_from_scan is not None else list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_r)
    if not (win_shared[1] > win_shared[0]):
        win_shared = [lvl_min, lvl_max]

    anchor = _pick_quieter_anchor()
    off_l = float((l_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_r = float((r_st1 or {}).get("offset_db", 0.0) or 0.0)
    if anchor is not None and np.isfinite(float(anchor["offset_db"])):
        off_shared = float(anchor["offset_db"])
    else:
        off_shared = min(float(off_l), float(off_r))
    tgt_l = _as_stat_float(l_st1, "eff_target_db", np.nan)
    tgt_r = _as_stat_float(r_st1, "eff_target_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_level_db"])):
        target_shared = float(anchor["target_level_db"])
    elif np.isfinite(tgt_l) and np.isfinite(tgt_r):
        target_shared = min(float(tgt_l), float(tgt_r))
    elif np.isfinite(tgt_l):
        target_shared = float(tgt_l)
    elif np.isfinite(tgt_r):
        target_shared = float(tgt_r)
    else:
        target_shared = None
    tshift_l = _as_stat_float(l_st1, "target_shift_db", np.nan)
    tshift_r = _as_stat_float(r_st1, "target_shift_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_shift_db"])):
        target_shift_shared = float(anchor["target_shift_db"])
    elif np.isfinite(tshift_l) and np.isfinite(tshift_r):
        target_shift_shared = min(float(tshift_l), float(tshift_r))
    elif np.isfinite(tshift_l):
        target_shift_shared = float(tshift_l)
    elif np.isfinite(tshift_r):
        target_shift_shared = float(tshift_r)
    else:
        target_shift_shared = None

    try:
        strategy_req = str(getattr(cfg, "stereo_link_strategy", "shared") or "shared").strip().lower()
    except Exception:
        strategy_req = "shared"
    if strategy_req not in ("shared", "hybrid", "auto"):
        strategy_req = "shared"

    tilt_l = _as_stat_float(l_st1, "tilt_slope_db_per_oct", np.nan)
    tilt_r = _as_stat_float(r_st1, "tilt_slope_db_per_oct", np.nan)
    off_diff = abs(float(off_l) - float(off_r))
    tilt_diff = abs(float(tilt_l) - float(tilt_r)) if (np.isfinite(tilt_l) and np.isfinite(tilt_r)) else 0.0
    tilt_abs_max = max(abs(float(tilt_l)) if np.isfinite(tilt_l) else 0.0, abs(float(tilt_r)) if np.isfinite(tilt_r) else 0.0)

    guard_triggered = bool(
        (off_diff > 1.5)
        or (tilt_diff > 0.7)
        or (tilt_abs_max > 1.2)
    )
    strategy_resolved = "hybrid" if (strategy_req == "auto" and guard_triggered) else ("shared" if strategy_req == "auto" else strategy_req)

    shared_auto_gain_db = None
    try:
        margin_db = float(getattr(cfg, "auto_gain_margin_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0)
    except Exception:
        margin_db = 0.0
    if (not np.isfinite(margin_db)) or (margin_db < 0.0):
        margin_db = 0.0
    try:
        ag_l = float((l_st1 or {}).get("auto_global_gain_db", np.nan))
        ag_r = float((r_st1 or {}).get("auto_global_gain_db", np.nan))
        if np.isfinite(ag_l) and np.isfinite(ag_r):
            # More negative dB value is the safer shared attenuation.
            shared_auto_gain_db = min(ag_l, ag_r)
        else:
            peak_l = float((l_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
            peak_r = float((r_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
            peak_shared = max(0.0, peak_l, peak_r)
            shared_auto_gain_db = -(peak_shared + margin_db)
    except Exception:
        shared_auto_gain_db = None

    cfg2 = copy.deepcopy(cfg)
    try:
        cfg2.stereo_link = False
        if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
            cfg2.auto_gain_db_override = float(shared_auto_gain_db)
    except Exception:
        pass

    if strategy_resolved == "hybrid":
        stereo_ctx_l = StereoLinkContext(
            forced_window_hz=(float(win_l[0]), float(win_l[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_r = StereoLinkContext(
            forced_window_hz=(float(win_r[0]), float(win_r[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
    else:
        stereo_ctx = StereoLinkContext(
            forced_window_hz=(float(win_shared[0]), float(win_shared[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_l = stereo_ctx
        stereo_ctx_r = stereo_ctx

    l_imp2, l_st2 = generate_filter(f_l, m_l, p_l, cfg2, stereo_link_ctx=stereo_ctx_l)
    r_imp2, r_st2 = generate_filter(f_r, m_r, p_r, cfg2, stereo_link_ctx=stereo_ctx_r)

    try:
        if isinstance(l_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            l_st2["offset_method"] = str(l_st2.get("offset_method", "")) + f" ({mode_tag})"
            l_st2["stereo_link_mode"] = str(strategy_resolved)
            l_st2["stereo_link_requested_mode"] = str(strategy_req)
            l_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            l_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            l_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            l_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            l_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                l_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                l_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                l_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            l_st2["stereo_link_window_used"] = [float(win_l[0]), float(win_l[1])] if strategy_resolved == "hybrid" else [float(win_shared[0]), float(win_shared[1])]
            if strategy_resolved != "hybrid":
                l_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                l_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
        if isinstance(r_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            r_st2["offset_method"] = str(r_st2.get("offset_method", "")) + f" ({mode_tag})"
            r_st2["stereo_link_mode"] = str(strategy_resolved)
            r_st2["stereo_link_requested_mode"] = str(strategy_req)
            r_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            r_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            r_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            r_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            r_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                r_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                r_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                r_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            r_st2["stereo_link_window_used"] = [float(win_r[0]), float(win_r[1])] if strategy_resolved == "hybrid" else [float(win_shared[0]), float(win_shared[1])]
            if strategy_resolved != "hybrid":
                r_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                r_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
    except Exception:
        pass

    return l_imp2, l_st2, r_imp2, r_st2
