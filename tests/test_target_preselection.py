import numpy as np

from camillafir.common.house_curves import get_house_curve_by_name
from camillafir.io.auto_mode.target_preselection import _auto_select_builtin_target_curve


def _synth_target_like_measurement(hc_name: str):
    f = np.geomspace(20.0, 20000.0, 512).astype(float)
    hc_f, hc_m = get_house_curve_by_name(hc_name)
    hc_f = np.asarray(hc_f, dtype=float).reshape(-1)
    hc_m = np.asarray(hc_m, dtype=float).reshape(-1)
    mask = np.isfinite(hc_f) & np.isfinite(hc_m) & (hc_f > 0.0)
    base = np.interp(f, hc_f[mask], hc_m[mask])

    # Add smooth room-like coloration so the case is not a trivial exact-curve match.
    ripple = 0.12 * np.sin(np.log(f) * 2.3) + 0.06 * np.cos(np.log(f) * 5.2)
    lf_hump = 0.4 * np.exp(-0.5 * ((np.log10(f) - np.log10(55.0)) / 0.12) ** 2)
    hf_roll = -0.15 * np.clip(np.log10(f / 6000.0), 0.0, None)
    ml = base + ripple + lf_hump + hf_roll
    mr = base + (0.9 * ripple) + (0.85 * lf_hump) + hf_roll + 0.04 * np.sin(np.log(f) * 7.0)
    return f, ml, mr


def test_target_preselection_distinguishes_flat_from_cinema():
    f, ml, mr = _synth_target_like_measurement("Flat")

    picked = _auto_select_builtin_target_curve({}, f_l=f, m_l=ml, f_r=f, m_r=mr)

    assert picked is not None
    assert str(picked.get("selected_hc_mode")) == "Flat"


def test_target_preselection_distinguishes_studio_from_bk_medium_family():
    f, ml, mr = _synth_target_like_measurement("Studio")

    picked = _auto_select_builtin_target_curve({}, f_l=f, m_l=ml, f_r=f, m_r=mr)

    assert picked is not None
    assert str(picked.get("selected_hc_mode")) == "Studio"


def test_target_preselection_distinguishes_bk_strong_from_bk_medium():
    f, ml, mr = _synth_target_like_measurement("BK_Strong")

    picked = _auto_select_builtin_target_curve({}, f_l=f, m_l=ml, f_r=f, m_r=mr)

    assert picked is not None
    assert str(picked.get("selected_hc_mode")) == "BK_Strong"
