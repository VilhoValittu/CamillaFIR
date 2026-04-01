from camillafir.ui import ng_controls as ctrl
from camillafir.ui.ng_advanced_presets import (
    AGGRESSIVE,
    NORMAL,
    SAFE,
    apply_bass_safety_preset,
    apply_conf_pull_preset,
    apply_shaping_preset,
    build_bass_safety_summary,
    build_conf_pull_summary,
    build_shaping_summary,
)


_STRINGS = {
    "adv_summary_global_rail": "Global rail",
    "adv_summary_boost_rail": "Boost rail",
    "adv_summary_cut_rail": "Cut rail",
    "adv_summary_max_cut": "Max cut",
    "adv_summary_transition": "Transition",
    "adv_summary_smoothing": "Smoothing",
    "adv_summary_phase_limit": "Phase limit",
    "adv_summary_exc_prot": "Excursion protection",
    "adv_summary_low_bass_cut": "Low-bass cut",
    "adv_summary_hpf": "HPF",
    "adv_summary_bass_first": "Bass First",
    "adv_summary_floor": "Floor",
    "adv_summary_ceil": "Ceil",
    "adv_summary_span": "Active span",
    "state_on": "ON",
    "state_off": "OFF",
    "df_smoothing_label": "Frequency correction consistency",
    "reg_strength": "Regularization",
    "low_bass_cut_strength_label": "strength",
    "smoothing_level": "FIR resolution",
    "filter_smooth_12": "1/12 Octave (Standard)",
    "filter_smooth_24": "1/24 Octave (Fine)",
}


def _t(key: str) -> str:
    return _STRINGS.get(key, key)


class _DummyControl:
    def __init__(self, value):
        self.value = value


def _register_defaults() -> None:
    ctrl.reset()
    for name, value in {
        "max_slope_db_per_oct": 12.0,
        "max_cut_db": 30.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "trans_width": 100,
        "filter_smooth": 12,
        "phase_limit": 400.0,
        "reg_strength": 30.0,
        "df_smoothing": False,
        "exc_prot": True,
        "exc_freq": 22.0,
        "low_bass_cut_enable": True,
        "low_bass_cut_hz": 25.0,
        "low_bass_cut_strength": 0.70,
        "hpf_enable": False,
        "hpf_freq": 20.0,
        "hpf_slope": 24,
        "bass_first_ai": False,
        "bass_first_mode_max_hz": 200.0,
        "conf_pull_floor": 0.10,
        "conf_pull_ceil": 0.90,
    }.items():
        ctrl.register(name, _DummyControl(value))


def test_apply_shaping_preset_updates_existing_controls():
    _register_defaults()

    apply_shaping_preset(AGGRESSIVE)

    assert ctrl.value("max_slope_db_per_oct") == 18.0
    assert ctrl.value("max_cut_db") == 36.0
    assert ctrl.value("max_slope_boost_db_per_oct") == 6.0
    assert ctrl.value("filter_smooth") == 24
    assert ctrl.value("df_smoothing") is True


def test_apply_bass_safety_preset_updates_hidden_and_visible_fields():
    _register_defaults()

    apply_bass_safety_preset(SAFE)

    assert ctrl.value("exc_prot") is True
    assert ctrl.value("low_bass_cut_strength") == 1.0
    assert ctrl.value("hpf_enable") is True
    assert ctrl.value("bass_first_ai") is False


def test_apply_conf_pull_preset_uses_requested_range():
    _register_defaults()

    apply_conf_pull_preset(NORMAL)
    assert ctrl.value("conf_pull_floor") == 0.10
    assert ctrl.value("conf_pull_ceil") == 0.90

    apply_conf_pull_preset(SAFE)
    assert ctrl.value("conf_pull_floor") == 0.20
    assert ctrl.value("conf_pull_ceil") == 0.75


def test_build_shaping_summary_reports_hidden_effective_state():
    _register_defaults()
    apply_shaping_preset(AGGRESSIVE)

    summary = build_shaping_summary(t=_t)

    assert "Global rail: 18 dB/oct" in summary
    assert "Boost rail: 6 dB/oct" in summary
    assert "Transition: 60 Hz" in summary
    assert "Smoothing: 1/24 Octave (Fine)" in summary
    assert "Frequency correction consistency: ON" in summary


def test_build_bass_safety_summary_spells_out_off_states():
    _register_defaults()
    apply_bass_safety_preset(AGGRESSIVE)

    summary = build_bass_safety_summary(t=_t)

    assert "Excursion protection: OFF" in summary
    assert "Low-bass cut: OFF" in summary
    assert "HPF: OFF" in summary
    assert "Bass First: ON <= 200 Hz" in summary


def test_build_conf_pull_summary_reports_active_span():
    _register_defaults()
    apply_conf_pull_preset(NORMAL)

    summary = build_conf_pull_summary(t=_t)

    assert summary == "Floor: 0.1 | Ceil: 0.9 | Active span: 0.8"
