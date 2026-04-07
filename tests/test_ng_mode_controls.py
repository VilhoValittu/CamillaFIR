import pytest

from camillafir.ui import ng_controls as ctrl
from camillafir.ui.ng_mode_controls import (
    _build_taps_auto_info_markdown,
    _update_auto_mode_fields_state,
    update_afdw_cycles_ui,
    update_mixed_freq_ui,
    update_xo_ui,
    update_lvl_ui,
    update_target_curve_controls_ui,
    update_tdc_controls_ui,
)


_STRINGS = {
    "auto_taps_title": "Auto-taps (Multi-rate)",
    "auto_taps_body": "Auto taps body",
}


def _t(key: str) -> str:
    return _STRINGS[key]


class _DummyControl:
    def __init__(self, value):
        self.value = value
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class _DummyContainer:
    def __init__(self):
        self.visible = None

    def set_visibility(self, visible):
        self.visible = bool(visible)


def test_build_taps_auto_info_markdown_off_state():
    text = _build_taps_auto_info_markdown(
        multi_rate=False,
        base_taps=65536,
        t=_t,
    )

    assert text == "_Auto-taps (Multi-rate): OFF_"


def test_build_taps_auto_info_markdown_uses_current_base_taps():
    text = _build_taps_auto_info_markdown(
        multi_rate=True,
        base_taps=131072,
        t=_t,
    )

    assert text.startswith("### Auto-taps (Multi-rate)\nAuto taps body")
    assert "**Reference:** 44.1 kHz -> 131,072 taps" in text
    assert "- **44.1 kHz** -> **131,072** taps" in text
    assert "- **88.2 kHz** -> **262,144** taps" in text
    assert text.count("kHz** -> **") == 6


def test_update_lvl_ui_shows_manual_target_scope_for_manual_mode():
    ctrl.reset()
    ctrl.register("mode", _DummyControl("ADVANCED"))
    ctrl.register("lvl_mode", _DummyControl("Manual"))
    scope = _DummyContainer()
    output_scope = _DummyContainer()
    ctrl.register_container("lvl_manual_scope", scope)
    ctrl.register_container("output_tilt_scope", output_scope)

    update_lvl_ui(t=_t)

    assert scope.visible is True
    assert output_scope.visible is True


def test_update_lvl_ui_hides_output_tilt_scope_outside_advanced_manual_mode():
    ctrl.reset()
    ctrl.register("mode", _DummyControl("BASIC"))
    ctrl.register("lvl_mode", _DummyControl("Manual"))
    scope = _DummyContainer()
    output_scope = _DummyContainer()
    ctrl.register_container("lvl_manual_scope", scope)
    ctrl.register_container("output_tilt_scope", output_scope)

    update_lvl_ui(t=_t)

    assert scope.visible is True
    assert output_scope.visible is False


def test_auto_mode_keeps_filter_type_enabled():
    ctrl.reset()
    ctrl.register("filter_type", _DummyControl("Asymmetric (low-latency)"))
    ctrl.register("mag_correct", _DummyControl(True))
    ctrl.register("auto_goal", _DummyControl("balanced"))
    ctrl.register("auto_target_mode", _DummyControl("auto"))
    ctrl.register("hc_mode", _DummyControl("Harman6"))
    ctrl.register("hc_custom_file", _DummyControl(None))

    _update_auto_mode_fields_state(is_auto=True, t=_t)

    assert ctrl.get("filter_type").enabled is True
    assert ctrl.get("mag_correct").enabled is False
    assert ctrl.get("auto_goal").enabled is True


def test_auto_mode_scope_visibility_tracks_current_mode():
    ctrl.reset()
    ctrl.register("filter_type", _DummyControl("Asymmetric"))
    ctrl.register("mag_correct", _DummyControl(True))
    ctrl.register("auto_goal", _DummyControl("balanced"))
    ctrl.register("auto_target_mode", _DummyControl("auto"))
    ctrl.register("hc_mode", _DummyControl("Harman6"))
    ctrl.register("hc_custom_file", _DummyControl(None))
    auto_scope = _DummyContainer()
    ctrl.register_container("auto_mode_scope", auto_scope)

    _update_auto_mode_fields_state(is_auto=False, t=_t)
    assert auto_scope.visible is False

    _update_auto_mode_fields_state(is_auto=True, t=_t)
    assert auto_scope.visible is True


def test_target_curve_selection_enabled_in_auto_mode_when_selected_target_mode_is_used():
    ctrl.reset()
    ctrl.register("mode", _DummyControl("AUTO"))
    ctrl.register("auto_target_mode", _DummyControl("selected"))
    ctrl.register("hc_mode", _DummyControl("Harman6"))
    ctrl.register("hc_custom_file", _DummyControl(None))

    update_target_curve_controls_ui()

    assert ctrl.get("hc_mode").enabled is True
    assert ctrl.get("hc_custom_file").enabled is True


@pytest.mark.parametrize("auto_target_mode", ["auto", "adaptive"])
def test_target_curve_selection_locked_in_auto_mode_when_target_mode_is_not_selected(auto_target_mode):
    ctrl.reset()
    ctrl.register("mode", _DummyControl("AUTO"))
    ctrl.register("auto_target_mode", _DummyControl(auto_target_mode))
    ctrl.register("hc_mode", _DummyControl("Harman6"))
    ctrl.register("hc_custom_file", _DummyControl(None))

    update_target_curve_controls_ui()

    assert ctrl.get("hc_mode").enabled is False
    assert ctrl.get("hc_custom_file").enabled is False


def test_update_tdc_controls_ui_toggles_only_details_scope():
    ctrl.reset()
    ctrl.register("enable_tdc", _DummyControl(False))
    section_scope = _DummyContainer()
    details_scope = _DummyContainer()
    ctrl.register_container("tdc_section_scope", section_scope)
    ctrl.register_container("tdc_details_scope", details_scope)

    update_tdc_controls_ui(t=_t)

    assert details_scope.visible is False
    assert section_scope.visible is None


def test_update_afdw_cycles_ui_toggles_only_details_scope():
    ctrl.reset()
    ctrl.register("enable_afdw", _DummyControl(False))
    section_scope = _DummyContainer()
    details_scope = _DummyContainer()
    ctrl.register_container("afdw_section_scope", section_scope)
    ctrl.register_container("afdw_details_scope", details_scope)

    update_afdw_cycles_ui(t=_t)

    assert details_scope.visible is False
    assert section_scope.visible is None


def test_mixed_split_hidden_in_auto_mode_even_with_mixed_filter():
    ctrl.reset()
    ctrl.register("mode", _DummyControl("AUTO"))
    ctrl.register("filter_type", _DummyControl("Mixed Phase"))
    ctrl.register("mixed_freq", _DummyControl(200.0))
    scope = _DummyContainer()
    ctrl.register_container("update_mixed_freq_scope", scope)

    update_mixed_freq_ui(t=_t)

    assert scope.visible is False
    assert ctrl.get("mixed_freq").enabled is False


@pytest.mark.parametrize("mode", ["BASIC", "ADVANCED"])
def test_mixed_split_visible_in_basic_or_advanced_mode_with_mixed_filter(mode):
    ctrl.reset()
    ctrl.register("mode", _DummyControl(mode))
    ctrl.register("filter_type", _DummyControl("Mixed Phase"))
    ctrl.register("mixed_freq", _DummyControl(200.0))
    scope = _DummyContainer()
    ctrl.register_container("update_mixed_freq_scope", scope)

    update_mixed_freq_ui(t=_t)

    assert scope.visible is True
    assert ctrl.get("mixed_freq").enabled is True


@pytest.mark.parametrize("filter_type", ["Linear", "Asymmetric"])
def test_update_xo_ui_enables_xo_tab_for_supported_filter_types(filter_type):
    ctrl.reset()
    ctrl.register("filter_type", _DummyControl(filter_type))
    ctrl.register("tab_xo", _DummyControl(None))
    ctrl.register("xo1_f", _DummyControl(500.0))
    ctrl.register("xo1_s", _DummyControl(12))
    scope = _DummyContainer()
    ctrl.register_container("xo_tab_content_scope", scope)

    update_xo_ui()

    assert ctrl.get("tab_xo").enabled is True
    assert ctrl.get("xo1_f").enabled is True
    assert ctrl.get("xo1_s").enabled is True
    assert scope.visible is True


@pytest.mark.parametrize("filter_type", ["Mixed", "Minimum"])
def test_update_xo_ui_disables_xo_tab_for_unsupported_filter_types(filter_type):
    ctrl.reset()
    ctrl.register("filter_type", _DummyControl(filter_type))
    ctrl.register("tab_xo", _DummyControl(None))
    ctrl.register("xo1_f", _DummyControl(500.0))
    ctrl.register("xo1_s", _DummyControl(12))
    scope = _DummyContainer()
    ctrl.register_container("xo_tab_content_scope", scope)

    update_xo_ui()

    assert ctrl.get("tab_xo").enabled is False
    assert ctrl.get("xo1_f").enabled is False
    assert ctrl.get("xo1_s").enabled is False
    assert scope.visible is False
