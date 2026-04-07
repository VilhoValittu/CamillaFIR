from camillafir.resources.i8n.camillafir_i18n import TRANSLATIONS
from camillafir.ui_i18n import (
    AFDW_PRESET_TIGHT,
    LAYOUT_STEREO,
    LVL_ALGO_AVERAGE,
    LVL_MODE_MANUAL,
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT,
    TDC_PRESET_SAFE,
    normalize_afdw_preset_key,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
    normalize_tdc_preset_key,
    tr_options,
)


def test_tr_options_preserves_stable_keys():
    options = tr_options(lambda key: f"tr:{key}", {"mono": "layout_mono", "stereo": "layout_stereo"})

    assert options == {
        "mono": "tr:layout_mono",
        "stereo": "tr:layout_stereo",
    }


def test_normalize_layout_value_accepts_legacy_and_translated_labels():
    assert normalize_layout_value("stereo") == LAYOUT_STEREO
    assert normalize_layout_value("Stereo") == LAYOUT_STEREO
    assert normalize_layout_value(TRANSLATIONS["fi"]["layout_stereo"]) == LAYOUT_STEREO


def test_normalize_level_mode_value_accepts_legacy_and_translated_labels():
    assert normalize_lvl_mode_value("manual") == LVL_MODE_MANUAL
    assert normalize_lvl_mode_value("Manual") == LVL_MODE_MANUAL
    assert normalize_lvl_mode_value(TRANSLATIONS["fi"]["lvl_mode_manual"]) == LVL_MODE_MANUAL


def test_normalize_level_algo_value_accepts_legacy_and_translated_labels():
    assert normalize_lvl_algo_value("average") == LVL_ALGO_AVERAGE
    assert normalize_lvl_algo_value("Average") == LVL_ALGO_AVERAGE
    assert normalize_lvl_algo_value(TRANSLATIONS["fi"]["lvl_algo_average"]) == LVL_ALGO_AVERAGE


def test_normalize_output_tilt_source_value_accepts_legacy_and_translated_labels():
    assert normalize_output_tilt_source_value("manual_target_tilt") == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    assert normalize_output_tilt_source_value("Use Manual Tilt value") == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    assert (
        normalize_output_tilt_source_value(TRANSLATIONS["fi"]["output_tilt_use_manual_target_tilt"])
        == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    )


def test_normalize_preset_keys_accept_legacy_and_translated_labels():
    assert normalize_tdc_preset_key("Safe") == TDC_PRESET_SAFE
    assert normalize_tdc_preset_key(TRANSLATIONS["fi"]["tdc_preset_safe"]) == TDC_PRESET_SAFE
    assert normalize_afdw_preset_key("Tight") == AFDW_PRESET_TIGHT
    assert normalize_afdw_preset_key(TRANSLATIONS["fi"]["afdw_preset_tight"]) == AFDW_PRESET_TIGHT
