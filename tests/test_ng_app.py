from pathlib import Path

from camillafir.ui import ng_app
from camillafir.resources.i8n.camillafir_i18n import TRANSLATIONS
from camillafir.ui.ng_app import _external_link_head_html, _load_user_manual_text


def test_external_link_head_html_opens_http_links_in_new_window():
    html = _external_link_head_html()

    assert "e.preventDefault();e.stopPropagation();" in html
    assert "window.open(a.href,'_blank','noopener,noreferrer');" in html
    assert "/^https?:\\/\\//i.test(a.getAttribute('href'))" in html


def test_load_user_manual_text_reads_manual_markdown():
    manual_text = _load_user_manual_text()

    assert manual_text.startswith("# CamillaFIR User Manual")
    assert "## 1. What is CamillaFIR?" in manual_text


def test_manual_button_translation_keys_exist_for_en_and_fi():
    for lang in ("en", "fi"):
        assert "open_manual_btn" in TRANSLATIONS[lang]
        assert "manual_close_btn" in TRANSLATIONS[lang]


def test_target_preview_hint_translation_keys_exist_for_en_and_fi():
    for lang in ("en", "fi"):
        assert "ui_target_preview" in TRANSLATIONS[lang]
        assert "target_preview_legend_hint" in TRANSLATIONS[lang]


def test_file_status_translation_keys_exist_for_en_and_fi():
    keys = {
        "file_status_loaded",
        "file_status_not_loaded",
        "file_status_name",
        "file_status_format",
        "file_status_size",
        "file_status_clear",
        "file_status_unknown",
        "file_status_preview_source",
        "file_status_preview_upload",
        "file_status_preview_path",
        "file_status_preview_none",
        "file_status_upload",
        "file_status_local_path",
        "file_status_on_disk",
        "file_status_path_found",
        "file_status_path_missing",
    }
    for lang in ("en", "fi"):
        for key in keys:
            assert key in TRANSLATIONS[lang]


def test_advanced_guided_ui_translation_keys_exist_for_en_and_fi():
    keys = {
        "adv_shaping_title",
        "adv_shaping_fine_tune_title",
        "adv_bass_safety_title",
        "adv_bass_safety_fine_tune_title",
        "adv_conf_pull_title",
        "adv_conf_pull_tuning_title",
        "preset_safe",
        "preset_normal",
        "preset_aggressive",
        "adv_summary_global_rail",
        "adv_summary_boost_rail",
        "adv_summary_cut_rail",
        "adv_summary_max_cut",
        "adv_summary_transition",
        "adv_summary_smoothing",
        "adv_summary_phase_limit",
        "adv_summary_exc_prot",
        "adv_summary_low_bass_cut",
        "adv_summary_hpf",
        "adv_summary_bass_first",
        "adv_summary_floor",
        "adv_summary_ceil",
        "adv_summary_span",
        "state_on",
        "state_off",
        "toast_adv_preset_locked_auto",
    }
    for lang in ("en", "fi"):
        for key in keys:
            assert key in TRANSLATIONS[lang]


def test_resolve_user_manual_path_supports_frozen_bundle(tmp_path, monkeypatch):
    bundled_manual = tmp_path / "docs" / "User_Manual.md"
    bundled_manual.parent.mkdir(parents=True)
    bundled_manual.write_text("# Bundled manual\n", encoding="utf-8")

    monkeypatch.setattr(ng_app.sys, "_MEIPASS", str(tmp_path), raising=False)
    try:
        assert ng_app._resolve_user_manual_path() == Path(bundled_manual)
    finally:
        monkeypatch.delattr(ng_app.sys, "_MEIPASS", raising=False)
