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


def test_resolve_user_manual_path_supports_frozen_bundle(tmp_path, monkeypatch):
    bundled_manual = tmp_path / "docs" / "User_Manual.md"
    bundled_manual.parent.mkdir(parents=True)
    bundled_manual.write_text("# Bundled manual\n", encoding="utf-8")

    monkeypatch.setattr(ng_app.sys, "_MEIPASS", str(tmp_path), raising=False)
    try:
        assert ng_app._resolve_user_manual_path() == Path(bundled_manual)
    finally:
        monkeypatch.delattr(ng_app.sys, "_MEIPASS", raising=False)
