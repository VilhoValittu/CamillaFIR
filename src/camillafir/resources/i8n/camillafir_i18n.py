# camillafir_i18n.py
import json
import locale
import os
import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> str:
    """Returns the path to a resource, whether inside an EXE package or in development."""
    if hasattr(sys, "_MEIPASS"):
        base = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(base):
            return base
        # Fallback for older bundle layouts
        alt = os.path.join(sys._MEIPASS, "camillafir", "resources", relative_path)
        return alt
    package_root = Path(__file__).resolve().parents[1]
    return str(package_root / relative_path)


TRANS_FILE = get_resource_path(os.path.join("i8n", "translations.json"))


def load_translations() -> dict:
    try:
        with open(TRANS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


TRANSLATIONS = load_translations()


def t(key: str) -> str:
    """
    Translation helper (fi/en). Falls back to English, then to key.
    Preserves current special-cases from camillafir.py.
    """
    lang = locale.getlocale()[0]
    lang = "fi" if lang and "fi" in lang.lower() else "en"

    # current hardcoded special cases
    if key == "zoom_hint":
        return "(Vinkki: Voit zoomata hiirellä kuvaajaa)" if lang == "fi" else "(Hint: Use mouse to zoom)"
    return TRANSLATIONS.get(lang, TRANSLATIONS.get("en", {})).get(key, key)
