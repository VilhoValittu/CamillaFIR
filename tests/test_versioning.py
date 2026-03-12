import importlib
import sys
import types

from camillafir.app_paths import program_version_token, safe_filters_dir
from camillafir.version import normalize_version


def test_normalize_version_accepts_tag_and_ui_formats():
    assert normalize_version("3.6.0") == "v.3.6.0"
    assert normalize_version("v3.6.0") == "v.3.6.0"
    assert normalize_version("v.3.6.0") == "v.3.6.0"


def test_resolve_version_prefers_package_build_version(monkeypatch):
    monkeypatch.delenv("CAMILLAFIR_VERSION", raising=False)
    build_version = types.ModuleType("camillafir.build_version")
    build_version.VERSION = "3.6.1"
    sys.modules["camillafir.build_version"] = build_version
    sys.modules.pop("camillafir.version", None)
    version_mod = importlib.import_module("camillafir.version")

    assert version_mod.resolve_version() == "v.3.6.1"


def test_program_version_token_and_filters_dir_use_same_version_format(tmp_path):
    version = "v.3.6.0"
    assert program_version_token(version) == "v3.6.0"

    filters_dir = safe_filters_dir(str(tmp_path), program_version=version)

    assert filters_dir.endswith("v3.6.0")
