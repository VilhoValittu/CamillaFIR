from pathlib import Path


def test_pywebio_hook_includes_default_theme_css():
    hook_path = Path(__file__).resolve().parents[1] / "pyinstaller_hooks" / "hook-pywebio.py"
    hook_text = hook_path.read_text(encoding="utf-8")

    assert '"html/css/bs-theme/default.min.css"' in hook_text
    assert '"html/css/bs-theme/dark.min.css"' in hook_text
