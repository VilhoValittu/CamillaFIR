import os
import pathlib
import subprocess
import sys

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

_IMPORT_SCRIPT = """
import builtins
import importlib
import sys

module_name = sys.argv[1]

for key in list(sys.modules):
    if key == "pywebio" or key.startswith("pywebio."):
        del sys.modules[key]

orig_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pywebio" or name.startswith("pywebio."):
        raise ModuleNotFoundError("No module named 'pywebio'")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
try:
    importlib.import_module(module_name)
finally:
    builtins.__import__ = orig_import
"""


@pytest.mark.parametrize(
    "module_name",
    [
        "camillafir.io.camillafir_automatic_mode",
        "camillafir.io.auto_mode.target_preselection",
        "camillafir.camillafir",
    ],
)
def test_headless_imports_do_not_require_pywebio(module_name):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_SRC) if not existing else os.pathsep.join([str(_SRC), existing])
    )

    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, module_name],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        f"Import failed for {module_name}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
