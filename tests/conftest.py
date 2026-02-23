import sys
import pathlib
import pytest

# Ensure project root (and optional src/) are importable regardless of cwd
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if _p.exists():
        _sp = str(_p)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

from camillafir.io.measurements_txt import parse_measurements_from_path


@pytest.fixture(scope="session")
def lr_measurements():
    """
    Load identical L/R measurements from tests/data/L.txt
    (file added by user).
    """
    p = pathlib.Path(__file__).parent / "data" / "L.txt"
    f, m, pdeg = parse_measurements_from_path(str(p))
    return (f, m, pdeg), (f, m, pdeg)
