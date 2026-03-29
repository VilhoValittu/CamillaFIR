from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "spec_name",
    [
        "CamillaFIR_standalone.spec",
        "CamillaFIR_linux.spec",
        "CamillaFIR_macos.spec",
    ],
)
def test_user_manual_is_included_in_pyinstaller_spec(spec_name: str):
    spec_path = Path(__file__).resolve().parents[1] / spec_name
    spec_text = spec_path.read_text(encoding="utf-8")

    assert '("docs/User_Manual.md", "docs")' in spec_text
