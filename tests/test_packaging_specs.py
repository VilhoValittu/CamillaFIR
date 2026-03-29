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


def test_release_workflow_verifies_manual_in_recursive_bundle_layouts():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert 'find dist/CamillaFIR -path "*docs/User_Manual.md" -print -quit | grep -q .' in workflow_text
    assert 'Get-ChildItem -Path "dist\\CamillaFIR" -Recurse -Filter "User_Manual.md" -File' in workflow_text
    assert "test -f dist/CamillaFIR/docs/User_Manual.md" not in workflow_text
    assert 'Test-Path "dist\\CamillaFIR\\docs\\User_Manual.md"' not in workflow_text
