# -*- mode: python ; coding: utf-8 -*-
#
# macOS app-bundle spec for CamillaFIR.
# Produces: dist/CamillaFIR.app
#

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [
    # i18n (expected at _MEIPASS/i8n/translations.json)
    ("src/camillafir/resources/i8n/translations.json", "i8n"),
    # plotly.js (expected at _MEIPASS/assets/plotly.min.js)
    ("src/camillafir/resources/plotly/plotly.min.js", "assets"),
    # UI logo (expected at _MEIPASS/camillafir/ui/assets/camillafir_logo.png)
    ("src/camillafir/ui/assets/camillafir_logo.png", "camillafir/ui/assets"),
]

# PyWebIO loads some components dynamically; keep these minimal hidden imports.
hiddenimports = [
    "pywebio.platform.tornado_http",
    "pywebio.platform.tornado_websocket",
    "pywebio.platform.tornado",
    "pywebio.platform",
    "pywebio.session",
    "pywebio.io_ctrl",
]
hiddenimports += collect_submodules("optuna")

# Include PyWebIO static assets if present (kept minimal by package)
datas += collect_data_files("pywebio")

a = Analysis(
    ["src/camillafir/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tests",
        "tools",
        ".pytest_cache",
        "matplotlib.tests",
        "scipy.tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CamillaFIR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CamillaFIR",
)

app = BUNDLE(
    coll,
    name="CamillaFIR.app",
    icon=None,
    bundle_identifier=None,
)
