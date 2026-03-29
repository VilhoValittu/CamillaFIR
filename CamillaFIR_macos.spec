# -*- mode: python ; coding: utf-8 -*-
#
# macOS app-bundle spec for CamillaFIR.
# Produces: dist/CamillaFIR.app
#

block_cipher = None

datas = [
    # i18n (expected at _MEIPASS/i8n/translations.json)
    ("src/camillafir/resources/i8n/translations.json", "i8n"),
    # AUTO-mode priors (expected relative to camillafir/resources)
    ("src/camillafir/resources/auto_mode_filter_priors.json", "camillafir/resources"),
    # plotly.js (expected at _MEIPASS/assets/plotly.min.js)
    ("src/camillafir/resources/plotly/plotly.min.js", "assets"),
    # UI logo (expected at _MEIPASS/camillafir/ui/assets/camillafir_logo.png)
    ("src/camillafir/ui/assets/camillafir_logo.png", "camillafir/ui/assets"),
]

# PyWebIO loads some components dynamically; keep these minimal hidden imports.
hiddenimports = [
    "pywebio.platform.tornado_http",
    "pywebio.platform.tornado",
    "pywebio.platform",
    "pywebio.session",
    "pywebio.io_ctrl",
    # AUTO mode imports Optuna dynamically; keep only the modules it needs.
    "optuna",
]

a = Analysis(
    ["src/camillafir/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyinstaller_hooks"],
    hooksconfig={
        # CamillaFIR renders matplotlib only via Agg.
        "matplotlib": {"backends": ["Agg"]},
    },
    runtime_hooks=[],
    excludes=[
        "tests",
        "tools",
        ".pytest_cache",
        "matplotlib.tests",
        "scipy.tests",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
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
    icon="src/camillafir/ui/assets/camillafir_logo.icns",
    bundle_identifier=None,
)
