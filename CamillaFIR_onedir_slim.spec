# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# --- Hidden imports: koko camillafir-paketti ---
hiddenimports = collect_submodules("camillafir")

# --- Data files (runtime resources) ---
datas = [
    # i18n
    (
        "src/camillafir/resources/i8n/translations.json",
        "camillafir/resources/i8n",
    ),
    # plotly
    (
        "src/camillafir/resources/plotly/plotly.min.js",
        "camillafir/resources/plotly",
    ),
]

a = Analysis(
    ["src/camillafir/camillafir.py"],
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
    a.binaries,
    a.datas,
    [],
    name="CamillaFIR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # True jos haluat konsoli-ikkunan
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="CamillaFIR",
)
