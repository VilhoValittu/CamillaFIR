# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_data_files,
    collect_submodules,
)

block_cipher = None

# Build these BEFORE Analysis() (2-tuples are OK here)
binaries = []
datas = []
hiddenimports = []

# ----------------------------
# Resources (datas: (src, dest))
# ----------------------------
datas += [
    ("translations.json", "."),
    ("assets/plotly.min.js", "assets"),
]

# ----------------------------
# NumPy / SciPy binaries
# (binaries: typically (dest, src) or (src, dest) depending on helper;
# these helpers return what Analysis expects.)
# ----------------------------
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")

# ----------------------------
# Matplotlib datas
# ----------------------------
datas += collect_data_files("matplotlib", include_py_files=False)

# ----------------------------
# Hidden imports
# ----------------------------
hiddenimports += collect_submodules("pywebio")
hiddenimports += collect_submodules("scipy.signal")
hiddenimports += [
    "scipy.fft",
    "scipy.io",
    "scipy.io.wavfile",
    "scipy.signal.windows",
    "matplotlib.backends.backend_agg",
    "matplotlib.figure",
    # App internal modules (varmistus)
    "camillafir_dsp",
    "camillafir_modes",
    "camillafir_ui_helpers",
    "models",
]

analysis = Analysis(
    ["camillafir.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    exclude_binaries=True,
    name="CamillaFIR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=True,
    name="CamillaFIR",
)
