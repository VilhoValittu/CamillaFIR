# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_data_files,
    collect_submodules,
)

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# ----------------------------
# App resources
# ----------------------------
datas += [("translations.json", ".")]
datas += [
    ("assets/plotly.min.js", "assets"),
]

# ----------------------------
# NumPy (dynlibs + minimal data)
# ----------------------------
binaries += collect_dynamic_libs("numpy")

# ----------------------------
# SciPy (dynlibs + only used submodules)
# ----------------------------
binaries += collect_dynamic_libs("scipy")
hiddenimports += [
    "scipy.signal",
    "scipy.fft",
    "scipy.io",
    "scipy.io.wavfile",
    "scipy.ndimage",
]

# ----------------------------
# Matplotlib (Agg-only)
# ----------------------------
datas += collect_data_files("matplotlib", include_py_files=False)
hiddenimports += [
    "matplotlib.backends.backend_agg",
]

# ----------------------------
# PyWebIO
# ----------------------------
datas += collect_data_files("pywebio", include_py_files=False)

# ----------------------------
# Plotly (optional, but makes export/plots robust)
# ----------------------------
datas += collect_data_files("plotly", include_py_files=False)
hiddenimports += collect_submodules("plotly")

# ----------------------------
# Kaleido (Plotly PNG export)
# ----------------------------
# Kaleido 1.x uses internal "scopes" and helper deps that PyInstaller may miss.
datas += collect_data_files("kaleido", include_py_files=False)
hiddenimports += [
    "kaleido",
    "kaleido.scopes",
    "kaleido.scopes.plotly",
    # Kaleido runtime deps (commonly needed)
    "choreographer",
    "logistro",
    "orjson",
    "packaging",
]

# ----------------------------
# EXCLUDES: trim tests + GUI backends + heavy extras
# ----------------------------
excludes = [
    "tkinter",
    "PyQt5",
    "PySide6",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "pandas",
    "numpy.tests",
    "scipy.tests",
    "matplotlib.tests",
]

# ----------------------------
# Project modules (ensure dynamic imports don't miss these)
# ----------------------------
hiddenimports += collect_submodules("camillafir_io")
hiddenimports += [
    "camillafir_analysis",
    "camillafir_config",
    "camillafir_dsp",
    "camillafir_housecurve",
    "camillafir_i18n",
    "camillafir_leveling",
    "camillafir_modes",
    "camillafir_pipeline",
    "camillafir_plot",
    "camillafir_ui_helpers",
    "camillafir_wav_window",
    "models",
]

# DSP package modules (new structure)
hiddenimports += [
    "dsp",
    "dsp.analysis",
    "dsp.smoothing",
    "dsp.limits",
    "dsp.tdc",
    "dsp.phase",
    "dsp.bassfirst",
]

a = Analysis(
    ["camillafir.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CamillaFIR",
    debug=False,
    strip=False,
    upx=False,  # keep OFF: SciPy/NumPy can be sensitive to UPX
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="CamillaFIR",
)
