from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules


datas = collect_data_files(
    "plotly",
    includes=[
        "package_data/plotly.min.js",
        "package_data/templates/*.json",
        "validators/**/*.*",
    ],
)

hiddenimports = collect_submodules("plotly.validators") + ["pandas", "cmath"]
