import os
import sys

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from nicegui import ui
from camillafir.camillafir import PROGRAM_NAME, configure_main_app


def main():
    os.environ.setdefault("CAMILLAFIR_AUTO_PROFILE", "1")
    configure_main_app()
    ui.run(
        port=8080,
        show=True,
        dark=True,
        reload=False,
        title=PROGRAM_NAME,
    )


if __name__ == "__main__":
    main()
