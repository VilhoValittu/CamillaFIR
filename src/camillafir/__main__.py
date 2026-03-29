from nicegui import ui
from camillafir.camillafir import PROGRAM_NAME, _get_main_app


def main():
    _get_main_app()  # register @ui.page('/') route
    ui.run(
        port=8080,
        show=True,
        dark=True,
        reload=False,
        title=PROGRAM_NAME,
    )


if __name__ == "__main__":
    main()
