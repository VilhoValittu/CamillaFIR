from nicegui import ui
from camillafir.camillafir import PROGRAM_NAME, configure_main_app


def main():
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
