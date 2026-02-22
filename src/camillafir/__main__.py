from pywebio import start_server
from camillafir.camillafir import main as _app, resolve_static_dir


def main():
    # Force fixed port for standalone app entrypoint
    start_server(
        _app,
        port=8080,
        debug=True,
        auto_open_webbrowser=True,
        static_dir=resolve_static_dir(),
    )


if __name__ == "__main__":
    main()
