from pywebio import start_server
from camillafir.camillafir import main as _app


def main():
    # Force fixed port for standalone app entrypoint
    start_server(_app, port=8080, debug=True, auto_open_webbrowser=True)


if __name__ == "__main__":
    main()
