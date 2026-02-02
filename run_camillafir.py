import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))
    runpy.run_module("camillafir.camillafir", run_name="__main__")


if __name__ == "__main__":
    main()
