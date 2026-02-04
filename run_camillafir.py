import os
import sys

# Ensure src/ is on sys.path so package imports work when running from repo root.
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from camillafir.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
