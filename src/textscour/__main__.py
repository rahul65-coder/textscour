"""Executable package entry point for `python -m textscour`.

Delegates command-line execution to `textscour.cli.main()`.
"""

from __future__ import annotations

import sys

from textscour.cli import main

if __name__ == "__main__":
    sys.exit(main())
