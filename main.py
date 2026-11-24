#!/usr/bin/env python3
"""FocusBuddy entrypoint (Author #1 / Vaishnavi — session app)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from focusbuddy.data.storage import ensure_data_dirs
from focusbuddy.ui.main_window import MainWindow


def main() -> None:
    ensure_data_dirs()
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
