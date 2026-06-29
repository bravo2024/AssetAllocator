from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import app.main  # noqa: F401, E402  delegate to app/main.py
