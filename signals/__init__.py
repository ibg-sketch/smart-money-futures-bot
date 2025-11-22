"""Public entrypoints for signal generation.

This module defensively loads the scoring layer so deployments that run the
bot from different working directories (or under packaged paths like /app)
still resolve the bundled `signals` package. If the scoring module is truly
missing, a clear error is raised to prompt a redeploy that includes the
refactored signal files.
"""

from importlib import import_module
from pathlib import Path
import sys


# Ensure the project root (one level above this file) is importable even when
# the process starts from a different working directory (e.g., /app).
_PKG_DIR = Path(__file__).resolve().parent
_ROOT = _PKG_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


try:
    decide_signal = import_module("signals.scoring").decide_signal
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    available = sorted(p.name for p in _PKG_DIR.glob("*.py"))
    raise ModuleNotFoundError(
        "signals.scoring is missing. Ensure the signals package (features, "
        "scoring, formatting, calibration) is deployed alongside main.py. "
        f"Current contents: {available}"
    ) from exc


__all__ = ["decide_signal"]
