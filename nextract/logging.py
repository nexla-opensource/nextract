from __future__ import annotations

import warnings
from nextract.telemetry.logging import setup_logging  # noqa: E402

warnings.warn(
    "nextract.logging is deprecated. Use nextract.telemetry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["setup_logging"]
