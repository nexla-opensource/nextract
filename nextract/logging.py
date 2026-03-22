from __future__ import annotations

import warnings

warnings.warn(
    "nextract.logging is deprecated. Use nextract.telemetry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from nextract.telemetry.logging import setup_logging

__all__ = ["setup_logging"]
