from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog to emit JSON logs on stderr.

    Logs always go to stderr so CLI machine-readable output on stdout
    (e.g. ``print_json`` from ``nextract extract`` / ``batch``) stays
    pipe-clean for tools like ``jq``.

    Call this explicitly from application entry points (CLI, scripts).
    Do not rely on side effects at import time.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Prefer a single StreamHandler bound to stderr.
    # Drop any StreamHandlers writing to stdout so extraction JSON
    # on stdout is never mixed with log lines.
    kept: list[logging.Handler] = []
    has_stderr = False
    for handler in root.handlers:
        stream = getattr(handler, "stream", None)
        if isinstance(handler, logging.StreamHandler) and stream is sys.stdout:
            continue
        if isinstance(handler, logging.StreamHandler) and stream is sys.stderr:
            has_stderr = True
            handler.setLevel(level)
        kept.append(handler)
    root.handlers = kept

    if not has_stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
