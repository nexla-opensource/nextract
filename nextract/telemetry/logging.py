from __future__ import annotations

import logging
import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog for pretty console logs."""
    # Note: Do not call logging.basicConfig() inside a library.
    # Users should configure their own logging before calling setup_logging().
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
