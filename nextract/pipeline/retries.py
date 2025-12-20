from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_factor: float = 2.0
