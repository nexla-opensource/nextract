"""Deprecated retry policy stub.

This module is deprecated. Retry configuration is handled by:
- ``ExtractionPlan.max_retries`` and ``ExtractionPlan.backoff_factor`` for pipeline-level retries.
- ``ProviderConfig.max_retries`` for provider-level retries.
- ``RuntimeConfig.max_run_retries`` for agent-runner retries.

Will be removed in a future release.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryPolicy:
    """Deprecated: use ExtractionPlan.max_retries / backoff_factor instead."""

    max_retries: int = 3
    backoff_factor: float = 2.0
