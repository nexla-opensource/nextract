from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsEvent:
    """Lightweight metrics event container."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


class MetricsClient:
    """Placeholder metrics client for future integrations."""

    def emit(self, event: MetricsEvent) -> None:
        _ = event
        return
