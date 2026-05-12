"""Core type definitions: enums, request/response data classes.

This module holds the data-class layer that used to live in base.py,
breaking the circular import between base.py (ABCs) and
artifacts.py/config.py (which need Modality, ProviderRequest, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(Enum):
    """Modality determines available features."""

    VISUAL = "visual"
    TEXT = "text"
    HYBRID = "hybrid"


@dataclass
class ProviderRequest:
    """Normalized provider request across text, vision, and structured outputs."""

    messages: list[dict[str, Any]]
    images: list[str] | None = None
    schema: dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Normalized provider response."""

    text: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None
