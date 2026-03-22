from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Modality


@dataclass
class DocumentArtifact:
    """Canonical representation of an ingested document."""

    source_path: str
    mime_type: str
    content: bytes | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CharInterval:
    """Character position interval in source document."""

    start_pos: int
    end_pos: int

    def __len__(self) -> int:
        return self.end_pos - self.start_pos


@dataclass
class DocumentChunk:
    """Generic document chunk used across modalities."""

    id: str
    content: str | bytes
    source_path: str
    modality: Modality
    metadata: dict[str, Any] = field(default_factory=dict)
    char_interval: CharInterval | None = None


@dataclass
class TextChunk:
    """Text chunk for text-based extractors."""

    id: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    char_interval: CharInterval | None = None


@dataclass
class ImageChunk:
    """Image chunk for visual extractors."""

    id: str
    images: list[Any]
    source_path: str
    page_range: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Citation back to a document chunk and span."""

    source_path: str
    chunk_id: str
    span: CharInterval | None = None
    snippet: str | None = None


@dataclass
class ConfidenceScore:
    """Confidence score with optional rationale."""

    value: float
    rationale: str | None = None


@dataclass
class FieldResult:
    """Field-level result with optional confidence and citations."""

    name: str
    value: Any
    confidence: ConfidenceScore | None = None
    citations: list[Citation] = field(default_factory=list)


@dataclass
class ExtractorResult:
    """Result of running an extractor over chunks."""

    name: str
    provider_name: str
    results: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Normalized extraction result with metadata."""

    data: Any
    field_metadata: dict[str, FieldResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation output for structured data."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
