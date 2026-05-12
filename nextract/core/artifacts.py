from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import Modality


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
    images: list[bytes | Any]  # bytes, bytearray, or PIL.Image.Image
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


@dataclass(frozen=True)
class ProviderUsage:
    """Typed usage metrics from a provider call."""

    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderUsage:
        return cls(
            requests=data.get("requests", 0) if isinstance(data.get("requests"), int) else 0,
            tool_calls=data.get("tool_calls", 0) if isinstance(data.get("tool_calls"), int) else 0,
            input_tokens=data.get("input_tokens", 0) if isinstance(data.get("input_tokens"), int) else 0,
            output_tokens=data.get("output_tokens", 0) if isinstance(data.get("output_tokens"), int) else 0,
            details=data.get("details", {}) if isinstance(data.get("details"), dict) else {},
        )


@dataclass(frozen=True)
class ChunkExtraction:
    """Typed result from extracting a single chunk."""

    chunk_id: str
    response: dict[str, Any] | list[Any] | str
    usage: ProviderUsage | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "response": self.response,
        }
        if self.usage is not None:
            d["usage"] = self.usage.to_dict()
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkExtraction:
        usage = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage = ProviderUsage.from_dict(raw_usage)
        return cls(
            chunk_id=data.get("chunk_id", ""),
            response=data.get("response"),
            usage=usage,
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )


@dataclass
class ExtractorResult:
    """Result of running an extractor over chunks."""

    name: str
    provider_name: str
    results: list[ChunkExtraction]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregated_usage(self) -> ProviderUsage:
        totals = ProviderUsage()
        for r in self.results:
            if r.usage is not None:
                totals = ProviderUsage(
                    requests=totals.requests + r.usage.requests,
                    tool_calls=totals.tool_calls + r.usage.tool_calls,
                    input_tokens=totals.input_tokens + r.usage.input_tokens,
                    output_tokens=totals.output_tokens + r.usage.output_tokens,
                    details=totals.details,
                )
        return totals


@dataclass
class ExtractionResult:
    """Normalized extraction result with metadata."""

    data: dict[str, Any] | list[Any] | str | None
    field_metadata: dict[str, FieldResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation output for structured data."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
