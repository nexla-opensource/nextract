from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any


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


class BaseProvider(ABC):
    """Base interface for all LLM/API providers."""

    @abstractmethod
    def initialize(self, config: "ProviderConfig") -> None:
        """Initialize provider with configuration."""
        ...

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a response for the given request."""
        ...

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether provider supports vision inputs."""
        ...

    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether provider supports structured output."""
        ...

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return provider capabilities."""
        ...


class BaseExtractor(ABC):
    """Base interface for all extractors."""

    @abstractmethod
    def initialize(self, config: "ExtractorConfig") -> None:
        """Initialize extractor with configuration."""
        ...

    @abstractmethod
    def run(
        self,
        input_data: Any,
        provider: BaseProvider,
        **kwargs: Any,
    ) -> "ExtractorResult":
        """Run extraction using the given provider."""
        ...

    @classmethod
    @abstractmethod
    def get_modality(cls) -> Modality:
        """Return the modality this extractor uses."""
        ...

    @classmethod
    @abstractmethod
    def get_supported_providers(cls) -> list[str]:
        """Return list of compatible provider names."""
        ...

    @abstractmethod
    def validate_config(self, config: "ExtractorConfig") -> bool:
        """Validate extractor configuration."""
        ...


class BaseChunker(ABC):
    """Base interface for chunkers."""

    @classmethod
    @abstractmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        """Return modalities where this chunker is applicable."""
        ...

    @abstractmethod
    def chunk(self, document: "DocumentArtifact", config: "ChunkerConfig") -> list["DocumentChunk"]:
        """Chunk document according to the chunker."""
        ...

    @abstractmethod
    def validate_config(self, config: "ChunkerConfig") -> bool:
        """Validate chunker configuration."""
        ...


class BaseValidator(ABC):
    """Base interface for validators."""

    @abstractmethod
    def validate(self, data: dict[str, Any], schema: dict[str, Any], **kwargs: Any) -> "ValidationResult":
        """Validate extracted data."""
        ...


class BaseFormatter(ABC):
    """Base interface for output formatters."""

    @abstractmethod
    def format(self, result: ExtractionResult, **kwargs: Any) -> Any:
        """Format extraction results."""
        ...


from .artifacts import DocumentArtifact, DocumentChunk, ExtractorResult, ValidationResult  # noqa: E402
from .config import ProviderConfig, ExtractorConfig, ChunkerConfig  # noqa: E402

if TYPE_CHECKING:
    from .artifacts import ExtractionResult
