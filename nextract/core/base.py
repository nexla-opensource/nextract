from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional


class Modality(Enum):
    """Modality determines available features."""

    VISUAL = "visual"
    TEXT = "text"
    HYBRID = "hybrid"


@dataclass
class ProviderRequest:
    """Normalized provider request across text, vision, and structured outputs."""

    messages: List[Dict[str, Any]]
    images: Optional[List[str]] = None
    schema: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Normalized provider response."""

    text: str
    structured_output: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Any = None


class BaseProvider(ABC):
    """Base interface for all LLM/API providers."""

    @abstractmethod
    def initialize(self, config: "ProviderConfig") -> None:
        """Initialize provider with configuration."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a response for the given request."""
        raise NotImplementedError

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether provider supports vision inputs."""
        raise NotImplementedError

    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether provider supports structured output."""
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities."""
        raise NotImplementedError


class BaseExtractor(ABC):
    """Base interface for all extractors."""

    @abstractmethod
    def initialize(self, config: "ExtractorConfig") -> None:
        """Initialize extractor with configuration."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        input_data: Any,
        provider: BaseProvider,
        **kwargs: Any,
    ) -> "ExtractorResult":
        """Run extraction using the given provider."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_modality(cls) -> Modality:
        """Return the modality this extractor uses."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_providers(cls) -> List[str]:
        """Return list of compatible provider names."""
        raise NotImplementedError

    @abstractmethod
    def validate_config(self, config: "ExtractorConfig") -> bool:
        """Validate extractor configuration."""
        raise NotImplementedError


class BaseChunker(ABC):
    """Base interface for chunkers."""

    @classmethod
    @abstractmethod
    def get_applicable_modalities(cls) -> List[Modality]:
        """Return modalities where this chunker is applicable."""
        raise NotImplementedError

    @abstractmethod
    def chunk(self, document: "DocumentArtifact", config: "ChunkerConfig") -> List["DocumentChunk"]:
        """Chunk document according to the chunker."""
        raise NotImplementedError

    @abstractmethod
    def validate_config(self, config: "ChunkerConfig") -> bool:
        """Validate chunker configuration."""
        raise NotImplementedError


class BaseValidator(ABC):
    """Base interface for validators."""

    @abstractmethod
    def validate(self, data: Dict[str, Any], schema: Dict[str, Any], **kwargs: Any) -> "ValidationResult":
        """Validate extracted data."""
        raise NotImplementedError


class BaseFormatter(ABC):
    """Base interface for output formatters."""

    @abstractmethod
    def format(self, result: ExtractionResult, **kwargs: Any) -> Any:
        """Format extraction results."""
        raise NotImplementedError


from .artifacts import DocumentArtifact, DocumentChunk, ExtractorResult, ValidationResult  # noqa: E402
from .config import ProviderConfig, ExtractorConfig, ChunkerConfig  # noqa: E402

if TYPE_CHECKING:
    from .artifacts import ExtractionResult
