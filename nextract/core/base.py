from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .types import Modality, ProviderRequest, ProviderResponse


class BaseProvider(ABC):
    """Base interface for all LLM/API providers."""

    name: str  # Set during initialize() from config

    @abstractmethod
    def initialize(self, config: ProviderConfig) -> None:
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

    def get_name(self) -> str:
        """Return the provider name."""
        return getattr(self, "name", "unknown")


class BaseExtractor(ABC):
    """Base interface for all extractors."""

    @abstractmethod
    def initialize(self, config: ExtractorConfig) -> None:
        """Initialize extractor with configuration."""
        ...

    @abstractmethod
    def run(
        self,
        input_data: Any,
        provider: BaseProvider,
        **kwargs: Any,
    ) -> ExtractorResult:
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
    def validate_config(self, config: ExtractorConfig) -> bool:
        """Validate extractor configuration."""
        ...


class BaseChunker(ABC):
    """Base interface for chunkers."""

    @classmethod
    @abstractmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        """Return modalities where this chunker is applicable."""
        ...

    def initialize(self, config: ChunkerConfig) -> None:
        """Initialize chunker with configuration. Override for resource preloading."""
        ...

    @abstractmethod
    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[DocumentChunk]:
        """Chunk document according to the chunker."""
        ...

    @abstractmethod
    def validate_config(self, config: ChunkerConfig) -> bool:
        """Validate chunker configuration."""
        ...


class BaseValidator(ABC):
    """Base interface for validators."""

    @abstractmethod
    def validate(self, data: Any, schema: dict[str, Any], **kwargs: Any) -> ValidationResult:
        """Validate extracted data."""
        ...


class BaseFormatter(ABC):
    """Base interface for output formatters."""

    @abstractmethod
    def format(self, result: ExtractionResult, **kwargs: Any) -> Any:
        """Format extraction results."""
        ...


if TYPE_CHECKING:
    from .artifacts import DocumentArtifact, DocumentChunk, ExtractionResult, ExtractorResult, ValidationResult
    from .config import ChunkerConfig, ExtractorConfig, ProviderConfig
