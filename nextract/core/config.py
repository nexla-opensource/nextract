from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Modality


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    name: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not self.name or not self.model:
            raise ValueError("Provider name and model are required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return True


@dataclass
class ExtractorConfig:
    """Configuration for an extractor."""

    name: str
    provider: ProviderConfig
    fallback_provider: Optional[ProviderConfig] = None
    enable_caching: bool = True
    batch_size: int = 1
    modality: Optional[Modality] = None
    extractor_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        self.provider.validate()
        if self.fallback_provider:
            self.fallback_provider.validate()

        import nextract.extractors  # noqa: F401

        from nextract.registry.extractor_registry import ExtractorRegistry

        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(self.name)
        if extractor_class:
            supported_providers = extractor_class.get_supported_providers()
            if self.provider.name not in supported_providers:
                raise ValueError(
                    f"Extractor '{self.name}' does not support "
                    f"provider '{self.provider.name}'"
                )
        return True


@dataclass
class ChunkerConfig:
    """Configuration for a chunker."""

    name: str
    pages_per_chunk: int = 5
    page_overlap: int = 1
    max_image_dimension: int = 2048
    image_quality: int = 95
    chunk_size: int = 2000
    chunk_overlap: int = 200
    preserve_tables: bool = True
    preserve_sections: bool = True
    respect_sentence_boundaries: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 10000

    def validate(self, modality: Modality) -> bool:
        import nextract.chunking  # noqa: F401

        from nextract.registry.chunker_registry import ChunkerRegistry

        chunker_registry = ChunkerRegistry.get_instance()
        chunker_class = chunker_registry.get(self.name)

        if chunker_class:
            applicable = chunker_class.get_applicable_modalities()
            if modality not in applicable:
                raise ValueError(
                    f"Chunker '{self.name}' is not applicable "
                    f"to modality '{modality.value}'. "
                    f"Applicable modalities: {[m.value for m in applicable]}"
                )

        if modality == Modality.VISUAL:
            if self.pages_per_chunk < 1:
                raise ValueError("pages_per_chunk must be >= 1")
            if self.page_overlap >= self.pages_per_chunk:
                raise ValueError("page_overlap must be < pages_per_chunk")
        elif modality == Modality.TEXT:
            if self.chunk_size < self.min_chunk_size:
                raise ValueError(f"chunk_size must be >= {self.min_chunk_size}")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be < chunk_size")

        return True


@dataclass
class ExtractionPlan:
    """Complete extraction plan."""

    extractor: ExtractorConfig
    chunker: ChunkerConfig
    num_passes: int = 1
    include_confidence: bool = True
    include_citations: bool = True
    include_raw_text: bool = False
    auto_suggest_schema: bool = False
    schema_validation: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0
    validation_rules: List[str] = field(default_factory=list)
    strict_validation: bool = False

    def validate(self) -> bool:
        self.extractor.validate()

        import nextract.extractors  # noqa: F401

        from nextract.registry.extractor_registry import ExtractorRegistry

        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(self.extractor.name)
        if extractor_class:
            modality = extractor_class.get_modality()
            self.chunker.validate(modality)

        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")

        return True
