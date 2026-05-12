from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .types import Modality


# Known keys in extra_params that contain secrets
_SECRET_EXTRA_PARAMS = frozenset({
    "aws_secret_key", "aws_session_token", "api_key", "secret_key",
    "password", "token", "credential",
})


def _mask_extra_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of extra_params with secret values masked."""
    return {
        k: "***" if k in _SECRET_EXTRA_PARAMS else v
        for k, v in extra_params.items()
    }


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    name: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not self.name or not self.model:
            raise ValueError("Provider name and model are required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return True

    def __repr__(self) -> str:
        extra = _mask_extra_params(self.extra_params)
        return (
            f"ProviderConfig(name={self.name!r}, model={self.model!r}, "
            f"api_base={self.api_base!r}, timeout={self.timeout}, "
            f"max_retries={self.max_retries}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, extra_params={extra!r})"
        )


@dataclass
class ExtractorConfig:
    """Configuration for an extractor."""

    name: str
    provider: ProviderConfig
    fallback_provider: ProviderConfig | None = None
    enable_caching: bool = True
    batch_size: int = 1
    modality: Modality | None = None
    extractor_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        self.provider.validate()
        if self.fallback_provider:
            self.fallback_provider.validate()

        if not self.name:
            raise ValueError("Extractor name is required")

        return True


@dataclass
class ChunkerConfig:
    """Configuration for a chunker.

    Contains both visual and text parameters. Use validate() with the
    target modality to get relevant validation and warnings about
    irrelevant settings.
    """

    name: str
    # Visual-modality parameters
    pages_per_chunk: int = 5
    page_overlap: int = 1
    max_image_dimension: int = 2048
    image_quality: int = 95
    # Text-modality parameters
    chunk_size: int = 2000
    chunk_overlap: int = 200
    preserve_tables: bool = True
    preserve_sections: bool = True
    respect_sentence_boundaries: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 10000

    def validate(self, modality: Modality) -> bool:
        if not self.name:
            raise ValueError("Chunker name is required")

        if modality == Modality.VISUAL:
            if self.pages_per_chunk < 1:
                raise ValueError("pages_per_chunk must be >= 1")
            if self.page_overlap >= self.pages_per_chunk:
                raise ValueError("page_overlap must be < pages_per_chunk")
            if self.chunk_size != 2000 or self.chunk_overlap != 200:
                warnings.warn(
                    f"Text chunking settings (chunk_size={self.chunk_size}, "
                    f"chunk_overlap={self.chunk_overlap}) are set but ignored "
                    f"for VISUAL modality chunker '{self.name}'",
                    stacklevel=3,
                )
        elif modality == Modality.TEXT:
            if self.chunk_size < self.min_chunk_size:
                raise ValueError(f"chunk_size must be >= {self.min_chunk_size}")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be < chunk_size")
            if self.pages_per_chunk != 5 or self.page_overlap != 1:
                warnings.warn(
                    f"Visual chunking settings (pages_per_chunk={self.pages_per_chunk}, "
                    f"page_overlap={self.page_overlap}) are set but ignored "
                    f"for TEXT modality chunker '{self.name}'",
                    stacklevel=3,
                )

        return True


@dataclass
class ExtractionPlan:
    """Complete extraction plan.

    Retry configuration: ``max_retries`` and ``backoff_factor`` here are the
    single source of truth for pipeline-level retries. These propagate to
    ``ProviderConfig.max_retries`` at plan validation time if the provider
    config does not specify an override.
    """

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
    validation_rules: list[str] = field(default_factory=list)
    strict_validation: bool = False

    def validate(self) -> bool:
        self.extractor.validate()

        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")
        if self.num_passes > 20:
            raise ValueError(f"num_passes must be <= 20, got {self.num_passes}")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")

        # Propagate max_retries to provider if not explicitly set
        if self.retry_on_failure and self.max_retries != self.extractor.provider.max_retries:
            self.extractor.provider.max_retries = self.max_retries

        return True
