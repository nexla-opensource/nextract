from .base import (
    BaseChunker,
    BaseExtractor,
    BaseFormatter,
    BaseProvider,
    BaseValidator,
    Modality,
    ProviderRequest,
    ProviderResponse,
)
from .config import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from .artifacts import (
    CharInterval,
    Citation,
    ConfidenceScore,
    DocumentArtifact,
    DocumentChunk,
    ExtractionResult,
    ExtractorResult,
    FieldResult,
    ImageChunk,
    TextChunk,
    ValidationResult,
)
from .exceptions import (
    ChunkerError,
    ExtractorError,
    NextractError,
    PipelineError,
    PlanError,
    ProviderError,
    ValidationError,
)

__all__ = [
    "BaseChunker",
    "BaseExtractor",
    "BaseFormatter",
    "BaseProvider",
    "BaseValidator",
    "Modality",
    "ProviderRequest",
    "ProviderResponse",
    "ChunkerConfig",
    "ExtractionPlan",
    "ExtractorConfig",
    "ProviderConfig",
    "CharInterval",
    "Citation",
    "ConfidenceScore",
    "DocumentArtifact",
    "DocumentChunk",
    "ExtractionResult",
    "ExtractorResult",
    "FieldResult",
    "ImageChunk",
    "TextChunk",
    "ValidationResult",
    "ChunkerError",
    "ExtractorError",
    "NextractError",
    "PipelineError",
    "PlanError",
    "ProviderError",
    "ValidationError",
]


def extract(*args, **kwargs):
    """Legacy extract entrypoint retained for backward compatibility."""
    from nextract.legacy_core import extract as legacy_extract

    return legacy_extract(*args, **kwargs)


def batch_extract(*args, **kwargs):
    """Legacy batch extraction entrypoint retained for backward compatibility."""
    from nextract.legacy_core import batch_extract as legacy_batch_extract

    return legacy_batch_extract(*args, **kwargs)
