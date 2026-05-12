from __future__ import annotations

from typing import Any

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nextract")
except PackageNotFoundError:
    __version__ = "unknown"

from nextract.config import get_default_model_for_provider  # noqa: E402
from nextract.core import (  # noqa: E402
    BaseChunker,
    BaseExtractor,
    BaseFormatter,
    BaseProvider,
    ChunkerConfig,
    ExtractionPlan,
    ExtractionResult,
    ExtractorConfig,
    Modality,
    ProviderConfig,
)
from nextract.output import CsvFormatter, HtmlFormatter, JsonFormatter, MarkdownFormatter  # noqa: E402
from nextract.pipeline import BatchExtractionResult, BatchPipeline, ExtractionPipeline  # noqa: E402
from nextract.registry import (  # noqa: E402
    ChunkerRegistry,
    ExtractorRegistry,
    ProviderRegistry,
    register_chunker,
    register_extractor,
    register_provider,
)
from nextract.schema import SchemaGenerator, SchemaSuggestion  # noqa: E402


def extract_simple(
    document: str,
    schema: dict,
    provider: str,
    model: str | None = None,
    prompt: str | None = None,
    mode: str = "auto",
) -> Any:
    """Simplest extraction helper with automatic mode selection.

    Args:
        document: Path to the document to extract from.
        schema: JSON Schema for the expected output.
        provider: Provider name (e.g., "openai", "anthropic", "google").
        model: Model name (defaults to provider's default).
        prompt: Optional extraction prompt.
        mode: Extraction mode - "auto", "text", "visual", "ocr", or "textract".
            "auto" selects based on document type: visual for images/PDFs,
            text for text files. "text" forces text-only extraction.
            "visual" forces VLM extraction. "ocr" forces OCR extraction.
            "textract" forces AWS Textract extraction.
    """
    if model is None:
        model = get_default_model_for_provider(provider)
    provider_config = ProviderConfig(name=provider, model=model)

    if mode == "auto":
        mode = _detect_extraction_mode(document)

    extractor_name, chunker_name = _mode_to_extractor_chunker(mode, provider)

    extractor_config = ExtractorConfig(name=extractor_name, provider=provider_config)
    chunker_config = ChunkerConfig(name=chunker_name)
    plan = ExtractionPlan(extractor=extractor_config, chunker=chunker_config)

    pipeline = ExtractionPipeline(plan)
    return pipeline.extract(document=document, schema=schema, prompt=prompt)


def _detect_extraction_mode(document: str) -> str:
    """Detect the best extraction mode based on document type."""
    import os

    path = document
    ext = os.path.splitext(path)[1].lower()

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
    visual_extensions = image_extensions | {".pdf"}
    text_extensions = {".txt", ".csv", ".json", ".xml", ".html", ".md", ".yaml", ".yml"}

    if ext in image_extensions:
        return "visual"
    if ext in visual_extensions:
        return "visual"
    if ext in text_extensions:
        return "text"

    # Default to text for unknown extensions
    return "text"


def _mode_to_extractor_chunker(mode: str, provider: str) -> tuple[str, str]:
    """Map extraction mode to extractor and chunker names."""
    ocr_providers = {"tesseract", "easyocr", "paddleocr"}

    if mode == "textract":
        return "textract", "page"
    if mode == "ocr" or provider in ocr_providers:
        return "ocr", "page"
    if mode == "visual":
        return "vlm", "page"
    # mode == "text" or default
    return "text", "semantic"


def extract(
    document: str | list[str],
    schema: dict,
    plan: ExtractionPlan,
    prompt: str | None = None,
    examples: list[dict] | None = None,
    include_extra: bool = False,
):
    """Run extraction with a fully specified plan."""
    pipeline = ExtractionPipeline(plan)
    return pipeline.extract(
        document=document,
        schema=schema,
        prompt=prompt,
        examples=examples,
        include_extra=include_extra,
    )


def batch_extract(
    documents: list[str],
    schema: dict,
    plan: ExtractionPlan,
    prompt: str | None = None,
    examples: list[dict] | None = None,
    include_extra: bool = False,
    max_workers: int = 4,
):
    """Run batch extraction with a fully specified plan."""
    pipeline = BatchPipeline(plan=plan, max_workers=max_workers)
    return pipeline.extract_batch(
        documents=documents,
        schema=schema,
        prompt=prompt,
        examples=examples,
        include_extra=include_extra,
    )


def get_available_chunkers(extractor_name: str) -> list[str]:
    extractor_class = ExtractorRegistry.get_instance().get(extractor_name)
    if not extractor_class:
        return []
    modality = extractor_class.get_modality()
    return ChunkerRegistry.get_instance().get_chunkers_for_modality(modality)


__all__ = [
    "BaseChunker",
    "BaseExtractor",
    "BaseFormatter",
    "BaseProvider",
    "BatchPipeline",
    "BatchExtractionResult",
    "ChunkerConfig",
    "CsvFormatter",
    "ExtractionPipeline",
    "ExtractionPlan",
    "ExtractionResult",
    "ExtractorConfig",
    "HtmlFormatter",
    "JsonFormatter",
    "MarkdownFormatter",
    "Modality",
    "ProviderConfig",
    "ProviderRegistry",
    "ExtractorRegistry",
    "ChunkerRegistry",
    "SchemaGenerator",
    "SchemaSuggestion",
    "batch_extract",
    "extract",
    "extract_simple",
    "get_available_chunkers",
    "get_default_model_for_provider",
    "register_chunker",
    "register_extractor",
    "register_provider",
    "__version__",
]
