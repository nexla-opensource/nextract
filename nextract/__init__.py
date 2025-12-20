from __future__ import annotations

from typing import Any

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover - fallback for older importlib
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("nextract") if version else "unknown"
except PackageNotFoundError:
    __version__ = "unknown"

from nextract.core import (  # noqa: E402
    ChunkerConfig,
    ExtractionPlan,
    ExtractorConfig,
    Modality,
    ProviderConfig,
)
from nextract.output import CsvFormatter, HtmlFormatter, JsonFormatter, MarkdownFormatter  # noqa: E402
from nextract.pipeline import BatchExtractionResult, BatchPipeline, ExtractionPipeline  # noqa: E402
from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry  # noqa: E402
from nextract.schema import SchemaGenerator, SchemaSuggestion  # noqa: E402


def extract_simple(
    document: str,
    schema: dict,
    provider: str,
    model: str | None = None,
    prompt: str | None = None,
) -> Any:
    """Simplest extraction helper using a default text pipeline."""
    provider_config = ProviderConfig(name=provider, model=model or "gpt-4o")
    extractor_config = ExtractorConfig(name="text", provider=provider_config)
    chunker_config = ChunkerConfig(name="semantic")
    plan = ExtractionPlan(extractor=extractor_config, chunker=chunker_config)

    pipeline = ExtractionPipeline(plan)
    return pipeline.extract(document=document, schema=schema, prompt=prompt)


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
    "BatchPipeline",
    "BatchExtractionResult",
    "ChunkerConfig",
    "CsvFormatter",
    "ExtractionPipeline",
    "ExtractionPlan",
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
    "__version__",
]
