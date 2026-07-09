from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from nextract.cli.helpers import default_chunker_for_extractor
from nextract.config import get_default_model_for_provider
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.pipeline import BatchPipeline

app = typer.Typer(add_completion=False, help="Extract structured data from multiple documents")


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def _is_failed_result(value) -> bool:
    """Return True if a batch ExtractionResult represents a failure.

    BatchPipeline records failures as ``metadata["error"]`` with ``data=None``.
    """
    metadata = value.metadata or {}
    return "error" in metadata


@app.command("batch", help="Extract structured data from multiple documents in parallel")
def cli_batch(
    documents: list[Path] = typer.Argument(
        ..., exists=True, readable=True, help="Input document paths"
    ),
    schema: Path = typer.Option(
        ...,
        "--schema",
        "-s",
        exists=True,
        readable=True,
        help="Path to JSON Schema file defining the extraction structure",
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Optional extraction prompt / instructions"
    ),
    extractor: str = typer.Option(
        "auto",
        "--extractor",
        help=(
            "Extraction technique: auto (default; PDF/images→vlm, text files→text), "
            "text, vlm, ocr, hybrid, textract, llamaindex. "
            "For auto, all documents must resolve to the same mode."
        ),
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        help="LLM/API backend: openai, anthropic, google, azure, bedrock, local, …",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model id for the provider (defaults from provider if omitted)",
    ),
    chunker: str | None = typer.Option(
        None,
        "--chunker",
        help="Chunker name (default: auto from extractor modality — page/semantic/hybrid)",
    ),
    pages_per_chunk: int = typer.Option(5, "--pages-per-chunk", help="Pages per visual chunk"),
    page_overlap: int = typer.Option(1, "--page-overlap", help="Page overlap for visual chunks"),
    chunk_size: int = typer.Option(2000, "--chunk-size", help="Text chunk size"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Text chunk overlap"),
    max_workers: int = typer.Option(4, "--max-workers", help="Max parallel workers"),
    include_extra: bool = typer.Option(
        False, "--include-extra", help="Include extra unmapped fields"
    ),
    num_passes: int = typer.Option(
        1,
        "--num-passes",
        help=(
            "Re-run extraction N times and merge (array dedupe or partial-output merge). "
            "Does not use MultiPassExtractor strategies (union/majority/…)."
        ),
    ),
    enable_suggestions: bool = typer.Option(
        False, "--enable-suggestions", help="Emit post-batch improvement suggestions"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON to file instead of stdout"
    ),
) -> None:
    try:
        schema_obj = _load_schema(schema)

        resolved_extractor = extractor
        if extractor.lower().strip() == "auto":
            from nextract import _detect_extraction_mode, _mode_to_extractor_chunker

            modes = {_detect_extraction_mode(str(doc)) for doc in documents}
            if len(modes) > 1:
                raise typer.BadParameter(
                    "Batch --extractor=auto requires all documents to share the same "
                    f"auto mode; got modes {sorted(modes)}. "
                    "Pass --extractor explicitly for mixed batches."
                )
            mode = next(iter(modes)) if modes else "text"
            resolved_extractor, auto_chunker = _mode_to_extractor_chunker(mode, provider)
            resolved_chunker = chunker or auto_chunker
        else:
            resolved_chunker = chunker or default_chunker_for_extractor(resolved_extractor)

        if model is None:
            model = get_default_model_for_provider(provider)
        provider_config = ProviderConfig(name=provider, model=model)
        extractor_config = ExtractorConfig(name=resolved_extractor, provider=provider_config)
        chunker_config = ChunkerConfig(
            name=resolved_chunker,
            pages_per_chunk=pages_per_chunk,
            page_overlap=page_overlap,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        plan = ExtractionPlan(
            extractor=extractor_config,
            chunker=chunker_config,
            num_passes=num_passes,
        )
        # Validate plan before credential warnings so pure config errors surface first.
        from nextract.validate import PlanValidator

        PlanValidator.raise_for_invalid(plan)

        from nextract.credentials import warn_if_missing_credentials

        warn_if_missing_credentials(provider, api_key=provider_config.api_key)

        batch_pipeline = BatchPipeline(
            plan=plan,
            max_workers=max_workers,
            enable_suggestions=enable_suggestions,
        )
        batch_result = batch_pipeline.extract_batch(
            documents=[str(doc) for doc in documents],
            schema=schema_obj,
            prompt=prompt,
            include_extra=include_extra,
        )

        payload = {
            "results": {
                key: {"data": value.data, "metadata": value.metadata}
                for key, value in batch_result.results.items()
            },
            "suggestions": [
                {"description": suggestion.description, "impact": suggestion.impact}
                for suggestion in batch_result.suggestions
            ],
        }
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            typer.echo(f"Wrote {output}", err=True)
        else:
            print_json(data=payload)

        total = len(batch_result.results)
        fail_count = sum(1 for value in batch_result.results.values() if _is_failed_result(value))
        ok_count = total - fail_count
        typer.echo(f"Batch complete: {ok_count} ok, {fail_count} failed ({total} total)", err=True)

        if fail_count > 0:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Batch extraction failed: {exc}", err=True)
        raise typer.Exit(code=1)
