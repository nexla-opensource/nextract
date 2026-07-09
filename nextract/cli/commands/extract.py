from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from nextract.cli.helpers import default_chunker_for_extractor
from nextract.config import get_default_model_for_provider
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.pipeline import ExtractionPipeline

app = typer.Typer(add_completion=False, help="Extract structured data from a document")


def _parse_kv(values: list[str] | None) -> dict[str, str]:
    params: dict[str, str] = {}
    if not values:
        return params
    for item in values:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid param '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        params[key] = value
    return params


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


@app.command("extract", help="Extract structured data from a document using a JSON schema")
def cli_extract(
    document: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to the input document"
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
            "text, vlm, ocr, hybrid, textract, llamaindex"
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
    num_passes: int = typer.Option(
        1,
        "--num-passes",
        help=(
            "Re-run extraction N times and merge (array dedupe or partial-output merge). "
            "Does not use MultiPassExtractor strategies (union/majority/…)."
        ),
    ),
    extractor_params: list[str] | None = typer.Option(
        None, "--extractor-params", help="Extractor params as key=value"
    ),
    provider_params: list[str] | None = typer.Option(
        None, "--provider-params", help="Provider params as key=value"
    ),
    include_extra: bool = typer.Option(
        False, "--include-extra", help="Include extra unmapped fields"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON to file instead of stdout"
    ),
) -> None:
    try:
        schema_obj = _load_schema(schema)

        resolved_extractor = extractor
        if extractor.lower().strip() == "auto":
            # Match extract_simple: PDF/images → visual (vlm), plain text → text.
            from nextract import _detect_extraction_mode, _mode_to_extractor_chunker

            mode = _detect_extraction_mode(str(document))
            resolved_extractor, auto_chunker = _mode_to_extractor_chunker(mode, provider)
            resolved_chunker = chunker or auto_chunker
        else:
            resolved_chunker = chunker or default_chunker_for_extractor(resolved_extractor)

        provider_config = ProviderConfig(
            name=provider,
            model=model or get_default_model_for_provider(provider),
            extra_params=_parse_kv(provider_params),
        )
        extractor_config = ExtractorConfig(
            name=resolved_extractor,
            provider=provider_config,
            extractor_params=_parse_kv(extractor_params),
        )
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
        # Validate plan before credential warnings so pure config errors
        # (unknown extractor, num_passes, modality mismatch) are not
        # preceded by missing-API-key noise.
        pipeline = ExtractionPipeline(plan)

        from nextract.credentials import warn_if_missing_credentials

        warn_if_missing_credentials(provider, api_key=provider_config.api_key)

        result = pipeline.extract(
            document=str(document),
            schema=schema_obj,
            prompt=prompt,
            include_extra=include_extra,
        )

        payload = {"data": result.data, "metadata": result.metadata}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            typer.echo(f"Wrote {output}", err=True)
        else:
            print_json(data=payload)
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Extraction failed: {exc}", err=True)
        raise typer.Exit(code=1)
