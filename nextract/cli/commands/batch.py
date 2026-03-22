from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from nextract.config import get_default_model_for_provider
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.pipeline import BatchPipeline

app = typer.Typer(add_completion=False)


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


@app.command("batch")
def cli_batch(
    documents: list[Path] = typer.Argument(..., exists=True, readable=True),
    schema: Path = typer.Option(..., "--schema", "-s"),
    prompt: str | None = typer.Option(None, "--prompt", "-p"),
    extractor: str = typer.Option("text", "--extractor"),
    provider: str = typer.Option("openai", "--provider"),
    model: str | None = typer.Option(None, "--model"),
    chunker: str = typer.Option("semantic", "--chunker"),
    max_workers: int = typer.Option(4, "--max-workers"),
    include_extra: bool = typer.Option(False, "--include-extra"),
    num_passes: int = typer.Option(1, "--num-passes"),
    enable_suggestions: bool = typer.Option(False, "--enable-suggestions"),
) -> None:
    try:
        schema_obj = _load_schema(schema)

        if model is None:
            model = get_default_model_for_provider(provider)
        provider_config = ProviderConfig(name=provider, model=model)
        extractor_config = ExtractorConfig(name=extractor, provider=provider_config)
        chunker_config = ChunkerConfig(name=chunker)

        plan = ExtractionPlan(
            extractor=extractor_config,
            chunker=chunker_config,
            num_passes=num_passes,
        )
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
        print_json(data=payload)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Batch extraction failed: {exc}", err=True)
        raise typer.Exit(code=1)
