from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from nextract.config import get_default_model_for_provider
from nextract.core import ProviderConfig
from nextract.schema import SchemaGenerator

app = typer.Typer(add_completion=False)


@app.command("suggest-schema")
def cli_suggest_schema(
    samples: list[Path] = typer.Argument(..., exists=True, readable=True),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    provider: str = typer.Option("openai", "--provider"),
    model: str | None = typer.Option(None, "--model"),
    examples: Path | None = typer.Option(None, "--examples"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    try:
        if model is None:
            model = get_default_model_for_provider(provider)
        provider_config = ProviderConfig(name=provider, model=model)
        generator = SchemaGenerator(provider=provider_config)

        examples_data = None
        if examples:
            examples_data = json.loads(examples.read_text())

        schema = generator.suggest_schema(
            sample_documents=[str(sample) for sample in samples],
            prompt=prompt,
            examples=examples_data,
        )

        if output:
            generator.save_schema(schema, output)
        else:
            print_json(data=schema)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Schema suggestion failed: {exc}", err=True)
        raise typer.Exit(code=1)
