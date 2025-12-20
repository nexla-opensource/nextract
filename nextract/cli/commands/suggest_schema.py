from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print_json

from nextract.core import ProviderConfig
from nextract.schema import SchemaGenerator

app = typer.Typer(add_completion=False)


@app.command("suggest-schema")
def cli_suggest_schema(
    samples: list[Path] = typer.Argument(..., exists=True, readable=True),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    provider: str = typer.Option("openai", "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    examples: Optional[Path] = typer.Option(None, "--examples"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    provider_config = ProviderConfig(name=provider, model=model or "gpt-4o")
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
