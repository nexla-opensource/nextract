from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print_json

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.pipeline import ExtractionPipeline

app = typer.Typer(add_completion=False)


def _parse_kv(values: Optional[list[str]]) -> dict[str, str]:
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


@app.command("extract")
def cli_extract(
    document: Path = typer.Argument(..., exists=True, readable=True),
    schema: Path = typer.Option(..., "--schema", "-s", help="Path to JSON Schema file"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    extractor: str = typer.Option("text", "--extractor"),
    provider: str = typer.Option("openai", "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    chunker: str = typer.Option("semantic", "--chunker"),
    pages_per_chunk: int = typer.Option(5, "--pages-per-chunk"),
    page_overlap: int = typer.Option(1, "--page-overlap"),
    chunk_size: int = typer.Option(2000, "--chunk-size"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap"),
    num_passes: int = typer.Option(1, "--num-passes"),
    extractor_params: Optional[list[str]] = typer.Option(None, "--extractor-params"),
    provider_params: Optional[list[str]] = typer.Option(None, "--provider-params"),
    include_extra: bool = typer.Option(False, "--include-extra"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    schema_obj = _load_schema(schema)

    provider_config = ProviderConfig(
        name=provider,
        model=model or "gpt-4o",
        extra_params=_parse_kv(provider_params),
    )
    extractor_config = ExtractorConfig(
        name=extractor,
        provider=provider_config,
        extractor_params=_parse_kv(extractor_params),
    )
    chunker_config = ChunkerConfig(
        name=chunker,
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
    pipeline = ExtractionPipeline(plan)
    result = pipeline.extract(
        document=str(document),
        schema=schema_obj,
        prompt=prompt,
        include_extra=include_extra,
    )

    payload = {"data": result.data, "metadata": result.metadata}
    if output:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_json(data=payload)

