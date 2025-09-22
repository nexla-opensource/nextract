from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional

import typer
from rich import print_json

from .core import extract, batch_extract
from .schema import is_pydantic_model

app = typer.Typer(add_completion=False, help="nextract — structured extraction over Pydantic AI Agent")

def _load_json_schema(path: Path) -> dict:
    return json.loads(path.read_text())

def _load_pydantic_model(dotted: str):
    """Import 'module:ClassName' or 'module.ClassName'."""
    if ":" in dotted:
        mod_name, cls_name = dotted.split(":", 1)
    else:
        mod_name, cls_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    model = getattr(mod, cls_name)
    if not is_pydantic_model(model):
        raise typer.BadParameter(f"{dotted} is not a Pydantic BaseModel subclass")
    return model

@app.command("extract")
def cli_extract(
    files: list[Path] = typer.Argument(..., exists=True, readable=True),
    schema: Optional[Path] = typer.Option(None, "--schema", "-s", help="Path to JSON Schema file"),
    pydantic_model: Optional[str] = typer.Option(None, "--pydantic-model", "-m", help="Dotted path to BaseModel class, e.g. 'pkg.module:MyModel'"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Optional user prompt to guide extraction"),
    examples_file: Optional[Path] = typer.Option(None, "--examples", "-e", help="Optional JSON file with list of examples"),
    include_extra: bool = typer.Option(False, "--include-extra", help="Include `extra` bag for out-of-schema fields (JSON Schema mode)"),
    return_pydantic: bool = typer.Option(False, "--return-pydantic", help="Return Pydantic instance when --pydantic-model is used"),
) -> None:
    """Extract structured data from one or more files (processed together in a single run)."""
    if (schema is None) == (pydantic_model is None):
        raise typer.BadParameter("Provide exactly one of --schema or --pydantic-model")
    examples = None
    if examples_file:
        examples = json.loads(examples_file.read_text())
    if schema:
        schema_obj = _load_json_schema(schema)
        result = extract(
            [str(p) for p in files],
            schema_or_model=schema_obj,
            user_prompt=prompt,
            examples=examples,
            include_extra=include_extra,
            return_pydantic=False,
        )
    else:
        model_type = _load_pydantic_model(pydantic_model)  # type: ignore[arg-type]
        result = extract(
            [str(p) for p in files],
            schema_or_model=model_type,
            user_prompt=prompt,
            examples=examples,
            include_extra=False,  # n/a for model mode
            return_pydantic=return_pydantic,
        )
    print_json(data=result)

@app.command("batch")
def cli_batch(
    files: list[Path] = typer.Argument(..., exists=True, readable=True),
    schema: Optional[Path] = typer.Option(None, "--schema", "-s"),
    pydantic_model: Optional[str] = typer.Option(None, "--pydantic-model", "-m"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    examples_file: Optional[Path] = typer.Option(None, "--examples", "-e"),
    include_extra: bool = typer.Option(False, "--include-extra"),
    return_pydantic: bool = typer.Option(False, "--return-pydantic"),
    max_concurrency: int = typer.Option(4, "--max-concurrency", "-c"),
) -> None:
    """Process each file independently in parallel (one run per file)."""
    if (schema is None) == (pydantic_model is None):
        raise typer.BadParameter("Provide exactly one of --schema or --pydantic-model")
    examples = None
    if examples_file:
        examples = json.loads(examples_file.read_text())
    # Prepare batch as list of single-file groups
    batch = [[str(p)] for p in files]

    if schema:
        schema_obj = json.loads(schema.read_text())
        result = batch_extract(
            batch,
            schema_or_model=schema_obj,
            user_prompt=prompt,
            examples=examples,
            include_extra=include_extra,
            return_pydantic=False,
            max_concurrency=max_concurrency,
        )
    else:
        model_type = _load_pydantic_model(pydantic_model)  # type: ignore[arg-type]
        result = batch_extract(
            batch,
            schema_or_model=model_type,
            user_prompt=prompt,
            examples=examples,
            include_extra=False,
            return_pydantic=return_pydantic,
            max_concurrency=max_concurrency,
        )
    print_json(data=result)
