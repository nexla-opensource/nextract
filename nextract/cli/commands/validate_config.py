from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.validate import PlanValidator

app = typer.Typer(add_completion=False, help="Validate an extraction plan configuration")


def _load_provider(data: dict[str, Any] | None) -> ProviderConfig | None:
    if not data or not isinstance(data, dict):
        return None
    return ProviderConfig(**data)


def _load_plan(path: Path) -> ExtractionPlan:
    data = json.loads(path.read_text())

    extractor_data = data.get("extractor")
    if not extractor_data or not isinstance(extractor_data, dict):
        raise KeyError("Missing or invalid 'extractor' section in plan configuration")

    provider_data = extractor_data.get("provider")
    if not provider_data or not isinstance(provider_data, dict):
        raise KeyError("Missing or invalid 'provider' section in extractor configuration")

    chunker_data = data.get("chunker")
    if not chunker_data or not isinstance(chunker_data, dict):
        raise KeyError("Missing or invalid 'chunker' section in plan configuration")

    provider_cfg = ProviderConfig(**provider_data)
    fallback_provider = _load_provider(extractor_data.get("fallback_provider"))

    extractor_kwargs: dict[str, Any] = {
        "name": extractor_data.get("name", "text"),
        "provider": provider_cfg,
        "fallback_provider": fallback_provider,
        "extractor_params": extractor_data.get("extractor_params", {}),
    }
    if "enable_caching" in extractor_data:
        extractor_kwargs["enable_caching"] = extractor_data["enable_caching"]
    if "batch_size" in extractor_data:
        extractor_kwargs["batch_size"] = extractor_data["batch_size"]
    if "modality" in extractor_data:
        extractor_kwargs["modality"] = extractor_data["modality"]

    extractor_cfg = ExtractorConfig(**extractor_kwargs)
    chunker_cfg = ChunkerConfig(**chunker_data)
    plan_kwargs = {
        "num_passes": data.get("num_passes", 1),
        "include_confidence": data.get("include_confidence", True),
        "include_citations": data.get("include_citations", False),
        "include_raw_text": data.get("include_raw_text", False),
        "auto_suggest_schema": data.get("auto_suggest_schema", False),
        "schema_validation": data.get("schema_validation", True),
        "retry_on_failure": data.get("retry_on_failure", True),
        "max_retries": data.get("max_retries", 3),
        "backoff_factor": data.get("backoff_factor", 2.0),
        "validation_rules": data.get("validation_rules", []),
        "strict_validation": data.get("strict_validation", False),
    }
    return ExtractionPlan(extractor=extractor_cfg, chunker=chunker_cfg, **plan_kwargs)


@app.command("validate-config", help="Validate an extraction plan JSON configuration file")
def validate_config(
    plan_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to plan JSON file"),
) -> None:
    try:
        plan = _load_plan(plan_path)
    except (KeyError, TypeError) as exc:
        typer.echo(f"Invalid plan format: {exc}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Failed to load plan: {exc}", err=True)
        raise typer.Exit(code=1)

    result = PlanValidator.validate_extraction_plan(plan)
    if result.valid:
        typer.echo("Plan is valid")
    else:
        typer.echo("Plan is invalid")
        for error in result.errors:
            typer.echo(f"- {error}")
        raise typer.Exit(code=1)
