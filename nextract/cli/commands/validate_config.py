from __future__ import annotations

import json
from pathlib import Path

import typer

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.validate import PlanValidator

app = typer.Typer(add_completion=False)


def _load_plan(path: Path) -> ExtractionPlan:
    data = json.loads(path.read_text())

    provider_cfg = ProviderConfig(**data["extractor"]["provider"])
    extractor_cfg = ExtractorConfig(
        name=data["extractor"]["name"],
        provider=provider_cfg,
        fallback_provider=None,
        extractor_params=data["extractor"].get("extractor_params", {}),
    )
    chunker_cfg = ChunkerConfig(**data["chunker"])
    plan_kwargs = {
        "num_passes": data.get("num_passes", 1),
        "include_confidence": data.get("include_confidence", True),
        "include_citations": data.get("include_citations", True),
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


@app.command("validate-config")
def validate_config(plan_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    plan = _load_plan(plan_path)
    result = PlanValidator.validate_extraction_plan(plan)
    if result.valid:
        typer.echo("Plan is valid")
    else:
        typer.echo("Plan is invalid")
        for error in result.errors:
            typer.echo(f"- {error}")
