from __future__ import annotations

import typer

from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry
from nextract.registry.bootstrap import ensure_plugins_loaded

ensure_plugins_loaded()

app = typer.Typer(add_completion=False)


@app.command("extractors")
def list_extractors() -> None:
    registry = ExtractorRegistry.get_instance()
    extractors = registry.list_extractors()
    typer.echo("Available extractors:")
    for name in extractors:
        typer.echo(f"- {name}")


@app.command("chunkers")
def list_chunkers(extractor: str = typer.Option(..., "--extractor")) -> None:
    extractor_class = ExtractorRegistry.get_instance().get(extractor)
    if not extractor_class:
        raise typer.BadParameter(f"Unknown extractor: {extractor}")

    modality = extractor_class.get_modality()
    chunkers = ChunkerRegistry.get_instance().get_chunkers_for_modality(modality)

    typer.echo(f"Available chunkers for '{extractor}' extractor ({modality.value} modality):")
    for name in chunkers:
        typer.echo(f"- {name}")


@app.command("providers")
def list_providers() -> None:
    registry = ProviderRegistry.get_instance()
    providers = registry.list_providers()
    typer.echo("Available providers:")
    for name in providers:
        typer.echo(f"- {name}")
