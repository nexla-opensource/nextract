from __future__ import annotations

import typer

from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry
from nextract.registry.bootstrap import ensure_plugins_loaded

ensure_plugins_loaded()

app = typer.Typer(
    add_completion=False,
    help="List available extractors, providers, and chunkers",
)

# Chunkers that currently delegate to semantic (stubs / experimental).
_EXPERIMENTAL_CHUNKERS = frozenset({"section", "table_aware"})


@app.command("extractors", help="List registered extractors with modality and providers")
def list_extractors() -> None:
    registry = ExtractorRegistry.get_instance()
    extractors = registry.list_extractors()
    typer.echo("Available extractors:")
    for name in extractors:
        extractor_cls = registry.get(name)
        if extractor_cls is None:
            typer.echo(f"- {name}")
            continue
        modality = extractor_cls.get_modality().value
        providers = extractor_cls.get_supported_providers()
        providers_str = ", ".join(providers) if providers else "none"
        typer.echo(f"- {name}  modality={modality}  providers=[{providers_str}]")


@app.command("chunkers", help="List chunkers compatible with an extractor")
def list_chunkers(
    extractor: str = typer.Option(..., "--extractor", help="Extractor name to filter chunkers by modality"),
) -> None:
    registry = ExtractorRegistry.get_instance()
    extractor_class = registry.get(extractor)
    if not extractor_class:
        available = ", ".join(registry.list_extractors()) or "(none)"
        raise typer.BadParameter(
            f"Unknown extractor: {extractor}. Available extractors: {available}"
        )

    modality = extractor_class.get_modality()
    chunkers = ChunkerRegistry.get_instance().get_chunkers_for_modality(modality)

    typer.echo(f"Available chunkers for '{extractor}' extractor ({modality.value} modality):")
    for name in chunkers:
        if name in _EXPERIMENTAL_CHUNKERS:
            typer.echo(f"- {name}  [experimental/stub]")
        else:
            typer.echo(f"- {name}")


@app.command("providers", help="List registered providers")
def list_providers() -> None:
    registry = ProviderRegistry.get_instance()
    providers = registry.list_providers()
    typer.echo("Available providers:")
    for name in providers:
        if name == "aws":
            typer.echo(f"- {name}  (deprecated alias for bedrock)")
        else:
            typer.echo(f"- {name}")
