from __future__ import annotations

import typer

import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig
from nextract.registry import ExtractorRegistry, ProviderRegistry

app = typer.Typer(add_completion=False)


@app.command("check-provider")
def check_provider(
    provider: str = typer.Argument(...),
    model: str = typer.Option("gpt-4o", "--model"),
) -> None:
    provider_class = ProviderRegistry.get_instance().get(provider)
    if not provider_class:
        raise typer.BadParameter(f"Unknown provider: {provider}")

    instance = provider_class()
    instance.initialize(ProviderConfig(name=provider, model=model))
    capabilities = instance.get_capabilities()

    compatible = []
    for name in ExtractorRegistry.get_instance().list_extractors():
        extractor_class = ExtractorRegistry.get_instance().get(name)
        if extractor_class and provider in extractor_class.get_supported_providers():
            compatible.append(name)

    typer.echo(f"Provider: {provider}")
    typer.echo(f"Supports vision: {'Yes' if capabilities.get('vision') else 'No'}")
    typer.echo(
        f"Supports structured output: {'Yes' if capabilities.get('structured_output') else 'No'}"
    )
    typer.echo(f"Max tokens: {capabilities.get('max_tokens')}")
    typer.echo(f"Compatible extractors: {', '.join(compatible) if compatible else 'None'}")
