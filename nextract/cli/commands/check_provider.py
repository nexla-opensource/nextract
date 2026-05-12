from __future__ import annotations

import typer

from nextract.config import get_default_model_for_provider
from nextract.core import ProviderConfig
from nextract.registry import ExtractorRegistry, ProviderRegistry
from nextract.registry.bootstrap import ensure_plugins_loaded

ensure_plugins_loaded()

app = typer.Typer(add_completion=False)


@app.command("check-provider")
def check_provider(
    provider: str = typer.Argument(...),
    model: str = typer.Option(None, "--model", help="Model name (defaults to provider's default)"),
    smoke: bool = typer.Option(False, "--smoke", help="Perform a minimal test request to verify connectivity"),
) -> None:
    model = model or get_default_model_for_provider(provider)

    try:
        provider_class = ProviderRegistry.get_instance().get(provider)
        if not provider_class:
            raise typer.BadParameter(f"Unknown provider: {provider}")

        instance = provider_class()
        config = ProviderConfig(name=provider, model=model)
        instance.initialize(config)
        capabilities = instance.get_capabilities()

        # Resolve the Pydantic AI model ID
        resolved_model_id = ""
        if hasattr(instance, "_resolve_model_id"):
            resolved_model_id = instance._resolve_model_id()
        elif hasattr(instance, "_model_id"):
            resolved_model_id = instance._model_id()

        # Get required env vars
        required_env = instance.get_required_env_vars() if hasattr(instance, "get_required_env_vars") else []

        compatible = []
        for name in ExtractorRegistry.get_instance().list_extractors():
            extractor_class = ExtractorRegistry.get_instance().get(name)
            if extractor_class and provider in extractor_class.get_supported_providers():
                compatible.append(name)

        typer.echo(f"Provider: {provider}")
        typer.echo(f"Model: {model}")
        if resolved_model_id:
            typer.echo(f"Resolved model ID: {resolved_model_id}")
        typer.echo(f"Supports vision: {'Yes' if capabilities.get('vision') else 'No'}")
        typer.echo(
            f"Supports structured output: {'Yes' if capabilities.get('structured_output') else 'No'}"
        )
        typer.echo(f"Max tokens: {capabilities.get('max_tokens')}")
        if required_env:
            import os
            missing_env = [k for k in required_env if not os.getenv(k)]
            typer.echo(f"Required env vars: {', '.join(required_env)}")
            if missing_env:
                typer.echo(f"  Missing: {', '.join(missing_env)}", err=True)
        typer.echo(f"Compatible extractors: {', '.join(compatible) if compatible else 'None'}")

        if smoke:
            typer.echo("\n--- Smoke test ---")
            try:
                from nextract.core import ProviderRequest

                request = ProviderRequest(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Respond with the word 'ok'."},
                    ],
                    schema=None,
                    options={},
                )
                response = instance.generate(request)
                typer.echo(f"Smoke test passed. Response: {response.text[:100]}")
            except Exception as exc:
                typer.echo(f"Smoke test failed: {exc}", err=True)
                raise typer.Exit(code=1)

    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Provider check failed: {exc}", err=True)
        raise typer.Exit(code=1)
