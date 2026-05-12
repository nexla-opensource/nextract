from __future__ import annotations

from typing import Any

import structlog

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("local")
class LocalProvider(PydanticAIProvider):
    """Local provider using Ollama with credential wiring.

    By default uses Pydantic AI's OllamaProvider which connects to
    OLLAMA_BASE_URL (default: http://localhost:11434).
    """

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.ollama import OllamaProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai Ollama support not available. "
                "Install with: pip install pydantic-ai[ollama]"
            ) from exc

        api_base = self.config.api_base if self.config else None

        provider_kwargs: dict[str, Any] = {}
        if api_base:
            provider_kwargs["base_url"] = api_base

        pydantic_provider = OllamaProvider(**provider_kwargs)
        model_name = self.config.model if self.config else ""
        return OpenAIChatModel(model_name, provider=pydantic_provider)
