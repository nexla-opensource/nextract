from __future__ import annotations

from typing import Any

import structlog

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("openai")
class OpenAIProvider(PydanticAIProvider):
    """OpenAI provider with credential wiring."""

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai OpenAI support not available. "
                "Install with: pip install pydantic-ai[openai]"
            ) from exc

        api_key = self.config.api_key if self.config else None
        api_base = self.config.api_base if self.config else None

        provider_kwargs: dict[str, Any] = {}
        if api_key:
            provider_kwargs["api_key"] = api_key
        if api_base:
            provider_kwargs["base_url"] = api_base

        pydantic_provider = PydanticOpenAIProvider(**provider_kwargs)
        model_name = self.config.model if self.config else ""
        return OpenAIChatModel(model_name, provider=pydantic_provider)
