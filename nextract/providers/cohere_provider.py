from __future__ import annotations

from typing import Any

import structlog

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("cohere")
class CohereProvider(PydanticAIProvider):
    """Cohere provider with credential wiring."""

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.cohere import CohereModel
            from pydantic_ai.providers.cohere import CohereProvider as PydanticCohereProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai Cohere support not available. "
                "Install with: pip install pydantic-ai[cohere]"
            ) from exc

        api_key = self.config.api_key if self.config else None

        provider_kwargs: dict[str, Any] = {}
        if api_key:
            provider_kwargs["api_key"] = api_key

        pydantic_provider = PydanticCohereProvider(**provider_kwargs)
        model_name = self.config.model if self.config else ""
        return CohereModel(model_name, provider=pydantic_provider)
