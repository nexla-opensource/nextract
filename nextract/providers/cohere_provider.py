from __future__ import annotations

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider


@register_provider("cohere")
class CohereProvider(PydanticAIProvider):
    """Cohere provider wrapper."""

    pass
