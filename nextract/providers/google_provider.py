from __future__ import annotations

from typing import Any

import structlog

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("google")
class GoogleProvider(PydanticAIProvider):
    """Google provider with credential wiring.

    By default uses Google GLA (Gemini API). Set extra_params.backend='google-vertex'
    to use Vertex AI instead.
    """

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google_gla import GoogleGLAProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai Google support not available. "
                "Install with: pip install pydantic-ai[google]"
            ) from exc

        api_key = self.config.api_key if self.config else None
        backend = (self.config.extra_params or {}).get("backend", "google-gla") if self.config else "google-gla"

        provider_kwargs: dict[str, Any] = {}
        if api_key:
            provider_kwargs["api_key"] = api_key

        if backend == "google-vertex":
            try:
                from pydantic_ai.providers.google_vertex import GoogleVertexProvider
                pydantic_provider = GoogleVertexProvider(**provider_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "pydantic-ai Google Vertex support not available. "
                    "Install with: pip install pydantic-ai[google-vertex]"
                ) from exc
        else:
            pydantic_provider = GoogleGLAProvider(**provider_kwargs)

        model_name = self.config.model if self.config else ""
        return GoogleModel(model_name, provider=pydantic_provider)
