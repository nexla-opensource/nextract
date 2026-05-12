from __future__ import annotations

import os
from typing import Any

import structlog

from nextract.core import ProviderRequest
from nextract.core.exceptions import ProviderCapabilityError
from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider
from pydantic_ai import BinaryContent

log = structlog.get_logger(__name__)


@register_provider("azure")
class AzureProvider(PydanticAIProvider):
    """Azure OpenAI provider with credential wiring.

    Azure Chat Completions does NOT support document binary inputs.
    Use text-only extraction or ensure documents are pre-converted to text.
    """

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.azure import AzureProvider as PydanticAzureProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai Azure support not available. "
                "Install with: pip install pydantic-ai[azure]"
            ) from exc

        if not self.config:
            raise ValueError("Provider not initialized")

        api_key = self.config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = self.config.api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = self.config.extra_params.get("api_version") if self.config.extra_params else None
        api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")

        provider_kwargs: dict[str, Any] = {}
        if api_key:
            provider_kwargs["api_key"] = api_key
        if endpoint:
            provider_kwargs["azure_endpoint"] = endpoint
        if api_version:
            provider_kwargs["api_version"] = api_version

        pydantic_provider = PydanticAzureProvider(**provider_kwargs)
        return OpenAIChatModel(self.config.model, provider=pydantic_provider)

    def get_required_env_vars(self) -> list[str]:
        return ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]

    def _build_parts(self, request: ProviderRequest) -> tuple[str, list[str | BinaryContent]]:
        """Override to guard against unsupported document binary inputs.

        Azure Chat Completions does not support PDF/Office document binary content.
        """
        system_prompt, parts = super()._build_parts(request)

        # Check for unsupported binary content
        for part in parts:
            if isinstance(part, BinaryContent):
                media_type = getattr(part, "media_type", "") or ""
                unsupported_prefixes = ("application/pdf", "application/vnd.")
                if any(media_type.startswith(p) for p in unsupported_prefixes):
                    raise ProviderCapabilityError(
                        f"Azure Chat Completions does not support document binary input "
                        f"(media_type='{media_type}'). Use text-only extraction or convert "
                        f"documents to text before sending to Azure.",
                        provider="azure",
                        model=self.config.model if self.config else "",
                    )

        return system_prompt, parts
