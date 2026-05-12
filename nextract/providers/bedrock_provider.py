from __future__ import annotations

import os
from typing import Any

import structlog

from nextract.core import ProviderConfig
from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("bedrock")
class BedrockProvider(PydanticAIProvider):
    """AWS Bedrock provider with credential wiring.

    Uses Pydantic AI's BedrockConverseModel with the 'bedrock:' prefix.
    Credentials are sourced from ProviderConfig or AWS environment variables.
    """

    def build_model(self) -> Any:
        try:
            from pydantic_ai.models.bedrock import BedrockConverseModel
            from pydantic_ai.providers.bedrock import BedrockProvider as PydanticBedrockProvider
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai Bedrock support not available. "
                "Install with: pip install pydantic-ai[bedrock]"
            ) from exc

        if not self.config:
            raise ValueError("Provider not initialized")

        provider_kwargs: dict[str, Any] = {}

        # Wire credentials from config or environment
        access_key = self.config.api_key or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = self.config.extra_params.get("aws_secret_key") if self.config.extra_params else None
        secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = self.config.extra_params.get("aws_session_token") if self.config.extra_params else None
        session_token = session_token or os.environ.get("AWS_SESSION_TOKEN")
        region = self.config.extra_params.get("region") if self.config.extra_params else None
        region = region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")

        if access_key:
            provider_kwargs["aws_access_key_id"] = access_key
        if secret_key:
            provider_kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            provider_kwargs["aws_session_token"] = session_token
        if region:
            provider_kwargs["region_name"] = region
        if self.config.api_base:
            provider_kwargs["base_url"] = self.config.api_base

        pydantic_provider = PydanticBedrockProvider(**provider_kwargs)
        return BedrockConverseModel(self.config.model, provider=pydantic_provider)

    def get_required_env_vars(self) -> list[str]:
        return ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


# Keep 'aws' as a legacy alias pointing to BedrockProvider
@register_provider("aws", overwrite=True)
class AWSProvider(BedrockProvider):
    """Legacy AWS provider alias. Prefer 'bedrock' for LLM, 'textract' for OCR.

    .. deprecated::
        Use ``ProviderConfig(name="bedrock", ...)`` for Bedrock LLM access.
        Use ``ProviderConfig(name="textract", ...)`` for Textract OCR.
    """

    def initialize(self, config: ProviderConfig) -> None:
        import warnings
        warnings.warn(
            "Provider name 'aws' is deprecated. Use 'bedrock' for Bedrock LLM "
            "or 'textract' for AWS Textract OCR.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().initialize(config)
