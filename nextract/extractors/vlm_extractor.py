from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from pydantic_ai import BinaryContent

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality, ProviderRequest
from nextract.prompts import build_examples_block, combine_system_prompt
from nextract.registry import ProviderRegistry, register_extractor

log = structlog.get_logger(__name__)


@register_extractor("vlm")
class VLMExtractor(BaseExtractor):
    """Extract from images using vision-language models."""

    SUPPORTED_PROVIDERS = [
        "openai",
        "anthropic",
        "google",
        "azure",
        "local",
    ]

    def __init__(self) -> None:
        self.config: Optional[ExtractorConfig] = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.VISUAL

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        provider_class = ProviderRegistry.get_instance().get(config.provider.name)
        if provider_class:
            provider = provider_class()
            provider.initialize(config.provider)
            if not provider.supports_vision():
                raise ValueError(
                    f"Provider '{config.provider.name}' does not support vision"
                )
        return True

    def run(
        self,
        input_data: List[Any],
        provider: Any,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        include_extra: bool = False,
        **kwargs: Any,
    ) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        examples_block = build_examples_block(examples)
        system_prompt = combine_system_prompt(prompt, include_extra, examples_block)
        results: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(input_data):
            binary_parts: List[BinaryContent] = []
            metadata: Dict[str, Any] = {}

            if hasattr(chunk, "content"):
                data = chunk.content
                if isinstance(data, bytes):
                    media_type = chunk.metadata.get("media_type") if hasattr(chunk, "metadata") else None
                    binary_parts.append(
                        BinaryContent(data=data, media_type=media_type or "application/pdf")
                    )
                else:
                    messages_text = str(data)
                    metadata = getattr(chunk, "metadata", {})
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [{"type": "text", "text": messages_text}]},
                    ]
                    request = ProviderRequest(
                        messages=messages,
                        schema=schema,
                        options={
                            "include_extra": include_extra,
                            **kwargs,
                        },
                    )
                    response = self._safe_generate(provider, request)
                    payload = response.structured_output or response.text
                    results.append(
                        {
                            "chunk_id": getattr(chunk, "id", f"chunk_{idx}"),
                            "response": payload,
                            "metadata": metadata,
                            "usage": response.usage,
                        }
                    )
                    continue
                metadata = getattr(chunk, "metadata", {})
            elif hasattr(chunk, "images"):
                for image in chunk.images:
                    binary_parts.append(
                        BinaryContent(data=image, media_type="image/png")
                    )
                metadata = getattr(chunk, "metadata", {})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]

            request = ProviderRequest(
                messages=messages,
                schema=schema,
                options={
                    "include_extra": include_extra,
                    "binary_parts": binary_parts,
                    **kwargs,
                },
            )

            response = self._safe_generate(provider, request)
            payload = response.structured_output or response.text

            results.append(
                {
                    "chunk_id": getattr(chunk, "id", f"chunk_{idx}"),
                    "response": payload,
                    "metadata": metadata,
                    "usage": response.usage,
                }
            )

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="vlm",
            provider_name=provider_name.name if provider_name else "unknown",
            results=results,
            metadata={"modality": "visual", "num_chunks": len(input_data)},
        )

    def _safe_generate(self, provider: Any, request: ProviderRequest):
        try:
            return provider.generate(request)
        except Exception as exc:  # noqa: BLE001
            if self.config and self.config.fallback_provider:
                fallback_class = ProviderRegistry.get_instance().get(
                    self.config.fallback_provider.name
                )
                if not fallback_class:
                    raise
                fallback = fallback_class()
                fallback.initialize(self.config.fallback_provider)
                log.warning(
                    "vlm_fallback_provider", error=str(exc), provider=self.config.fallback_provider.name
                )
                return fallback.generate(request)
            raise
