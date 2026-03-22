from __future__ import annotations

from typing import Any

import structlog

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality, ProviderRequest, TextChunk
from nextract.extractors.fallback_mixin import FallbackMixin
from nextract.prompts import build_examples_block, combine_system_prompt
from nextract.registry import register_extractor

log = structlog.get_logger(__name__)


@register_extractor("text")
class TextExtractor(FallbackMixin, BaseExtractor):
    """Extract from text-only inputs."""

    SUPPORTED_PROVIDERS = [
        "openai",
        "anthropic",
        "google",
        "azure",
        "local",
        "cohere",
    ]

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.TEXT

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        return True

    def run(
        self,
        input_data: list[TextChunk],
        provider: Any,
        prompt: str,
        schema: dict[str, Any] | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
        **kwargs: Any,
    ) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        examples_block = build_examples_block(examples)
        system_prompt = combine_system_prompt(prompt, include_extra, examples_block)
        results: list[dict[str, Any]] = []

        for idx, chunk in enumerate(input_data):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": chunk.text},
                    ],
                },
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
                    "chunk_id": chunk.id if hasattr(chunk, "id") else f"chunk_{idx}",
                    "response": payload,
                    "metadata": chunk.metadata,
                    "usage": response.usage,
                }
            )

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="text",
            provider_name=provider_name.name if provider_name else "unknown",
            results=results,
            metadata={"modality": "text", "num_chunks": len(input_data)},
        )
