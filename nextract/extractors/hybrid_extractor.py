from __future__ import annotations

from typing import Any

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality
from nextract.extractors.text_extractor import TextExtractor
from nextract.extractors.vlm_extractor import VLMExtractor
from nextract.registry import ProviderRegistry, register_extractor


@register_extractor("hybrid")
class HybridExtractor(BaseExtractor):
    """Hybrid extractor for multi-modality workflows."""

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "azure", "local"]

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None
        self._vlm = VLMExtractor()
        self._text = TextExtractor()

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)
        self._vlm.initialize(config)
        self._text.initialize(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.HYBRID

    @classmethod
    def get_supported_providers(cls) -> list[str]:
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
        input_data: Any,
        provider: Any,
        prompt: str,
        schema: dict[str, Any] | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
        **kwargs: Any,
    ) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        visual_chunks = []
        text_chunks = []

        for chunk in input_data:
            if hasattr(chunk, "text") or getattr(chunk, "modality", None) == Modality.TEXT:
                text_chunks.append(chunk)
            else:
                visual_chunks.append(chunk)

        results: list[dict[str, Any]] = []

        if visual_chunks:
            vlm_result = self._vlm.run(
                visual_chunks,
                provider=provider,
                prompt=prompt,
                schema=schema,
                examples=examples,
                include_extra=include_extra,
                **kwargs,
            )
            results.extend(vlm_result.results)

        if text_chunks:
            text_result = self._text.run(
                text_chunks,
                provider=provider,
                prompt=prompt,
                schema=schema,
                examples=examples,
                include_extra=include_extra,
                **kwargs,
            )
            results.extend(text_result.results)

        # Preserve hybrid chunk ordering when both modalities are present.
        results.sort(key=self._result_sort_key)

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="hybrid",
            provider_name=provider_name.name if provider_name else "unknown",
            results=results,
            metadata={
                "modality": "hybrid",
                "num_chunks": len(input_data),
                "visual_chunks": len(visual_chunks),
                "text_chunks": len(text_chunks),
            },
        )

    @staticmethod
    def _result_sort_key(result: dict[str, Any]) -> tuple[int, int, str]:
        metadata = result.get("metadata") or {}
        hybrid_order = metadata.get("hybrid_order")
        if isinstance(hybrid_order, int):
            return (0, hybrid_order, str(result.get("chunk_id", "")))
        return (1, 0, str(result.get("chunk_id", "")))
