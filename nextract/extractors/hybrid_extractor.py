from __future__ import annotations

import re
from typing import Any

from nextract.core import BaseExtractor, ChunkExtraction, ExtractorConfig, ExtractorResult, Modality
from nextract.extractors.text_extractor import TextExtractor
from nextract.extractors.vlm_extractor import VLMExtractor
from nextract.registry import register_extractor


@register_extractor("hybrid")
class HybridExtractor(BaseExtractor):
    """Hybrid extractor for multi-modality workflows."""

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "azure", "local", "bedrock"]

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
        from nextract.core.model_capabilities import get_model_capability
        has_vision = get_model_capability(
            model=config.provider.model,
            capability="vision",
            default=False,
            provider=config.provider.name,
        )
        if not has_vision:
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
            # Prefer hybrid_source metadata from HybridChunker; fall back to types.
            metadata = getattr(chunk, "metadata", None) or {}
            hybrid_source = metadata.get("hybrid_source") if hasattr(metadata, "get") else None
            if hybrid_source == "text":
                text_chunks.append(chunk)
            elif hybrid_source == "visual":
                visual_chunks.append(chunk)
            elif hasattr(chunk, "text") or getattr(chunk, "modality", None) == Modality.TEXT:
                text_chunks.append(chunk)
            else:
                visual_chunks.append(chunk)

        results: list[ChunkExtraction] = []

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

        return ExtractorResult(
            name="hybrid",
            provider_name=provider.get_name(),
            results=results,
            metadata={
                "modality": "hybrid",
                "num_chunks": len(input_data),
                "visual_chunks": len(visual_chunks),
                "text_chunks": len(text_chunks),
            },
        )

    @staticmethod
    def _result_sort_key(result: ChunkExtraction) -> tuple[int, int, str]:
        """Stable sort key for ChunkExtraction results (not dicts)."""
        metadata = result.metadata or {}
        hybrid_order = metadata.get("hybrid_order") if hasattr(metadata, "get") else None
        if isinstance(hybrid_order, int):
            return (0, hybrid_order, str(result.chunk_id))
        # Extract numeric index from chunk_id (e.g. "chunk_3" -> 3) for stable ordering
        chunk_id = str(result.chunk_id)
        m = re.search(r"\d+", chunk_id)
        numeric_idx = int(m.group()) if m else 0
        return (1, numeric_idx, chunk_id)
