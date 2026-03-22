from __future__ import annotations

from typing import Any

import structlog

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality
from nextract.extractors.text_extractor import TextExtractor
from nextract.registry import register_extractor

log = structlog.get_logger(__name__)


@register_extractor("llamaindex")
class LlamaIndexExtractor(BaseExtractor):
    """LlamaIndex extractor with TextExtractor fallback."""

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "local"]

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None
        self._text = TextExtractor()

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)
        self._text.initialize(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.TEXT

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        required_params = ["index_path", "retriever_mode"]
        for param in required_params:
            if param not in config.extractor_params:
                raise ValueError(f"LlamaIndex requires '{param}' in extractor_params")
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

        enable_llamaindex = bool(self.config.extractor_params.get("enable_llamaindex", True))
        if not enable_llamaindex:
            log.info("llamaindex_disabled_fallback")
            return self._text.run(
                input_data,
                provider=provider,
                prompt=prompt,
                schema=schema,
                examples=examples,
                include_extra=include_extra,
                **kwargs,
            )

        try:
            from llama_index.core import Document, VectorStoreIndex
        except ImportError:
            log.warning("llamaindex_missing_fallback")
            return self._text.run(
                input_data,
                provider=provider,
                prompt=prompt,
                schema=schema,
                examples=examples,
                include_extra=include_extra,
                **kwargs,
            )

        documents = []
        for chunk in input_data:
            if hasattr(chunk, "text") and isinstance(chunk.text, str):
                documents.append(Document(text=chunk.text))
            elif hasattr(chunk, "content") and isinstance(chunk.content, str):
                documents.append(Document(text=chunk.content))

        if not documents:
            return ExtractorResult(
                name="llamaindex",
                provider_name=getattr(provider, "config", None).name if getattr(provider, "config", None) else "unknown",
                results=[],
                metadata={"modality": "text", "num_chunks": len(input_data), "llamaindex": True},
            )

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)
        payload: dict[str, Any] = {"response": str(response)}

        result = {
            "chunk_id": "llamaindex",
            "response": payload,
            "metadata": {"llamaindex": True},
            "usage": None,
        }

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="llamaindex",
            provider_name=provider_name.name if provider_name else "unknown",
            results=[result],
            metadata={"modality": "text", "num_chunks": len(input_data), "llamaindex": True},
        )
