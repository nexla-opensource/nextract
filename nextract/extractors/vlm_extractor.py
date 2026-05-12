from __future__ import annotations

from typing import Any

import structlog
from pydantic_ai import BinaryContent

from nextract.core import BaseExtractor, ChunkExtraction, ExtractorConfig, ExtractorResult, Modality, ProviderRequest, ProviderUsage
from nextract.extractors.fallback_mixin import FallbackMixin
from nextract.prompts import build_examples_block, combine_system_prompt
from nextract.registry import register_extractor

log = structlog.get_logger(__name__)


@register_extractor("vlm")
class VLMExtractor(FallbackMixin, BaseExtractor):
    """Extract from images using vision-language models."""

    SUPPORTED_PROVIDERS = [
        "openai",
        "anthropic",
        "google",
        "azure",
        "local",
        "bedrock",
    ]

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.VISUAL

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
        input_data: list[Any],
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
        results: list[ChunkExtraction] = []

        for idx, chunk in enumerate(input_data):
            binary_parts: list[BinaryContent] = []
            metadata: dict[str, Any] = {}

            if hasattr(chunk, "content"):
                data = chunk.content
                if isinstance(data, bytes):
                    media_type = (chunk.metadata or {}).get("media_type") if hasattr(chunk, "metadata") else None
                    binary_parts.append(
                        BinaryContent(data=data, media_type=media_type or "application/octet-stream")
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
                    payload = response.structured_output if response.structured_output is not None else response.text
                    usage = ProviderUsage.from_dict(response.usage) if isinstance(response.usage, dict) else response.usage
                    results.append(
                        ChunkExtraction(
                            chunk_id=getattr(chunk, "id", f"chunk_{idx}"),
                            response=payload,
                            metadata=metadata,
                            usage=usage,
                        )
                    )
                    continue
                metadata = getattr(chunk, "metadata", {})
            elif hasattr(chunk, "images"):
                for image in chunk.images:
                    image_bytes, media_type = self._normalize_image(image)
                    binary_parts.append(
                        BinaryContent(data=image_bytes, media_type=media_type)
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
            payload = response.structured_output if response.structured_output is not None else response.text
            usage = ProviderUsage.from_dict(response.usage) if isinstance(response.usage, dict) else response.usage

            results.append(
                ChunkExtraction(
                    chunk_id=getattr(chunk, "id", f"chunk_{idx}"),
                    response=payload,
                    metadata=metadata,
                    usage=usage,
                )
            )

        return ExtractorResult(
            name="vlm",
            provider_name=provider.get_name(),
            results=results,
            metadata={"modality": "visual", "num_chunks": len(input_data)},
        )

    @staticmethod
    def _normalize_image(image: Any) -> tuple[bytes, str]:
        """Normalize image to bytes and infer media type.

        Handles bytes, bytearray, and PIL.Image.Image objects.
        Non-bytes images (e.g., PIL images) are converted to PNG bytes.
        """
        from nextract.providers.ocr_utils import encode_image_to_bytes, infer_image_media_type

        data = encode_image_to_bytes(image)
        if data is None:
            raise TypeError(
                f"Unsupported image type for VLM: {type(image).__name__}. "
                f"Expected bytes, bytearray, or PIL.Image.Image."
            )
        if isinstance(image, (bytes, bytearray)):
            return data, infer_image_media_type(data)
        # PIL images are converted to PNG
        return data, "image/png"
