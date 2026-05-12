from __future__ import annotations

import base64
from typing import Any

import structlog

from nextract.core import BaseExtractor, ChunkExtraction, ExtractorConfig, ExtractorResult, Modality, ProviderRequest, ProviderUsage
from nextract.core.exceptions import ExtractorError
from nextract.extractors.fallback_mixin import OCRFallbackMixin
from nextract.registry import register_extractor

log = structlog.get_logger(__name__)


@register_extractor("ocr")
class OCRExtractor(OCRFallbackMixin, BaseExtractor):
    """OCR extractor for OCR-first workflows."""

    SUPPORTED_PROVIDERS = ["tesseract", "easyocr", "paddleocr"]

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
        # OCR providers bypass the LLM model capability table.
        # Instead, check the provider class directly.
        from nextract.registry import ProviderRegistry

        provider_name = config.provider.name

        # Check if the provider is a known OCR provider
        if provider_name in self.SUPPORTED_PROVIDERS:
            provider_class = ProviderRegistry.get_instance().get(provider_name)
            if provider_class is not None:
                provider_instance = provider_class()
                if provider_instance.supports_vision():
                    return True
            raise ValueError(
                f"OCR provider '{provider_name}' does not support vision"
            )

        # For non-OCR providers, fall back to model capability table
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
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        ocr_dpi = int(self.config.extractor_params.get("ocr_dpi", 300))
        language = self.config.extractor_params.get("language")

        results: list[ChunkExtraction] = []

        for idx, chunk in enumerate(input_data):
            images_b64 = self._chunk_to_images_b64(chunk)
            metadata = getattr(chunk, "metadata", {})
            response = None
            text = ""

            if images_b64:
                request = ProviderRequest(
                    messages=[],
                    images=images_b64,
                    schema=schema,
                    options={
                        "ocr_dpi": ocr_dpi,
                        "language": language,
                        **kwargs,
                    },
                )
                response = self._safe_generate(provider, request)
                text = response.text
            else:
                if hasattr(chunk, "text"):
                    text = getattr(chunk, "text", "")
                elif hasattr(chunk, "content") and isinstance(chunk.content, str):
                    text = chunk.content

            payload = response.structured_output if response and response.structured_output else {"text": text}
            usage = None
            if response and response.usage:
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
            name="ocr",
            provider_name=provider.get_name(),
            results=results,
            metadata={"modality": "visual", "num_chunks": len(input_data)},
        )

    def _chunk_to_images_b64(self, chunk: Any) -> list[str]:
        images_b64: list[str] = []

        if hasattr(chunk, "images"):
            for image in getattr(chunk, "images", []):
                images_b64.append(self._encode_image(image))
            return [img for img in images_b64 if img]

        if hasattr(chunk, "content") and isinstance(chunk.content, (bytes, bytearray)):
            encoded = base64.b64encode(bytes(chunk.content)).decode("ascii")
            return [encoded]

        return []

    def _encode_image(self, image: Any) -> str:
        if isinstance(image, (bytes, bytearray)):
            return base64.b64encode(bytes(image)).decode("ascii")

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pillow required for OCR image handling. Install with: pip install pillow"
            ) from exc

        if isinstance(image, Image.Image):
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        raise ExtractorError(
            f"Unsupported image type for OCR encoding: {type(image).__name__}. "
            f"Expected bytes, bytearray, or PIL.Image.Image."
        )
