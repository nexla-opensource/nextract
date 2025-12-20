from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import structlog

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality, ProviderRequest
from nextract.registry import ProviderRegistry, register_extractor

log = structlog.get_logger(__name__)


@register_extractor("ocr")
class OCRExtractor(BaseExtractor):
    """OCR extractor for OCR-first workflows."""

    SUPPORTED_PROVIDERS = ["tesseract", "easyocr", "paddleocr"]

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
        input_data: Any,
        provider: Any,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        ocr_dpi = int(self.config.extractor_params.get("ocr_dpi", 300))
        language = self.config.extractor_params.get("language")

        results: List[Dict[str, Any]] = []

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

            results.append(
                {
                    "chunk_id": getattr(chunk, "id", f"chunk_{idx}"),
                    "response": payload,
                    "metadata": metadata,
                    "usage": response.usage if response else None,
                }
            )

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="ocr",
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
                    "ocr_fallback_provider", error=str(exc), provider=self.config.fallback_provider.name
                )
                return fallback.generate(request)
            raise

    def _chunk_to_images_b64(self, chunk: Any) -> List[str]:
        images_b64: List[str] = []

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

        return ""
