from __future__ import annotations

from typing import Any

import structlog

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.providers.ocr_utils import decode_images
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("tesseract")
class TesseractProvider(BaseProvider):
    """OCR provider backed by Tesseract."""

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pytesseract required for Tesseract OCR. Install with: pip install pytesseract"
            ) from exc

        ocr_dpi = int(request.options.get("ocr_dpi", 300))
        language = request.options.get("language")
        if not language and self.config:
            language = (self.config.extra_params or {}).get("language")

        images = decode_images(request.images, ocr_dpi=ocr_dpi)
        if not images:
            log.warning("tesseract_no_images")
            return ProviderResponse(text="", raw=None)

        text_parts = []
        for image in images:
            if language:
                text_parts.append(pytesseract.image_to_string(image, lang=language))
            else:
                text_parts.append(pytesseract.image_to_string(image))

        return ProviderResponse(text="\n\n".join(text_parts), raw=None)

    def supports_vision(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return False

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "vision": True,
            "structured_output": False,
            "ocr": True,
        }
