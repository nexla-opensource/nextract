from __future__ import annotations

from typing import Any

import structlog

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.providers.ocr_utils import decode_images
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("easyocr")
class EasyOCRProvider(BaseProvider):
    """OCR provider backed by EasyOCR."""

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "easyocr required for EasyOCR. Install with: pip install easyocr"
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "numpy required for EasyOCR. Install with: pip install numpy"
            ) from exc

        ocr_dpi = int(request.options.get("ocr_dpi", 300))
        languages = request.options.get("languages")
        if not languages and self.config:
            languages = (self.config.extra_params or {}).get("languages")
        if not languages:
            languages = ["en"]

        reader = easyocr.Reader(list(languages), gpu=False)
        images = decode_images(request.images, ocr_dpi=ocr_dpi)
        if not images:
            log.warning("easyocr_no_images")
            return ProviderResponse(text="", raw=None)

        text_parts = []
        for image in images:
            result = reader.readtext(np.array(image), detail=0)
            text_parts.append(" ".join(result))

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
