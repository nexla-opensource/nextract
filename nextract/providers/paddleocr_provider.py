from __future__ import annotations

import threading
from typing import Any

import structlog

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.providers.ocr_utils import decode_images
from nextract.registry import register_provider

log = structlog.get_logger(__name__)


@register_provider("paddleocr")
class PaddleOCRProvider(BaseProvider):
    """OCR provider backed by PaddleOCR."""

    def __init__(self) -> None:
        self.name: str = "paddleocr"
        self.config: ProviderConfig | None = None
        self._ocr_cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name

    def _get_ocr(self, language: str) -> Any:
        from paddleocr import PaddleOCR

        with self._cache_lock:
            if language not in self._ocr_cache:
                self._ocr_cache[language] = PaddleOCR(lang=language, show_log=False)
            return self._ocr_cache[language]

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "paddleocr required for PaddleOCR. Install with: pip install paddleocr"
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "numpy required for PaddleOCR. Install with: pip install numpy"
            ) from exc

        ocr_dpi = int(request.options.get("ocr_dpi", 300))
        language = request.options.get("language")
        if not language and self.config:
            language = (self.config.extra_params or {}).get("language")
        if not language:
            language = "en"

        ocr = self._get_ocr(language)
        images = decode_images(request.images, ocr_dpi=ocr_dpi)
        if not images:
            log.warning("paddleocr_no_images")
            return ProviderResponse(text="", raw=None)

        text_parts = []
        for image in images:
            result = ocr.ocr(np.array(image))
            if not result:
                continue
            lines = []
            for page in result:
                if page is None:
                    continue
                for line in page:
                    if len(line) >= 2:
                        lines.append(line[1][0])
            text_parts.append(" ".join(lines))

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
