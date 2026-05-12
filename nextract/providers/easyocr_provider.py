from __future__ import annotations

import threading
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
        self.name: str = "easyocr"
        self.config: ProviderConfig | None = None
        self._reader_cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name

    def _get_reader(self, languages: tuple[str, ...]) -> Any:
        import easyocr

        gpu = False
        if self.config and self.config.extra_params:
            gpu = self.config.extra_params.get("gpu", False)

        key = ",".join(languages)
        with self._cache_lock:
            if key not in self._reader_cache:
                self._reader_cache[key] = easyocr.Reader(list(languages), gpu=gpu)
            return self._reader_cache[key]

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            import easyocr  # noqa: F401
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

        reader = self._get_reader(tuple(languages))
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
