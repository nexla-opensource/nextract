from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, List, Optional

import structlog

from PIL import Image

log = structlog.get_logger(__name__)


def decode_images(items: Optional[List[Any]], ocr_dpi: int = 300) -> List[Image.Image]:
    images: List[Image.Image] = []
    if not items:
        return images

    for item in items:
        images.extend(_item_to_images(item, ocr_dpi))

    return images


def _item_to_images(item: Any, ocr_dpi: int) -> List[Image.Image]:
    if isinstance(item, Image.Image):
        return [item]

    if isinstance(item, bytes):
        return _bytes_to_images(item, ocr_dpi)

    if isinstance(item, str):
        try:
            data = base64.b64decode(item)
        except Exception as exc:  # noqa: BLE001
            log.warning("ocr_base64_decode_failed", error=str(exc))
            return []
        return _bytes_to_images(data, ocr_dpi)

    return []


def _bytes_to_images(data: bytes, ocr_dpi: int) -> List[Image.Image]:
    if data[:4] == b"%PDF":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyMuPDF required for OCR PDF processing. Install with: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(stream=data, filetype="pdf")
        images: List[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(dpi=ocr_dpi)
            image = Image.open(BytesIO(pix.tobytes("png")))
            images.append(image)
        doc.close()
        return images

    try:
        image = Image.open(BytesIO(data))
        return [image]
    except Exception as exc:  # noqa: BLE001
        log.warning("ocr_image_decode_failed", error=str(exc))
        return []
