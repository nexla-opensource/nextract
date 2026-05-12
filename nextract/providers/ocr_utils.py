from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Any

import structlog

from PIL import Image

log = structlog.get_logger(__name__)


def infer_image_media_type(data: bytes) -> str:
    """Infer image media type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def decode_images(items: list[Any] | None, ocr_dpi: int = 300) -> list[Image.Image]:
    images: list[Image.Image] = []
    if not items:
        return images

    for item in items:
        images.extend(_item_to_images(item, ocr_dpi))

    return images


def _item_to_images(item: Any, ocr_dpi: int) -> list[Image.Image]:
    if isinstance(item, Image.Image):
        return [item]

    if isinstance(item, bytes):
        return _bytes_to_images(item, ocr_dpi)

    if isinstance(item, str):
        try:
            data = base64.b64decode(item)
        except (ValueError, binascii.Error) as exc:
            log.warning("ocr_base64_decode_failed", error=str(exc))
            return []
        return _bytes_to_images(data, ocr_dpi)

    return []


def _bytes_to_images(data: bytes, ocr_dpi: int) -> list[Image.Image]:
    if data[:4] == b"%PDF":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyMuPDF required for OCR PDF processing. Install with: pip install PyMuPDF"
            ) from exc

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            log.warning("ocr_pdf_open_failed", error=str(exc))
            raise ValueError(f"Failed to open PDF for OCR: {exc}") from exc
        try:
            images: list[Image.Image] = []
            for page in doc:
                pix = page.get_pixmap(dpi=ocr_dpi)
                image = Image.open(BytesIO(pix.tobytes("png")))
                images.append(image)
            return images
        finally:
            doc.close()

    try:
        image = Image.open(BytesIO(data))
        image.load()  # Force actual decode to catch corrupt images early
        return [image]
    except Exception as exc:
        log.warning("ocr_image_decode_failed", error=str(exc))
        raise ValueError(f"Failed to decode image for OCR: {exc}") from exc


def encode_image_to_bytes(image: Any) -> bytes | None:
    """Convert an image (bytes, bytearray, or PIL.Image.Image) to raw bytes.

    Shared utility used by VLMExtractor, OCRExtractor, and TextractExtractor
    to avoid duplicating image encoding logic.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Pillow required for image handling. Install with: pip install pillow"
        ) from exc

    if isinstance(image, PILImage.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return None
