from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def convert_image_to_png_bytes(path: str | Path) -> bytes:
    """Convert an image to PNG bytes for downstream processing."""
    file_path = Path(path)
    with Image.open(file_path) as image:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
