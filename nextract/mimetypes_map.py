from __future__ import annotations

import mimetypes
from pathlib import Path

# Extend standard guesses
_CUSTOM: dict[str, str] = {
    ".md": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".json": "application/json",
    ".xml": "application/xml",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
}

_TEXTUAL_EXTS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html", ".htm",
    ".xls", ".xlsx",
}

_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"
}

_PDF_EXTS = {".pdf"}

_ZIP_EXTS = {".zip"}

_UNSUPPORTED_BUT_ACCEPTABLE_AS_BINARY = {
    # Office documents that we convert to PDF before sending to LLMs
    ".doc", ".docx", ".ppt", ".pptx",
}

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _CUSTOM:
        return _CUSTOM[ext]
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"

def is_textual(path: Path) -> bool:
    return path.suffix.lower() in _TEXTUAL_EXTS

def is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS

def is_pdf(path: Path) -> bool:
    return path.suffix.lower() in _PDF_EXTS

def is_zip(path: Path) -> bool:
    return path.suffix.lower() in _ZIP_EXTS

def is_office_binary(path: Path) -> bool:
    return path.suffix.lower() in _UNSUPPORTED_BUT_ACCEPTABLE_AS_BINARY
