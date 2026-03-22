from __future__ import annotations

import shutil
from pathlib import Path

import structlog

from nextract.core import DocumentArtifact
from nextract.mimetypes_map import is_textual, is_pdf, is_office_binary
from nextract.pdf_extractor import extract_pdf_text
from nextract.files import _convert_office_to_pdf  # noqa: SLF001

log = structlog.get_logger(__name__)


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_text(document: DocumentArtifact, enable_ocr: bool = True) -> str | None:
    """Extract text from a document artifact when possible."""
    path = Path(document.source_path)

    if document.text is not None:
        return document.text

    if is_textual(path):
        return _read_text_file(path)

    if is_pdf(path):
        try:
            text, _analysis = extract_pdf_text(path, enable_ocr=enable_ocr, include_page_numbers=True)
            return text
        except Exception as exc:  # noqa: BLE001
            log.warning("pdf_text_extraction_failed", file=str(path), error=str(exc))
            return None

    if is_office_binary(path):
        pdf_path = _convert_office_to_pdf(path)
        if pdf_path and pdf_path.exists():
            try:
                text, _analysis = extract_pdf_text(pdf_path, enable_ocr=enable_ocr, include_page_numbers=True)
                return text
            except Exception as exc:  # noqa: BLE001
                log.warning("office_pdf_text_extraction_failed", file=str(path), error=str(exc))
                return None
            finally:
                # Clean up temporary PDF conversion directory
                if pdf_path.parent.name.startswith("nextract-"):
                    shutil.rmtree(pdf_path.parent, ignore_errors=True)

    return None
