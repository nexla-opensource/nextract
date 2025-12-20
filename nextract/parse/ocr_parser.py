from __future__ import annotations

import structlog

from nextract.core import DocumentArtifact
from nextract.pdf_extractor import extract_pdf_text
from nextract.mimetypes_map import is_pdf

log = structlog.get_logger(__name__)


class OCRParser:
    """OCR parser for scanned PDFs or images."""

    def parse(self, document: DocumentArtifact) -> str | None:
        if not is_pdf(__import__("pathlib").Path(document.source_path)):
            log.warning("ocr_parser_non_pdf", file=document.source_path)
            return None

        try:
            text, _analysis = extract_pdf_text(
                document.source_path,
                enable_ocr=True,
                include_page_numbers=True,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            log.warning("ocr_parser_failed", file=document.source_path, error=str(exc))
            return None
