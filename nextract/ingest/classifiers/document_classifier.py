from __future__ import annotations

from pathlib import Path

import structlog

from nextract.core import DocumentArtifact, Modality
from nextract.mimetypes_map import is_image, is_pdf, is_textual
from nextract.pdf_analyzer import PDFAnalyzer, PDFType

log = structlog.get_logger(__name__)


class DocumentClassifier:
    """Classify documents into modalities based on file type and content."""

    def __init__(self) -> None:
        self._pdf_analyzer = PDFAnalyzer()

    def classify(self, document: DocumentArtifact) -> Modality:
        path = Path(document.source_path)

        if is_image(path):
            return Modality.VISUAL
        if is_textual(path):
            return Modality.TEXT
        if is_pdf(path):
            try:
                analysis = self._pdf_analyzer.analyze(path)
                if analysis.pdf_type == PDFType.TEXT_RICH:
                    return Modality.TEXT
                if analysis.pdf_type == PDFType.SCANNED:
                    return Modality.VISUAL
                return Modality.HYBRID
            except Exception as exc:  # noqa: BLE001
                log.warning("pdf_classification_failed", file=str(path), error=str(exc))
                return Modality.HYBRID

        return Modality.TEXT
