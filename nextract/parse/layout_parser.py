from __future__ import annotations

import structlog

from nextract.core import DocumentArtifact

log = structlog.get_logger(__name__)


class LayoutParser:
    """Placeholder layout parser for document structure analysis."""

    def parse(self, document: DocumentArtifact) -> dict:
        log.warning("layout_parser_not_implemented", file=getattr(document, 'source_path', 'unknown'))
        raise NotImplementedError("LayoutParser is not yet implemented. Use TextParser or OCRParser instead.")
