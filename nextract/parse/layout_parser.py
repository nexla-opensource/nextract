from __future__ import annotations

from nextract.core import DocumentArtifact


class LayoutParser:
    """Placeholder layout parser for document structure analysis."""

    def parse(self, document: DocumentArtifact) -> dict:
        return {"source_path": document.source_path, "layout": "unimplemented"}
