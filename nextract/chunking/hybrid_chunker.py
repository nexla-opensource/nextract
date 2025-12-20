from __future__ import annotations

from pathlib import Path
from typing import List

from nextract.chunking.page_chunker import PageBasedChunker
from nextract.chunking.semantic_chunker import SemanticChunker
from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, DocumentChunk, Modality, TextChunk
from nextract.mimetypes_map import is_pdf, is_image
from nextract.registry import register_chunker


@register_chunker("hybrid")
class HybridChunker(BaseChunker):
    """Hybrid chunker that routes based on document modality."""

    def __init__(self) -> None:
        self._page_chunker = PageBasedChunker()
        self._semantic_chunker = SemanticChunker()

    @classmethod
    def get_applicable_modalities(cls) -> List[Modality]:
        return [Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> List[DocumentChunk | TextChunk]:
        path = Path(document.source_path)
        if is_pdf(path) or is_image(path):
            return self._page_chunker.chunk(document, config)
        return self._semantic_chunker.chunk(document, config)
