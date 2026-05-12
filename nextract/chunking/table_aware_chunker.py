from __future__ import annotations

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.registry import register_chunker
from nextract.chunking.semantic_chunker import SemanticChunker

log = structlog.get_logger(__name__)


@register_chunker("table_aware")
class TableAwareChunker(BaseChunker):
    """Chunker that preserves tables when possible — currently delegates to SemanticChunker.

    .. note::
        Table-aware boundary detection is not yet implemented.
        This chunker is functionally identical to ``semantic``.
        A future release will add table-boundary-aware chunking.
    """

    def __init__(self) -> None:
        self._fallback = SemanticChunker()

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        return self._fallback.validate_config(config)

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[TextChunk]:
        return self._fallback.chunk(document, config)
