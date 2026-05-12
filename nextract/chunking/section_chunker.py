from __future__ import annotations

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.registry import register_chunker
from nextract.chunking.semantic_chunker import SemanticChunker

log = structlog.get_logger(__name__)


@register_chunker("section")
class SectionChunker(BaseChunker):
    """Section-based chunker — currently delegates to SemanticChunker.

    .. note::
        Section-aware boundary detection is not yet implemented.
        This chunker is functionally identical to ``semantic``.
        A future release will add heading/section-break detection.
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
