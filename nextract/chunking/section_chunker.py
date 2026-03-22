from __future__ import annotations

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.registry import register_chunker
from nextract.chunking.semantic_chunker import SemanticChunker

log = structlog.get_logger(__name__)


@register_chunker("section")
class SectionChunker(BaseChunker):
    """Section-based chunker using semantic boundaries as a baseline."""

    def __init__(self) -> None:
        self._fallback = SemanticChunker()

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        return self._fallback.validate_config(config)

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[TextChunk]:
        log.warning("section_chunker_not_implemented", msg="SectionChunker delegates to SemanticChunker; section-aware chunking is not yet implemented")
        return self._fallback.chunk(document, config)
