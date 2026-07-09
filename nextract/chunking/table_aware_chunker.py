from __future__ import annotations

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.registry import register_chunker
from nextract.chunking.semantic_chunker import SemanticChunker

log = structlog.get_logger(__name__)


@register_chunker("table_aware")
class TableAwareChunker(BaseChunker):
    """Experimental/stub table-preserving chunker.

    Currently delegates entirely to :class:`SemanticChunker` (sentence packing).
    Table-boundary detection is **not** implemented; ``preserve_tables`` on
    :class:`~nextract.core.config.ChunkerConfig` has no effect.

    Listed as ``[experimental/stub]`` in CLI ``nextract list chunkers``.
    Prefer ``semantic`` or ``fixed_size`` until table boundaries are real.
    """

    _warned: bool = False

    def __init__(self) -> None:
        self._fallback = SemanticChunker()

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        return self._fallback.validate_config(config)

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[TextChunk]:
        if not TableAwareChunker._warned:
            log.warning(
                "table_aware_chunker_experimental",
                message=(
                    "TableAwareChunker is experimental/stub and currently identical to "
                    "semantic chunking; table-boundary detection is not implemented."
                ),
            )
            TableAwareChunker._warned = True
        return self._fallback.chunk(document, config)
