from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nextract.chunking.page_chunker import PageBasedChunker
from nextract.chunking.semantic_chunker import SemanticChunker
from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, DocumentChunk, Modality, TextChunk
from nextract.mimetypes_map import is_pdf, is_image
from nextract.registry import register_chunker


@register_chunker("hybrid")
class HybridChunker(BaseChunker):
    """Hybrid chunker that combines visual and text chunks when possible."""

    def __init__(self) -> None:
        self._page_chunker = PageBasedChunker()
        self._semantic_chunker = SemanticChunker()

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        self._page_chunker.validate_config(config)
        self._semantic_chunker.validate_config(config)
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[DocumentChunk | TextChunk]:
        path = Path(document.source_path)
        if is_pdf(path) or is_image(path):
            visual_chunks = self._tag_chunks(
                self._page_chunker.chunk(document, config),
                source="visual",
                priority=0,
            )
            text_chunks = self._tag_chunks(
                self._semantic_chunker.chunk(document, config),
                source="text",
                priority=1,
            )
            return [*visual_chunks, *text_chunks]

        return self._tag_chunks(
            self._semantic_chunker.chunk(document, config),
            source="text",
            priority=0,
        )

    def _tag_chunks(
        self,
        chunks: list[DocumentChunk | TextChunk],
        source: str,
        priority: int,
    ) -> list[DocumentChunk | TextChunk]:
        tagged: list[DocumentChunk | TextChunk] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(getattr(chunk, "metadata", {}))
            metadata["hybrid_source"] = source
            metadata["hybrid_order"] = (index * 2) + priority
            tagged.append(
                replace(
                    chunk,
                    id=f"{chunk.id}_{source}",
                    metadata=metadata,
                )
            )
        return tagged
