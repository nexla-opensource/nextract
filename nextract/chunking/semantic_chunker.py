from __future__ import annotations

from typing import List

import structlog

from nextract.core import BaseChunker, CharInterval, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.parse import extract_text
from nextract.registry import register_chunker
from nextract.sentence_chunking import SentenceAwareChunker

log = structlog.get_logger(__name__)


@register_chunker("semantic")
class SemanticChunker(BaseChunker):
    """Semantic chunker for text extractors."""

    @classmethod
    def get_applicable_modalities(cls) -> List[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        if config.chunk_size < config.min_chunk_size:
            raise ValueError(f"chunk_size must be >= {config.min_chunk_size}")
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> List[TextChunk]:
        text = extract_text(document)
        if not text:
            log.warning("semantic_chunker_no_text", file=document.source_path)
            return []

        max_chars = max(config.min_chunk_size, config.chunk_size)
        chunker = SentenceAwareChunker(max_char_buffer=max_chars)

        chunks: List[TextChunk] = []
        for chunk in chunker.chunk_text(text, source_file=document.source_path):
            chunks.append(
                TextChunk(
                    id=f"chunk_{chunk.chunk_id}",
                    text=chunk.text,
                    source_path=document.source_path,
                    metadata={
                        "sentence_indices": chunk.sentence_indices,
                        "char_length": len(chunk.text),
                    },
                    char_interval=CharInterval(
                        start_pos=chunk.char_interval.start_pos,
                        end_pos=chunk.char_interval.end_pos,
                    ),
                )
            )

        return chunks
