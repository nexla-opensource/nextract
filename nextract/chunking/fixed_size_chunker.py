from __future__ import annotations

from typing import List

import structlog

from nextract.core import BaseChunker, CharInterval, ChunkerConfig, DocumentArtifact, Modality, TextChunk
from nextract.parse import extract_text
from nextract.registry import register_chunker

log = structlog.get_logger(__name__)


@register_chunker("fixed_size")
class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker for text extractors."""

    @classmethod
    def get_applicable_modalities(cls) -> List[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        if config.chunk_size < config.min_chunk_size:
            raise ValueError(f"chunk_size must be >= {config.min_chunk_size}")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> List[TextChunk]:
        text = extract_text(document)
        if not text:
            log.warning("fixed_size_chunker_no_text", file=document.source_path)
            return []

        chunk_size = max(config.min_chunk_size, config.chunk_size)
        overlap = min(config.chunk_overlap, max(0, chunk_size - 1))
        chunks: List[TextChunk] = []

        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                TextChunk(
                    id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    source_path=document.source_path,
                    metadata={"char_length": len(chunk_text)},
                    char_interval=CharInterval(start_pos=start, end_pos=end),
                )
            )
            chunk_id += 1
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break

        return chunks
