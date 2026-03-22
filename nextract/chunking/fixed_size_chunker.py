from __future__ import annotations

from pathlib import Path

import structlog

from nextract.core import BaseChunker, CharInterval, ChunkerConfig, DocumentArtifact, DocumentChunk, Modality, TextChunk
from nextract.mimetypes_map import is_audio, is_video
from nextract.parse import extract_text
from nextract.registry import register_chunker

log = structlog.get_logger(__name__)


@register_chunker("fixed_size")
class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker for text extractors."""

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.TEXT, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        if config.chunk_size < config.min_chunk_size:
            raise ValueError(f"chunk_size must be >= {config.min_chunk_size}")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[TextChunk | DocumentChunk]:
        text = extract_text(document)
        if not text:
            path = Path(document.source_path)
            if is_audio(path) or is_video(path):
                from nextract.chunking import _media_passthrough_chunk
                return _media_passthrough_chunk(document, path)
            log.warning("fixed_size_chunker_no_text", file=document.source_path)
            return []

        chunk_size = max(config.min_chunk_size, config.chunk_size)
        overlap = min(config.chunk_overlap, max(0, chunk_size - 1))
        chunks: list[TextChunk] = []

        step = chunk_size - overlap
        if step < 1:
            step = max(1, chunk_size // 2)

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
            start += step
            if start >= len(text):
                break

        return chunks
