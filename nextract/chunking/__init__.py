"""Chunking layer with modality-aware chunkers."""

from __future__ import annotations

from pathlib import Path

from nextract.core import DocumentArtifact, DocumentChunk, Modality
from nextract.mimetypes_map import guess_mime

from .page_chunker import PageBasedChunker
from .semantic_chunker import SemanticChunker
from .fixed_size_chunker import FixedSizeChunker
from .table_aware_chunker import TableAwareChunker
from .section_chunker import SectionChunker
from .hybrid_chunker import HybridChunker


def _media_passthrough_chunk(document: DocumentArtifact, path: Path) -> list[DocumentChunk]:
    """Create a single passthrough DocumentChunk for audio/video files."""
    data = document.content or path.read_bytes()
    return [
        DocumentChunk(
            id="chunk_0",
            content=data,
            source_path=document.source_path,
            modality=Modality.VISUAL,
            metadata={"media_type": guess_mime(path), "passthrough": True},
        )
    ]


__all__ = [
    "PageBasedChunker",
    "SemanticChunker",
    "FixedSizeChunker",
    "TableAwareChunker",
    "SectionChunker",
    "HybridChunker",
    "_media_passthrough_chunk",
]
