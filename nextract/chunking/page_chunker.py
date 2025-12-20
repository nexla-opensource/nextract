from __future__ import annotations

from pathlib import Path
from typing import List

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, DocumentChunk, Modality
from nextract.mimetypes_map import guess_mime, is_pdf, is_image
from nextract.registry import register_chunker

log = structlog.get_logger(__name__)


@register_chunker("page")
class PageBasedChunker(BaseChunker):
    """Chunk PDF pages for visual extractors."""

    @classmethod
    def get_applicable_modalities(cls) -> List[Modality]:
        return [Modality.VISUAL, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        if config.pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be >= 1")
        if config.page_overlap >= config.pages_per_chunk:
            raise ValueError("page_overlap must be < pages_per_chunk")
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> List[DocumentChunk]:
        path = document.source_path
        document_path = Path(path)

        if is_image(document_path):
            data = document_path.read_bytes()
            return [
                DocumentChunk(
                    id="chunk_0",
                    content=data,
                    source_path=path,
                    modality=Modality.VISUAL,
                    metadata={
                        "page_range": (1, 1),
                        "total_pages": 1,
                        "pages_in_chunk": 1,
                        "media_type": guess_mime(document_path),
                    },
                )
            ]

        if not is_pdf(document_path):
            raise ValueError("PageBasedChunker only supports PDFs or images")

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("PyMuPDF required for page chunking. Install with: pip install PyMuPDF") from exc

        doc = fitz.open(document_path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return []

        chunks: List[DocumentChunk] = []
        start_page = 0
        chunk_id = 0

        while start_page < total_pages:
            end_page = min(start_page + config.pages_per_chunk, total_pages)

            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            chunks.append(
                DocumentChunk(
                    id=f"chunk_{chunk_id}",
                    content=chunk_bytes,
                    source_path=path,
                    modality=Modality.VISUAL,
                    metadata={
                        "page_range": (start_page + 1, end_page),
                        "total_pages": total_pages,
                        "pages_in_chunk": end_page - start_page,
                        "overlap_with_previous": config.page_overlap if chunk_id > 0 else 0,
                        "media_type": "application/pdf",
                    },
                )
            )

            start_page = end_page - config.page_overlap
            chunk_id += 1

        doc.close()

        log.info(
            "pdf_chunked",
            file=path,
            total_pages=total_pages,
            num_chunks=len(chunks),
            pages_per_chunk=config.pages_per_chunk,
        )

        return chunks
