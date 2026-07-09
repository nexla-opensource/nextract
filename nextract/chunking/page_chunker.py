from __future__ import annotations

import shutil
from pathlib import Path

import structlog

from nextract.core import BaseChunker, ChunkerConfig, DocumentArtifact, DocumentChunk, Modality
from nextract.mimetypes_map import guess_mime, is_pdf, is_image, is_audio, is_video, is_office_binary
from nextract.registry import register_chunker

log = structlog.get_logger(__name__)

_OFFICE_PAGE_HINT = (
    "Install LibreOffice (`soffice`) or unoconv and retry, convert the file to PDF first, "
    "or use a text extractor (e.g. extractor='text' with chunker='semantic')."
)


@register_chunker("page")
class PageBasedChunker(BaseChunker):
    """Chunk PDF pages for visual extractors.

    Supports PDFs and images directly. Office documents (DOC/DOCX/PPT/PPTX,
    and other office binaries) are converted to PDF via LibreOffice/unoconv
    when available, then page-chunked.
    """

    @classmethod
    def get_applicable_modalities(cls) -> list[Modality]:
        return [Modality.VISUAL, Modality.HYBRID]

    def validate_config(self, config: ChunkerConfig) -> bool:
        if config.pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be >= 1")
        if config.page_overlap >= config.pages_per_chunk:
            raise ValueError("page_overlap must be < pages_per_chunk")
        return True

    def chunk(self, document: DocumentArtifact, config: ChunkerConfig) -> list[DocumentChunk]:
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

        if is_audio(document_path) or is_video(document_path):
            from nextract.chunking import _media_passthrough_chunk
            return _media_passthrough_chunk(document, document_path)

        if is_office_binary(document_path):
            return self._chunk_office_via_pdf(document, document_path, config)

        if not is_pdf(document_path):
            raise ValueError(
                f"PageBasedChunker only supports PDFs, images, or Office documents "
                f"(got {document_path.suffix!r} for {document_path.name!r}). "
                f"Convert to PDF first or use a text extractor "
                f"(e.g. extractor='text' with chunker='semantic')."
            )

        return self._chunk_pdf(document_path, source_path=path, config=config)

    def _chunk_office_via_pdf(
        self,
        document: DocumentArtifact,
        document_path: Path,
        config: ChunkerConfig,
    ) -> list[DocumentChunk]:
        """Convert Office → PDF then page-chunk; clean up temp conversion artifacts."""
        from nextract.ingest.converters.office import convert_office_to_pdf

        pdf_path, temp_dir = convert_office_to_pdf(document_path)
        if not pdf_path or not pdf_path.exists():
            raise ValueError(
                f"Cannot page-chunk Office document {document_path.name!r}: "
                f"conversion to PDF failed. {_OFFICE_PAGE_HINT}"
            )

        log.info(
            "office_converted_for_page_chunking",
            source=str(document_path),
            pdf=str(pdf_path),
        )
        try:
            return self._chunk_pdf(
                pdf_path,
                source_path=document.source_path,
                config=config,
            )
        finally:
            if temp_dir is not None and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _chunk_pdf(
        self,
        pdf_path: Path,
        *,
        source_path: str,
        config: ChunkerConfig,
    ) -> list[DocumentChunk]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("PyMuPDF required for page chunking. Install with: pip install PyMuPDF") from exc

        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            if total_pages == 0:
                return []

            chunks: list[DocumentChunk] = []
            start_page = 0
            chunk_id = 0

            while start_page < total_pages:
                prev_start = start_page
                end_page = min(start_page + config.pages_per_chunk, total_pages)

                chunk_doc = fitz.open()
                try:
                    chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
                    chunk_bytes = chunk_doc.tobytes()
                finally:
                    chunk_doc.close()

                chunks.append(
                    DocumentChunk(
                        id=f"chunk_{chunk_id}",
                        content=chunk_bytes,
                        source_path=source_path,
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
                if end_page >= total_pages:
                    break
                # Also ensure forward progress
                if start_page <= prev_start:
                    break
                chunk_id += 1
        finally:
            doc.close()

        log.info(
            "pdf_chunked",
            file=source_path,
            total_pages=total_pages,
            num_chunks=len(chunks),
            pages_per_chunk=config.pages_per_chunk,
        )

        return chunks
