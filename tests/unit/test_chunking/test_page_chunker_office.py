"""Unit tests for PageBasedChunker Office → PDF path and error messages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nextract.chunking.page_chunker import PageBasedChunker
from nextract.core import ChunkerConfig, DocumentArtifact, DocumentChunk, Modality


def _has_fitz() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _write_minimal_pdf(path: Path, pages: int = 2) -> Path:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def _office_artifact(tmp_path: Path, name: str = "report.docx") -> DocumentArtifact:
    path = tmp_path / name
    path.write_bytes(b"PK\x03\x04fake-office")
    return DocumentArtifact(
        source_path=str(path),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content=path.read_bytes(),
    )


class TestPageChunkerOfficeConversion:
    def test_office_converts_then_chunks(self, tmp_path: Path) -> None:
        if not _has_fitz():
            pytest.skip("PyMuPDF required for end-to-end PDF page chunking")

        pdf_path = _write_minimal_pdf(tmp_path / "converted.pdf", pages=3)
        temp_dir = tmp_path / "nextract-officepdf-report"
        temp_dir.mkdir()
        real_pdf = temp_dir / "report.pdf"
        real_pdf.write_bytes(pdf_path.read_bytes())

        document = _office_artifact(tmp_path, "report.docx")
        chunker = PageBasedChunker()
        config = ChunkerConfig(name="page", pages_per_chunk=2, page_overlap=0)

        with patch(
            "nextract.ingest.converters.office.convert_office_to_pdf",
            return_value=(real_pdf, temp_dir),
        ) as mock_convert:
            chunks = chunker.chunk(document, config)

        mock_convert.assert_called_once()
        assert len(chunks) == 2
        assert all(c.modality == Modality.VISUAL for c in chunks)
        # Provenance keeps original Office path, not temp PDF
        assert all(c.source_path == document.source_path for c in chunks)
        assert chunks[0].metadata["page_range"] == (1, 2)
        assert chunks[1].metadata["page_range"] == (3, 3)
        assert chunks[0].metadata["total_pages"] == 3
        # Temp conversion dir cleaned up
        assert not temp_dir.exists()

    def test_office_conversion_failure_clear_error(self, tmp_path: Path) -> None:
        document = _office_artifact(tmp_path, "slides.pptx")
        chunker = PageBasedChunker()
        config = ChunkerConfig(name="page")

        with patch(
            "nextract.ingest.converters.office.convert_office_to_pdf",
            return_value=(None, None),
        ):
            with pytest.raises(ValueError, match="conversion to PDF failed") as exc_info:
                chunker.chunk(document, config)

        msg = str(exc_info.value)
        assert "slides.pptx" in msg
        assert "LibreOffice" in msg or "soffice" in msg
        assert "text extractor" in msg.lower() or "semantic" in msg

    def test_office_cleanup_on_chunk_error(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "nextract-officepdf-cleanup"
        temp_dir.mkdir()
        real_pdf = temp_dir / "report.pdf"
        real_pdf.write_bytes(b"%PDF-1.4 fake")

        document = _office_artifact(tmp_path, "report.docx")
        chunker = PageBasedChunker()

        with (
            patch(
                "nextract.ingest.converters.office.convert_office_to_pdf",
                return_value=(real_pdf, temp_dir),
            ),
            patch.object(
                PageBasedChunker,
                "_chunk_pdf",
                side_effect=RuntimeError("chunk boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="chunk boom"):
                chunker.chunk(document, ChunkerConfig(name="page"))

        assert not temp_dir.exists()

    def test_unsupported_extension_clear_error(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        document = DocumentArtifact(
            source_path=str(path),
            mime_type="text/plain",
            text="hello",
        )
        chunker = PageBasedChunker()

        with pytest.raises(ValueError, match="only supports PDFs") as exc_info:
            chunker.chunk(document, ChunkerConfig(name="page"))

        msg = str(exc_info.value)
        assert ".txt" in msg
        assert "Convert to PDF" in msg or "text extractor" in msg

    def test_pdf_still_chunks_directly(self, tmp_path: Path) -> None:
        if not _has_fitz():
            pytest.skip("PyMuPDF required for end-to-end PDF page chunking")

        pdf_path = _write_minimal_pdf(tmp_path / "direct.pdf", pages=2)
        document = DocumentArtifact(
            source_path=str(pdf_path),
            mime_type="application/pdf",
        )
        chunker = PageBasedChunker()
        chunks = chunker.chunk(
            document,
            ChunkerConfig(name="page", pages_per_chunk=1, page_overlap=0),
        )
        assert len(chunks) == 2
        assert chunks[0].source_path == str(pdf_path)

    def test_office_path_delegates_to_chunk_pdf(self, tmp_path: Path) -> None:
        """Orchestration test that does not require PyMuPDF at runtime."""
        temp_dir = tmp_path / "nextract-officepdf-mock"
        temp_dir.mkdir()
        real_pdf = temp_dir / "report.pdf"
        real_pdf.write_bytes(b"%PDF-1.4")

        document = _office_artifact(tmp_path, "memo.docx")
        canned = [
            DocumentChunk(
                id="chunk_0",
                content=b"%PDF",
                source_path=document.source_path,
                modality=Modality.VISUAL,
                metadata={"page_range": (1, 1), "total_pages": 1},
            )
        ]
        chunker = PageBasedChunker()

        with (
            patch(
                "nextract.ingest.converters.office.convert_office_to_pdf",
                return_value=(real_pdf, temp_dir),
            ),
            patch.object(PageBasedChunker, "_chunk_pdf", return_value=canned) as mock_pdf,
        ):
            chunks = chunker.chunk(document, ChunkerConfig(name="page"))

        mock_pdf.assert_called_once()
        assert mock_pdf.call_args.kwargs["source_path"] == document.source_path
        assert mock_pdf.call_args.args[0] == real_pdf
        assert chunks == canned
        assert not temp_dir.exists()
