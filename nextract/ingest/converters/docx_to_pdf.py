from __future__ import annotations

from pathlib import Path

from nextract.ingest.converters.office import convert_office_to_pdf


def convert_docx_to_pdf(path: str | Path) -> tuple[Path | None, Path | None]:
    """Convert a Word document to PDF using available system tools.

    Returns:
        A tuple of (pdf_path, temp_dir) where temp_dir is the parent directory
        of pdf_path that should be cleaned up after use, or (None, None) on failure.
    """
    return convert_office_to_pdf(path)
