from __future__ import annotations

from pathlib import Path

from nextract.files import _convert_office_to_pdf  # noqa: SLF001


def convert_pptx_to_pdf(path: str | Path) -> Path | None:
    """Convert a PowerPoint document to PDF using available system tools."""
    file_path = Path(path)
    return _convert_office_to_pdf(file_path)
