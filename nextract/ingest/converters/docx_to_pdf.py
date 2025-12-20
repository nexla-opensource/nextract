from __future__ import annotations

from pathlib import Path
from typing import Optional

from nextract.files import _convert_office_to_pdf  # noqa: SLF001


def convert_docx_to_pdf(path: str | Path) -> Optional[Path]:
    """Convert a Word document to PDF using available system tools."""
    file_path = Path(path)
    return _convert_office_to_pdf(file_path)
