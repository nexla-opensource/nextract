from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from nextract.core import DocumentArtifact
from nextract.mimetypes_map import guess_mime


def load_documents(paths: Iterable[str | Path]) -> List[DocumentArtifact]:
    """Load document metadata into artifacts without eager parsing."""
    artifacts: List[DocumentArtifact] = []
    for path in paths:
        file_path = Path(path).expanduser().resolve()
        mime_type = guess_mime(file_path)
        artifacts.append(
            DocumentArtifact(
                source_path=str(file_path),
                mime_type=mime_type,
                metadata={"filename": file_path.name},
            )
        )
    return artifacts
