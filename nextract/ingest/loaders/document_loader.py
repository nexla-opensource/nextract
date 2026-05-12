from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nextract.core import DocumentArtifact
from nextract.mimetypes_map import guess_mime


def load_documents(paths: Iterable[str | Path], *, allowed_dirs: list[Path] | None = None) -> list[DocumentArtifact]:
    """Load document metadata into artifacts without eager parsing.

    Args:
        paths: Document file paths.
        allowed_dirs: Optional list of directories that documents must reside in.
            If provided, paths outside these directories raise ValueError.
    """
    artifacts: list[DocumentArtifact] = []
    for path in paths:
        file_path = Path(path).expanduser().resolve()

        # Path traversal protection: ensure resolved path doesn't escape allowed dirs
        if allowed_dirs:
            resolved_allowed = [d.resolve() for d in allowed_dirs]
            if not any(str(file_path).startswith(str(d)) for d in resolved_allowed):
                raise ValueError(
                    f"Path '{file_path}' is outside allowed directories. "
                    f"Allowed: {[str(d) for d in resolved_allowed]}"
                )

        mime_type = guess_mime(file_path)
        artifacts.append(
            DocumentArtifact(
                source_path=str(file_path),
                mime_type=mime_type,
                metadata={"filename": file_path.name},
            )
        )
    return artifacts
