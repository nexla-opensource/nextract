from __future__ import annotations

import structlog
from pathlib import Path

from nextract.core import DocumentArtifact, ValidationResult

log = structlog.get_logger(__name__)


class DocumentValidator:
    """Validate document existence and basic integrity."""

    def validate(self, document: DocumentArtifact) -> ValidationResult:
        path = Path(document.source_path)
        errors: list[str] = []

        if not path.exists():
            log.warning("file_not_found", path=str(path))
            errors.append(f"File not found: {path.name}")
        if path.exists() and path.is_dir():
            log.warning("path_is_directory", path=str(path))
            errors.append(f"Path is a directory: {path.name}")
        if path.exists() and path.is_file() and path.stat().st_size == 0:
            log.warning("file_is_empty", path=str(path))
            errors.append(f"File is empty: {path.name}")

        return ValidationResult(valid=not errors, errors=errors)
