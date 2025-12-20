from __future__ import annotations

from pathlib import Path

from nextract.core import DocumentArtifact, ValidationResult


class DocumentValidator:
    """Validate document existence and basic integrity."""

    def validate(self, document: DocumentArtifact) -> ValidationResult:
        path = Path(document.source_path)
        errors: list[str] = []

        if not path.exists():
            errors.append(f"File not found: {path}")
        if path.exists() and path.is_dir():
            errors.append(f"Path is a directory: {path}")
        if path.exists() and path.is_file() and path.stat().st_size == 0:
            errors.append(f"File is empty: {path}")

        return ValidationResult(valid=not errors, errors=errors)
