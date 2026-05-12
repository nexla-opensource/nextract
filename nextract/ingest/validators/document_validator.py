from __future__ import annotations

import structlog
from pathlib import Path

from nextract.core import DocumentArtifact, ValidationResult

log = structlog.get_logger(__name__)

_MAX_PDF_PAGES = 5000
_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class DocumentValidator:
    """Validate document existence and basic integrity."""

    def validate(self, document: DocumentArtifact) -> ValidationResult:
        path = Path(document.source_path)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            stat = path.stat()
        except FileNotFoundError:
            log.warning("file_not_found", path=str(path))
            errors.append(f"File not found: {path.name}")
            return ValidationResult(valid=False, errors=errors)
        except OSError as exc:
            errors.append(f"Cannot access file {path.name}: {exc}")
            return ValidationResult(valid=False, errors=errors)

        if stat.st_size == 0:
            log.warning("file_is_empty", path=str(path))
            errors.append(f"File is empty: {path.name}")

        if stat.st_size > _MAX_FILE_SIZE_BYTES:
            errors.append(
                f"File {path.name} is {stat.st_size} bytes, "
                f"exceeding limit of {_MAX_FILE_SIZE_BYTES} bytes"
            )

        if path.is_dir():
            log.warning("path_is_directory", path=str(path))
            errors.append(f"Path is a directory: {path.name}")

        # PDF-specific validation
        if path.suffix.lower() == ".pdf" and path.is_file():
            pdf_errors, pdf_warnings = self._validate_pdf(path)
            errors.extend(pdf_errors)
            warnings.extend(pdf_warnings)

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    def _validate_pdf(self, path: Path) -> tuple[list[str], list[str]]:
        """Validate PDF-specific constraints: page count, encryption."""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            try:
                if doc.is_encrypted:
                    errors.append(f"PDF {path.name} is encrypted/password-protected")
                page_count = doc.page_count
                if page_count > _MAX_PDF_PAGES:
                    errors.append(
                        f"PDF {path.name} has {page_count} pages, "
                        f"exceeding limit of {_MAX_PDF_PAGES}"
                    )
            finally:
                doc.close()
        except ImportError:
            warnings.append("PyMuPDF not available; skipping PDF structure validation")
        except Exception as exc:
            warnings.append(f"Could not validate PDF structure: {exc}")

        return errors, warnings
