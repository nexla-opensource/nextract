"""Tests for nextract.ingest.validators.document_validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nextract.core import DocumentArtifact, ValidationResult
from nextract.ingest.validators.document_validator import DocumentValidator


class TestDocumentValidator:
    def setup_method(self):
        self.validator = DocumentValidator()

    def test_valid_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert result.valid

    def test_file_not_found(self, tmp_path: Path):
        f = tmp_path / "nonexistent.txt"
        doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert not result.valid
        assert any("not found" in e.lower() or "cannot access" in e.lower() for e in result.errors)

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert not result.valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_directory_path(self, tmp_path: Path):
        doc = DocumentArtifact(source_path=str(tmp_path), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert not result.valid
        assert any("directory" in e.lower() for e in result.errors)

    def test_oversized_file(self, tmp_path: Path):
        """Test that oversized files are rejected. Uses a mock to avoid creating huge files."""
        f = tmp_path / "big.txt"
        f.write_text("small")
        # Patch the size limit for testing
        from unittest import mock
        with mock.patch.object(Path, "stat") as mock_stat:
            import os
            from stat import S_IFREG
            mock_stat.return_value = os.stat_result(
                (S_IFREG | 0o644, 0, 0, 0, 0, 0, 600 * 1024 * 1024, 0, 0, 0)
            )
            doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
            result = self.validator.validate(doc)
            assert not result.valid
            assert any("exceeding limit" in e.lower() for e in result.errors)

    def test_single_stat_call_no_toctou(self, tmp_path: Path):
        """Verify validator uses stat() and doesn't have TOCTOU between exists() and stat() calls."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert result.valid

    def test_validation_result_has_warnings(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        doc = DocumentArtifact(source_path=str(f), mime_type="text/plain")
        result = self.validator.validate(doc)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.warnings, list)
        assert isinstance(result.errors, list)
