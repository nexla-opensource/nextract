"""Unit tests for convert CLI helpers and validation wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from nextract.cli import app
from nextract.cli.commands.convert import (
    _empty_text_message,
    _fail_validation,
    _validate_theme,
)
from nextract.core import ValidationResult
from nextract.core.artifacts import DocumentArtifact

runner = CliRunner()


class TestValidateTheme:
    def test_valid_themes(self) -> None:
        assert _validate_theme("light") == "light"
        assert _validate_theme("DARK") == "dark"
        assert _validate_theme("System") == "system"

    def test_invalid_theme_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid theme"):
            _validate_theme("neon")


class TestEmptyTextMessage:
    def test_pdf_mentions_tesseract(self) -> None:
        msg = _empty_text_message(Path("scan.pdf")).lower()
        assert "tesseract" in msg
        assert "extracted text" in msg

    def test_office_mentions_libreoffice(self) -> None:
        msg = _empty_text_message(Path("doc.docx")).lower()
        assert "libreoffice" in msg or "soffice" in msg

    def test_image_mentions_vlm(self) -> None:
        msg = _empty_text_message(Path("photo.png")).lower()
        assert "vlm" in msg
        assert "image" in msg

    def test_textual_empty_does_not_suggest_office_or_ocr(self) -> None:
        """Whitespace/empty .txt should not dump Office/Tesseract/VLM advice."""
        msg = _empty_text_message(Path("notes.txt")).lower()
        assert "empty" in msg or "whitespace" in msg
        assert "libreoffice" not in msg
        assert "soffice" not in msg
        assert "tesseract" not in msg
        assert "vlm" not in msg


class TestFailValidation:
    def test_encrypted_adds_decrypt_guidance(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            _fail_validation(["PDF secret.pdf is encrypted/password-protected"])
        assert exc_info.value.exit_code == 1

    def test_generic_error_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            _fail_validation(["File is empty: foo.txt"])
        assert exc_info.value.exit_code == 1


class TestConvertDocumentValidatorWiring:
    def test_validation_errors_stop_convert(self, tmp_path: Path) -> None:
        doc = tmp_path / "sample.txt"
        doc.write_text("hello")

        artifact = DocumentArtifact(source_path=str(doc), mime_type="text/plain")
        with (
            patch(
                "nextract.cli.commands.convert.load_documents",
                return_value=[artifact],
            ),
            patch(
                "nextract.cli.commands.convert.DocumentValidator.validate",
                return_value=ValidationResult(
                    valid=False,
                    errors=["PDF secret.pdf is encrypted/password-protected"],
                ),
            ),
            patch("nextract.cli.commands.convert.extract_text") as extract_mock,
        ):
            result = runner.invoke(app, ["convert", str(doc), "--format", "markdown"])

        assert result.exit_code == 1
        combined = ((result.output or "") + (getattr(result, "stderr", None) or "")).lower()
        assert "encrypted" in combined or "password" in combined
        assert "decrypt" in combined or "unencrypted" in combined
        extract_mock.assert_not_called()

    def test_invalid_theme_cli(self, tmp_path: Path) -> None:
        doc = tmp_path / "sample.txt"
        doc.write_text("hello")
        result = runner.invoke(
            app,
            ["convert", str(doc), "--format", "html", "--theme", "rainbow"],
        )
        assert result.exit_code in (1, 2)
        combined = ((result.output or "") + (getattr(result, "stderr", None) or "")).lower()
        assert "theme" in combined
