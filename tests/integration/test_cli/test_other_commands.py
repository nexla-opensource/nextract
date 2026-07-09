"""
Integration tests for CLI commands: check-provider, convert, validate-config.
"""

import json

import pytest

from typer.testing import CliRunner


from nextract.cli import app

runner = CliRunner()


@pytest.mark.integration
class TestCheckProviderCommand:
    """Tests for check-provider command."""

    def test_check_provider_help(self):
        """Check-provider should show help."""
        result = runner.invoke(app, ["check-provider", "--help"])
        
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_check_provider_openai(self):
        """Check OpenAI provider capabilities."""
        result = runner.invoke(app, ["check-provider", "openai"])
        
        assert result.exit_code == 0
        assert "Provider: openai" in result.output
        assert "Supports vision:" in result.output
        assert "Supports structured output:" in result.output
        assert "Compatible extractors:" in result.output

    def test_check_provider_anthropic(self):
        """Check Anthropic provider capabilities."""
        result = runner.invoke(app, ["check-provider", "anthropic"])
        
        assert result.exit_code == 0
        assert "Provider: anthropic" in result.output

    def test_check_provider_tesseract_without_model(self):
        """OCR provider tesseract should work without --model (default model)."""
        result = runner.invoke(app, ["check-provider", "tesseract"])

        assert result.exit_code == 0
        assert "Provider: tesseract" in result.output
        assert "Model: default" in result.output

    def test_check_provider_unknown(self):
        """Unknown provider should fail."""
        result = runner.invoke(app, ["check-provider", "unknown_provider"])
        
        assert result.exit_code != 0


@pytest.mark.integration
class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_help(self):
        """Convert should show help and honest description of text formatting."""
        result = runner.invoke(app, ["convert", "--help"])
        
        assert result.exit_code == 0
        assert "--format" in result.output
        help_text = result.output.lower()
        assert "extracted text" in help_text or "formats" in help_text

    def test_convert_to_markdown(self, tmp_path):
        """Convert document to markdown."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001\nTotal: $100")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "markdown",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_html(self, tmp_path):
        """Convert document to HTML."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001\nTotal: $100")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "html",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_json(self, tmp_path):
        """Convert document to JSON."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001\nTotal: $100")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "json",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_output_file(self, tmp_path):
        """Convert document to output file."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001\nTotal: $100")
        output_file = tmp_path / "output.md"
        
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "markdown",
            "--output", str(output_file),
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()

    def test_convert_unknown_format_fails(self, tmp_path):
        """Unknown --format should fail with clear error (no silent JSON fallback)."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "xml",
        ])

        # Typer BadParameter exits 2; Exit(1) also acceptable for scripting trust
        assert result.exit_code in (1, 2)
        combined = (result.output or "") + (result.stderr or "")
        assert "unknown format" in combined.lower() or "xml" in combined.lower()

    def test_convert_empty_text_fails(self, tmp_path):
        """Empty document should fail with clear validation/empty message."""
        empty_doc = tmp_path / "empty.txt"
        empty_doc.write_text("")

        result = runner.invoke(app, [
            "convert", str(empty_doc),
            "--format", "markdown",
        ])

        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "empty" in combined.lower()

    def test_convert_whitespace_only_fails_with_hints(self, tmp_path):
        """Whitespace-only text should fail with empty/whitespace guidance, not install noise."""
        doc = tmp_path / "blank.txt"
        doc.write_text("   \n\t  \n")

        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "markdown",
        ])

        assert result.exit_code == 1
        combined = ((result.output or "") + (result.stderr or "")).lower()
        assert "empty" in combined or "whitespace" in combined
        # Plain text must not suggest Office/OCR/VLM tooling.
        assert "libreoffice" not in combined
        assert "soffice" not in combined
        assert "tesseract" not in combined
        assert "vlm" not in combined

    def test_convert_invalid_theme_fails(self, tmp_path):
        """Invalid --theme should fail with clear error (no silent clamp)."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "html",
            "--theme", "neon",
        ])

        assert result.exit_code in (1, 2)
        combined = ((result.output or "") + (result.stderr or "")).lower()
        assert "theme" in combined
        assert "neon" in combined or "light" in combined or "invalid" in combined

    def test_convert_valid_theme_html(self, tmp_path):
        """Valid HTML theme should succeed and appear in output."""
        doc = tmp_path / "sample.txt"
        doc.write_text("Invoice INV-001")
        result = runner.invoke(app, [
            "convert", str(doc),
            "--format", "html",
            "--theme", "dark",
        ])

        assert result.exit_code == 0
        assert 'data-theme="dark"' in result.output

    def test_convert_validation_rejects_empty_file(self, tmp_path):
        """DocumentValidator on convert path should reject empty files."""
        empty_doc = tmp_path / "zero.bin"
        empty_doc.write_bytes(b"")

        result = runner.invoke(app, [
            "convert", str(empty_doc),
            "--format", "json",
        ])

        assert result.exit_code == 1
        combined = ((result.output or "") + (result.stderr or "")).lower()
        assert "empty" in combined or "validation" in combined

    def test_convert_encrypted_pdf_guidance(self, tmp_path, monkeypatch):
        """Encrypted/password PDF validation should include decrypt guidance."""
        from nextract.core import ValidationResult
        from nextract.core.artifacts import DocumentArtifact

        pdf = tmp_path / "secret.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        def _fake_load(paths, **kwargs):
            return [
                DocumentArtifact(source_path=str(pdf), mime_type="application/pdf"),
            ]

        def _fake_validate(self, document):
            return ValidationResult(
                valid=False,
                errors=[f"PDF {pdf.name} is encrypted/password-protected"],
            )

        monkeypatch.setattr(
            "nextract.cli.commands.convert.load_documents",
            _fake_load,
        )
        monkeypatch.setattr(
            "nextract.ingest.validators.document_validator.DocumentValidator.validate",
            _fake_validate,
        )

        result = runner.invoke(app, [
            "convert", str(pdf),
            "--format", "markdown",
        ])

        assert result.exit_code == 1
        combined = ((result.output or "") + (result.stderr or "")).lower()
        assert "encrypted" in combined or "password" in combined
        assert "decrypt" in combined or "unencrypted" in combined


@pytest.mark.integration
class TestValidateConfigCommand:
    """Tests for validate-config command."""

    def test_validate_config_help(self):
        """Validate-config should show help."""
        result = runner.invoke(app, ["validate-config", "--help"])
        
        assert result.exit_code == 0

    def test_validate_valid_config(self, plan_config_file):
        """Valid config should pass."""
        result = runner.invoke(app, ["validate-config", str(plan_config_file)])
        
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, invalid_plan_config_file):
        """Invalid config should fail with exit code 1."""
        result = runner.invoke(app, ["validate-config", str(invalid_plan_config_file)])
        
        assert result.exit_code == 1
        assert "invalid" in result.output.lower() or "does not support" in result.output.lower()

    def test_validate_preserves_extractor_fields(self, tmp_path):
        """validate-config should load fallback_provider, batch_size, enable_caching."""
        plan = {
            "extractor": {
                "name": "text",
                "provider": {"name": "openai", "model": "gpt-4o"},
                "fallback_provider": {"name": "anthropic", "model": "claude-sonnet-4-20250514"},
                "batch_size": 5,
                "enable_caching": False,
            },
            "chunker": {"name": "semantic", "chunk_size": 2000},
            "num_passes": 1,
        }
        plan_path = tmp_path / "plan_with_fields.json"
        plan_path.write_text(json.dumps(plan, indent=2))

        from nextract.cli.commands.validate_config import _load_plan

        loaded = _load_plan(plan_path)
        assert loaded.extractor.batch_size == 5
        assert loaded.extractor.enable_caching is False
        assert loaded.extractor.fallback_provider is not None
        assert loaded.extractor.fallback_provider.name == "anthropic"

    def test_validate_missing_file(self):
        """Missing config file should fail."""
        result = runner.invoke(app, ["validate-config", "nonexistent.json"])
        
        assert result.exit_code != 0
