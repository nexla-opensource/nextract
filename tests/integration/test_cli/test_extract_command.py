"""
Integration tests for CLI extract command.
"""

import json
import pytest

from typer.testing import CliRunner

from tests.integration.conftest import has_provider_credentials

from nextract.cli import app

runner = CliRunner()


@pytest.mark.integration
class TestExtractCommandHelp:
    """Tests for extract command help and basic functionality."""

    def test_extract_help(self):
        """Extract command should show help."""
        result = runner.invoke(app, ["extract", "--help"])
        
        assert result.exit_code == 0
        assert "--schema" in result.output
        assert "--provider" in result.output
        assert "--extractor" in result.output

    def test_extract_missing_document(self, schema_file):
        """Missing document should fail."""
        result = runner.invoke(app, [
            "extract", "nonexistent.pdf",
            "--schema", str(schema_file),
        ])
        
        assert result.exit_code != 0

    def test_extract_missing_schema(self, sample_pdf_path):
        """Missing schema should fail."""
        result = runner.invoke(app, [
            "extract", str(sample_pdf_path),
            "--schema", "nonexistent.json",
        ])
        
        assert result.exit_code != 0


@pytest.mark.integration
class TestExtractCommandLive:
    """Live tests for extract command."""

    @pytest.mark.skipif(
        not has_provider_credentials("openai"),
        reason="Missing OpenAI credentials"
    )
    def test_extract_basic(self, sample_pdf_path, schema_file):
        """Basic extraction should work."""
        result = runner.invoke(app, [
            "extract", str(sample_pdf_path),
            "--schema", str(schema_file),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
        ])
        
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "data" in output

    @pytest.mark.skipif(
        not has_provider_credentials("openai"),
        reason="Missing OpenAI credentials"
    )
    def test_extract_with_prompt(self, sample_pdf_path, schema_file):
        """Extraction with custom prompt."""
        result = runner.invoke(app, [
            "extract", str(sample_pdf_path),
            "--schema", str(schema_file),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--prompt", "Extract invoice details carefully",
        ])
        
        assert result.exit_code == 0

    @pytest.mark.skipif(
        not has_provider_credentials("openai"),
        reason="Missing OpenAI credentials"
    )
    def test_extract_to_output_file(self, sample_pdf_path, schema_file, tmp_path):
        """Extraction to output file."""
        output_file = tmp_path / "output.json"
        
        result = runner.invoke(app, [
            "extract", str(sample_pdf_path),
            "--schema", str(schema_file),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--output", str(output_file),
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        output = json.loads(output_file.read_text())
        assert "data" in output
