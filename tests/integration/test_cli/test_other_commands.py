"""
Integration tests for CLI commands: check-provider, convert, validate-config.
"""

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

    def test_check_provider_unknown(self):
        """Unknown provider should fail."""
        result = runner.invoke(app, ["check-provider", "unknown_provider"])
        
        assert result.exit_code != 0


@pytest.mark.integration
class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_help(self):
        """Convert should show help."""
        result = runner.invoke(app, ["convert", "--help"])
        
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_convert_to_markdown(self, sample_pdf_path):
        """Convert document to markdown."""
        result = runner.invoke(app, [
            "convert", str(sample_pdf_path),
            "--format", "markdown",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_html(self, sample_pdf_path):
        """Convert document to HTML."""
        result = runner.invoke(app, [
            "convert", str(sample_pdf_path),
            "--format", "html",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_json(self, sample_pdf_path):
        """Convert document to JSON."""
        result = runner.invoke(app, [
            "convert", str(sample_pdf_path),
            "--format", "json",
        ])
        
        assert result.exit_code == 0

    def test_convert_to_output_file(self, sample_pdf_path, tmp_path):
        """Convert document to output file."""
        output_file = tmp_path / "output.md"
        
        result = runner.invoke(app, [
            "convert", str(sample_pdf_path),
            "--format", "markdown",
            "--output", str(output_file),
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()


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
        """Invalid config should fail."""
        result = runner.invoke(app, ["validate-config", str(invalid_plan_config_file)])
        
        assert "invalid" in result.output.lower() or "does not support" in result.output.lower()

    def test_validate_missing_file(self):
        """Missing config file should fail."""
        result = runner.invoke(app, ["validate-config", "nonexistent.json"])
        
        assert result.exit_code != 0
