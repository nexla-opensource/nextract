"""
Integration tests for CLI list commands.
"""

import pytest

from typer.testing import CliRunner

from nextract.cli import app

runner = CliRunner()


@pytest.mark.integration
class TestListExtractors:
    """Tests for list extractors command."""

    def test_list_extractors_help(self):
        """List extractors should show help."""
        result = runner.invoke(app, ["list", "extractors", "--help"])
        
        assert result.exit_code == 0

    def test_list_extractors(self):
        """List extractors should show available extractors."""
        result = runner.invoke(app, ["list", "extractors"])
        
        assert result.exit_code == 0
        assert "text" in result.output
        assert "vlm" in result.output


@pytest.mark.integration
class TestListChunkers:
    """Tests for list chunkers command."""

    def test_list_chunkers_help(self):
        """List chunkers should show help."""
        result = runner.invoke(app, ["list", "chunkers", "--help"])
        
        assert result.exit_code == 0
        assert "--extractor" in result.output

    def test_list_chunkers_for_text_extractor(self):
        """List chunkers for text extractor."""
        result = runner.invoke(app, ["list", "chunkers", "--extractor", "text"])
        
        assert result.exit_code == 0
        assert "semantic" in result.output
        assert "text" in result.output.lower()

    def test_list_chunkers_for_vlm_extractor(self):
        """List chunkers for VLM extractor."""
        result = runner.invoke(app, ["list", "chunkers", "--extractor", "vlm"])
        
        assert result.exit_code == 0
        assert "page" in result.output
        assert "visual" in result.output.lower()

    def test_list_chunkers_unknown_extractor(self):
        """Unknown extractor should fail."""
        result = runner.invoke(app, ["list", "chunkers", "--extractor", "unknown"])
        
        assert result.exit_code != 0


@pytest.mark.integration
class TestListProviders:
    """Tests for list providers command."""

    def test_list_providers(self):
        """List providers should show available providers."""
        result = runner.invoke(app, ["list", "providers"])
        
        assert result.exit_code == 0
        assert "openai" in result.output
        assert "anthropic" in result.output
