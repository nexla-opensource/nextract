"""
Integration tests for CLI batch command.
"""

import json
import pytest

from typer.testing import CliRunner

from tests.integration.conftest import has_provider_credentials

from nextract.cli import app

runner = CliRunner()


@pytest.mark.integration
class TestBatchCommandHelp:
    """Tests for batch command help."""

    def test_batch_help(self):
        """Batch command should show help."""
        result = runner.invoke(app, ["batch", "--help"])
        
        assert result.exit_code == 0
        assert "--schema" in result.output
        assert "--max-workers" in result.output


@pytest.mark.integration
class TestBatchCommandLive:
    """Live tests for batch command."""

    @pytest.mark.skipif(
        not has_provider_credentials("openai"),
        reason="Missing OpenAI credentials"
    )
    def test_batch_single_document(self, sample_pdf_path, schema_file):
        """Batch extraction with single document."""
        result = runner.invoke(app, [
            "batch", str(sample_pdf_path),
            "--schema", str(schema_file),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
        ])
        
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "results" in output

    @pytest.mark.skipif(
        not has_provider_credentials("openai"),
        reason="Missing OpenAI credentials"
    )
    @pytest.mark.slow
    def test_batch_multiple_documents(self, tmp_path, schema_file, sample_text_content):
        """Batch extraction with multiple documents."""
        docs = []
        for i in range(2):
            doc_path = tmp_path / f"doc_{i}.txt"
            doc_path.write_text(f"Invoice INV-{i:03d}\nTotal: ${100 * (i + 1)}")
            docs.append(str(doc_path))
        
        result = runner.invoke(app, [
            "batch", *docs,
            "--schema", str(schema_file),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--max-workers", "2",
        ])
        
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert len(output["results"]) == 2
