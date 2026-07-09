"""
Integration tests for CLI batch command.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from typer.testing import CliRunner

from tests.integration.conftest import has_provider_credentials

from nextract.cli import app
from nextract.core import ExtractionResult
from nextract.pipeline import BatchExtractionResult

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
        assert "--output" in result.output
        assert "--chunker" in result.output
        assert "--pages-per-chunk" in result.output


@pytest.mark.integration
class TestBatchCommandFailureExit:
    """Tests for batch exit codes when some documents fail."""

    def test_batch_exits_1_on_result_errors(self, tmp_path, schema_file):
        """If any result has metadata.error, batch should exit 1 and print summary."""
        doc = tmp_path / "doc.txt"
        doc.write_text("Invoice INV-001\nTotal: $100")

        failed = ExtractionResult(
            data=None,
            metadata={"error": "simulated provider failure"},
        )
        ok = ExtractionResult(data={"invoice_number": "INV-001"}, metadata={})
        mock_result = BatchExtractionResult(
            results={str(doc): failed, "other.txt": ok},
            suggestions=[],
        )

        with patch("nextract.cli.commands.batch.BatchPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract_batch.return_value = mock_result
            mock_cls.return_value = instance

            result = runner.invoke(app, [
                "batch", str(doc),
                "--schema", str(schema_file),
                "--provider", "openai",
                "--model", "gpt-4o-mini",
            ])

        assert result.exit_code == 1
        combined = (result.output or "") + (getattr(result, "stderr", None) or "")
        assert "1 ok" in combined or "failed" in combined.lower()


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
