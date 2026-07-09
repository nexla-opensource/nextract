"""Regression tests for logistical/UX error messages found in story testing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nextract import batch_extract, extract
from nextract.cli import app
from nextract.core import (
    ChunkerConfig,
    ExtractionPlan,
    ExtractionResult,
    ExtractorConfig,
    ProviderConfig,
)
from nextract.core.exceptions import PipelineError
from nextract.pipeline import BatchExtractionResult
from nextract.pipeline.orchestrator import ExtractionPipeline
from nextract.validate import PlanValidator

runner = CliRunner()


class TestUnknownComponentMessages:
    def test_plan_validator_unknown_extractor_lists_available(self) -> None:
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="notreal",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is False
        joined = " ".join(result.errors)
        assert "notreal" in joined
        assert "Available extractors:" in joined
        assert "text" in joined

    def test_plan_validator_unknown_chunker_lists_available(self) -> None:
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="does_not_exist"),
        )
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is False
        joined = " ".join(result.errors)
        assert "does_not_exist" in joined
        assert "Available chunkers:" in joined

    def test_pipeline_unknown_provider_lists_available(self) -> None:
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                # Use a supported provider name for plan validation, then
                # call _build_provider with an unknown name.
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        pipeline = ExtractionPipeline(plan)
        with pytest.raises(PipelineError, match="Available providers:"):
            pipeline._build_provider(ProviderConfig(name="not_a_provider", model="x"))


class TestExtractCliPlanValidationOrder:
    def test_num_passes_zero_no_credential_warning_first(self, tmp_path: Path) -> None:
        doc = tmp_path / "sample.txt"
        doc.write_text("hello invoice")
        schema = tmp_path / "schema.json"
        schema.write_text('{"type":"object","properties":{"x":{"type":"string"}}}')

        result = runner.invoke(
            app,
            [
                "extract",
                str(doc),
                "--schema",
                str(schema),
                "--extractor",
                "text",
                "--num-passes",
                "0",
            ],
        )
        assert result.exit_code == 1
        out = result.output or ""
        assert "num_passes" in out
        # Credential warning must not appear for pure plan validation failures.
        assert "credentials" not in out.lower()
        assert "OPENAI_API_KEY" not in out

    def test_unknown_extractor_lists_available_without_cred_warning(
        self, tmp_path: Path
    ) -> None:
        doc = tmp_path / "sample.txt"
        doc.write_text("hello")
        schema = tmp_path / "schema.json"
        schema.write_text('{"type":"object"}')

        result = runner.invoke(
            app,
            [
                "extract",
                str(doc),
                "--schema",
                str(schema),
                "--extractor",
                "notreal",
            ],
        )
        assert result.exit_code == 1
        out = result.output or ""
        assert "notreal" in out
        assert "Available extractors:" in out
        assert "credentials" not in out.lower()


class TestConvertWhitespaceText:
    def test_whitespace_only_txt_no_office_advice(self, tmp_path: Path) -> None:
        doc = tmp_path / "blank.txt"
        doc.write_text("   \n\n  ")
        result = runner.invoke(app, ["convert", str(doc), "--format", "markdown"])
        assert result.exit_code == 1
        out = (result.output or "").lower()
        assert "empty" in out or "whitespace" in out
        assert "libreoffice" not in out
        assert "soffice" not in out
        assert "tesseract" not in out


class TestListChunkersUnknownExtractor:
    def test_lists_available_extractors(self) -> None:
        result = runner.invoke(app, ["list", "chunkers", "--extractor", "unknown"])
        assert result.exit_code != 0
        out = result.output or ""
        assert "Unknown extractor" in out
        assert "Available extractors:" in out


class TestCheckProviderUnknown:
    def test_lists_available_providers(self) -> None:
        result = runner.invoke(app, ["check-provider", "unknown_provider"])
        assert result.exit_code != 0
        out = result.output or ""
        assert "Unknown provider" in out
        assert "Available providers:" in out


class TestPublicApiExtractEntryPoints:
    """Stories API-009 / API-010: drive public extract / batch_extract (not inspect)."""

    def _plan(self) -> ExtractionPlan:
        return ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )

    def test_extract_public_api_returns_pipeline_result(self, tmp_path: Path) -> None:
        doc = tmp_path / "invoice.txt"
        doc.write_text("Invoice INV-001")
        schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}
        expected = ExtractionResult(
            data={"invoice_number": "INV-001"},
            metadata={"source": str(doc)},
        )
        plan = self._plan()

        with patch("nextract.ExtractionPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract.return_value = expected
            mock_cls.return_value = instance

            result = extract(
                document=str(doc),
                schema=schema,
                plan=plan,
                prompt="extract invoice",
            )

        mock_cls.assert_called_once_with(plan)
        instance.extract.assert_called_once_with(
            document=str(doc),
            schema=schema,
            prompt="extract invoice",
            examples=None,
            include_extra=False,
        )
        assert result is expected
        assert result.data["invoice_number"] == "INV-001"

    def test_batch_extract_public_api_returns_batch_result(self, tmp_path: Path) -> None:
        doc = tmp_path / "invoice.txt"
        doc.write_text("Invoice INV-001")
        schema = {"type": "object"}
        plan = self._plan()
        expected = BatchExtractionResult(
            results={
                str(doc): ExtractionResult(data={"ok": True}, metadata={}),
            },
            suggestions=[],
        )

        with patch("nextract.BatchPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract_batch.return_value = expected
            mock_cls.return_value = instance

            result = batch_extract(
                documents=[str(doc)],
                schema=schema,
                plan=plan,
                max_workers=2,
            )

        mock_cls.assert_called_once_with(
            plan=plan,
            max_workers=2,
            enable_suggestions=False,
        )
        instance.extract_batch.assert_called_once_with(
            documents=[str(doc)],
            schema=schema,
            prompt=None,
            examples=None,
            include_extra=False,
        )
        assert result is expected
        assert str(doc) in result.results


class TestBatchCliPartialFailure:
    """Story CLI-015: batch CLI exits 1 when any result has metadata.error."""

    def test_batch_cli_exits_1_on_partial_failure(self, tmp_path: Path) -> None:
        doc_ok = tmp_path / "ok.txt"
        doc_ok.write_text("Invoice INV-001")
        doc_bad = tmp_path / "bad.txt"
        doc_bad.write_text("broken")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))

        mock_batch = BatchExtractionResult(
            results={
                str(doc_ok): ExtractionResult(data={"invoice": "INV-001"}, metadata={}),
                str(doc_bad): ExtractionResult(
                    data=None,
                    metadata={"error": "simulated failure"},
                ),
            },
            suggestions=[],
        )

        with patch("nextract.cli.commands.batch.BatchPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract_batch.return_value = mock_batch
            mock_cls.return_value = instance

            result = runner.invoke(
                app,
                [
                    "batch",
                    str(doc_ok),
                    str(doc_bad),
                    "--schema",
                    str(schema),
                    "--extractor",
                    "text",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 1
        combined = (result.output or "") + (getattr(result, "stderr", None) or "")
        assert "1 ok" in combined
        assert "1 failed" in combined or "failed" in combined.lower()
        instance.extract_batch.assert_called_once()
