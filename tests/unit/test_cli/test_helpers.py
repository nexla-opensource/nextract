"""Unit tests for CLI helpers and extract/batch UX wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from nextract.cli import app
from nextract.cli.commands.extract import _parse_kv
from nextract.cli.helpers import default_chunker_for_extractor
from nextract.core import ExtractionResult
from nextract.pipeline import BatchExtractionResult

runner = CliRunner()


class TestDefaultChunkerForExtractor:
    @pytest.mark.parametrize(
        ("extractor", "expected"),
        [
            ("vlm", "page"),
            ("ocr", "page"),
            ("textract", "page"),
            ("text", "semantic"),
            ("llamaindex", "semantic"),
            ("hybrid", "hybrid"),
            ("VLM", "page"),
            (" Text ", "semantic"),
            ("unknown", "semantic"),
        ],
    )
    def test_mapping(self, extractor: str, expected: str) -> None:
        assert default_chunker_for_extractor(extractor) == expected


class TestParseKv:
    def test_valid_params(self) -> None:
        assert _parse_kv(["a=1", "b=two"]) == {"a": "1", "b": "two"}

    def test_empty(self) -> None:
        assert _parse_kv(None) == {}
        assert _parse_kv([]) == {}

    def test_invalid_raises_bad_parameter(self) -> None:
        with pytest.raises(typer.BadParameter, match="key=value"):
            _parse_kv(["not-a-kv"])


class TestExtractCliUx:
    def test_schema_missing_path_exits_nonzero(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        result = runner.invoke(
            app,
            ["extract", str(doc), "--schema", str(tmp_path / "missing.json")],
        )
        assert result.exit_code != 0

    def test_auto_chunker_from_extractor(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("Invoice INV-001")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))

        mock_result = ExtractionResult(data={"ok": True}, metadata={})
        with patch("nextract.cli.commands.extract.ExtractionPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract.return_value = mock_result
            mock_cls.return_value = instance

            result = runner.invoke(
                app,
                [
                    "extract",
                    str(doc),
                    "--schema",
                    str(schema),
                    "--extractor",
                    "vlm",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0
        plan = mock_cls.call_args.kwargs.get("plan") or mock_cls.call_args.args[0]
        assert plan.chunker.name == "page"

    def test_explicit_chunker_overrides_default(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("Invoice INV-001")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))

        mock_result = ExtractionResult(data={"ok": True}, metadata={})
        with patch("nextract.cli.commands.extract.ExtractionPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract.return_value = mock_result
            mock_cls.return_value = instance

            result = runner.invoke(
                app,
                [
                    "extract",
                    str(doc),
                    "--schema",
                    str(schema),
                    "--extractor",
                    "vlm",
                    "--chunker",
                    "semantic",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0
        plan = mock_cls.call_args.kwargs.get("plan") or mock_cls.call_args.args[0]
        assert plan.chunker.name == "semantic"

    def test_output_writes_parent_dirs_and_message(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))
        out = tmp_path / "nested" / "out" / "result.json"

        mock_result = ExtractionResult(data={"x": 1}, metadata={})
        with patch("nextract.cli.commands.extract.ExtractionPipeline") as mock_cls:
            instance = MagicMock()
            instance.extract.return_value = mock_result
            mock_cls.return_value = instance

            result = runner.invoke(
                app,
                [
                    "extract",
                    str(doc),
                    "--schema",
                    str(schema),
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o-mini",
                    "--output",
                    str(out),
                ],
            )

        assert result.exit_code == 0
        assert out.exists()
        combined = (result.output or "") + (getattr(result, "stderr", None) or "")
        assert "Wrote" in combined
        assert json.loads(out.read_text())["data"] == {"x": 1}

    def test_invalid_params_surface_bad_parameter(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))

        result = runner.invoke(
            app,
            [
                "extract",
                str(doc),
                "--schema",
                str(schema),
                "--extractor-params",
                "not-a-kv",
            ],
        )
        assert result.exit_code != 0
        # Should not wrap as generic "Extraction failed"
        combined = (result.output or "") + (getattr(result, "stderr", None) or "")
        assert "Extraction failed" not in combined


class TestBatchCliUx:
    def test_schema_missing_path_exits_nonzero(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        result = runner.invoke(
            app,
            ["batch", str(doc), "--schema", str(tmp_path / "missing.json")],
        )
        assert result.exit_code != 0

    def test_auto_chunker_from_extractor(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("Invoice INV-001")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))

        mock_batch = BatchExtractionResult(
            results={str(doc): ExtractionResult(data={"ok": True}, metadata={})},
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
                    str(doc),
                    "--schema",
                    str(schema),
                    "--extractor",
                    "hybrid",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0
        plan = mock_cls.call_args.kwargs["plan"]
        assert plan.chunker.name == "hybrid"

    def test_output_option(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps({"type": "object", "properties": {}}))
        out = tmp_path / "batch_out.json"

        mock_batch = BatchExtractionResult(
            results={str(doc): ExtractionResult(data={"ok": True}, metadata={})},
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
                    str(doc),
                    "--schema",
                    str(schema),
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o-mini",
                    "--output",
                    str(out),
                ],
            )

        assert result.exit_code == 0
        assert out.exists()
        combined = (result.output or "") + (getattr(result, "stderr", None) or "")
        assert "Wrote" in combined
