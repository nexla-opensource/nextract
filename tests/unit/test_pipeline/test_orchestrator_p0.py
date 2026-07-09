"""Unit tests for P0 pipeline empty-result and plan validation fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nextract.core import (
    ChunkExtraction,
    ChunkerConfig,
    DocumentArtifact,
    ExtractionPlan,
    ExtractorConfig,
    ExtractorResult,
    PipelineError,
    PlanError,
    ProviderConfig,
    ProviderUsage,
    ValidationResult,
)
from nextract.pipeline.orchestrator import BatchPipeline, ExtractionPipeline
from nextract.validate import PlanValidator


def _make_plan(**overrides) -> ExtractionPlan:
    kwargs = {
        "extractor": ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o", max_retries=1),
        ),
        "chunker": ChunkerConfig(name="semantic"),
    }
    kwargs.update(overrides)
    return ExtractionPlan(**kwargs)


def _pipeline_with_mocks(
    plan: ExtractionPlan | None = None,
    *,
    chunks: list | None = None,
    extract_response: dict | None = None,
    doc_warnings: list[str] | None = None,
) -> ExtractionPipeline:
    """Build ExtractionPipeline without real provider/extractor/chunker init."""
    plan = plan or _make_plan()
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.plan = plan

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = chunks if chunks is not None else []
    pipeline.chunker = mock_chunker

    response = extract_response if extract_response is not None else {"name": "Ada", "age": 30}
    mock_extractor = MagicMock()
    mock_extractor.run.return_value = ExtractorResult(
        name="text",
        provider_name="openai",
        results=[
            ChunkExtraction(
                chunk_id="c0",
                response=response,
                usage=ProviderUsage(requests=1, input_tokens=10, output_tokens=5),
            )
        ],
    )
    pipeline.extractor = mock_extractor
    pipeline.provider = MagicMock()

    # Stash for test assertions / patching
    pipeline._test_doc_warnings = doc_warnings or []
    return pipeline


class TestPlanValidateInvoked:
    """plan.validate() must run via PlanValidator (production path)."""

    def test_invalid_num_passes_rejected_by_plan_validator(self):
        plan = _make_plan(num_passes=0)
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is False
        assert any("num_passes must be >= 1" in e for e in result.errors)

    def test_num_passes_above_limit_rejected_by_plan_validator(self):
        plan = _make_plan(num_passes=21)
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is False
        assert any("num_passes must be <=" in e for e in result.errors)

    def test_raise_for_invalid_rejects_bad_num_passes(self):
        plan = _make_plan(num_passes=0)
        with pytest.raises(PlanError, match="num_passes must be >= 1"):
            PlanValidator.raise_for_invalid(plan)

    def test_pipeline_init_rejects_invalid_num_passes(self):
        plan = _make_plan(num_passes=0)
        with pytest.raises(PlanError, match="num_passes must be >= 1"):
            ExtractionPipeline(plan)

    def test_max_retries_propagated_when_provider_unset(self):
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),  # max_retries=None
            ),
            chunker=ChunkerConfig(name="semantic"),
            max_retries=7,
        )
        assert plan.extractor.provider.max_retries is None
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True
        assert plan.extractor.provider.max_retries == 7
        assert plan.extractor.provider.backoff_factor == plan.backoff_factor

    def test_explicit_provider_max_retries_not_overwritten(self):
        plan = _make_plan(max_retries=7)
        # Provider was constructed with max_retries=1 — must not be clobbered.
        assert plan.extractor.provider.max_retries == 1
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True
        assert plan.extractor.provider.max_retries == 1

    def test_retry_on_failure_false_forces_single_attempt(self):
        """retry_on_failure=False is a kill-switch even if provider set max_retries."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o", max_retries=5),
            ),
            chunker=ChunkerConfig(name="semantic"),
            retry_on_failure=False,
            max_retries=7,
        )
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True
        assert plan.extractor.provider.max_retries == 1


class TestEmptyChunksRaise:
    """Empty chunk generation must fail, not return silent empty success."""

    def test_empty_chunks_raise_pipeline_error(self, tmp_path: Path, monkeypatch):
        doc = tmp_path / "empty_ish.txt"
        doc.write_text("hello")

        pipeline = _pipeline_with_mocks(chunks=[])

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="hello")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(valid=True, errors=[], warnings=[]),
        )

        with pytest.raises(PipelineError, match="No chunks generated") as exc_info:
            pipeline.extract(document=str(doc), schema={"type": "object", "properties": {}})

        message = str(exc_info.value)
        assert "empty document" in message or "LibreOffice" in message
        assert "OCR" in message or "modality" in message
        assert exc_info.value.stage == "chunk"


class TestDocumentWarningsSurfaced:
    """DocumentValidator warnings must appear in result metadata."""

    def test_doc_warnings_in_metadata(self, tmp_path: Path, monkeypatch):
        doc = tmp_path / "sample.txt"
        doc.write_text("invoice content")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        pipeline = _pipeline_with_mocks(
            chunks=[MagicMock()],
            extract_response={"name": "Ada", "age": 30},
        )

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="invoice")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(
                valid=True,
                errors=[],
                warnings=["PyMuPDF not available; skipping PDF structure validation"],
            ),
        )

        result = pipeline.extract(document=str(doc), schema=schema)
        assert "warnings" in result.metadata
        assert any("PyMuPDF" in w for w in result.metadata["warnings"])


class TestStrictValidation:
    """strict_validation must fail clearly when schema validation fails."""

    def test_strict_validation_raises_on_schema_failure(self, tmp_path: Path, monkeypatch):
        doc = tmp_path / "sample.txt"
        doc.write_text("invoice content")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        # age is a string -> schema validation fails
        plan = _make_plan(strict_validation=True, schema_validation=True)
        pipeline = _pipeline_with_mocks(
            plan=plan,
            chunks=[MagicMock()],
            extract_response={"name": "Ada", "age": "not-an-int"},
        )

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="invoice")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(valid=True, errors=[], warnings=[]),
        )

        with pytest.raises(PipelineError, match="strict_validation") as exc_info:
            pipeline.extract(document=str(doc), schema=schema)
        assert exc_info.value.stage == "validate"

    def test_non_strict_keeps_validation_in_metadata(self, tmp_path: Path, monkeypatch):
        doc = tmp_path / "sample.txt"
        doc.write_text("invoice content")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        plan = _make_plan(strict_validation=False, schema_validation=True)
        pipeline = _pipeline_with_mocks(
            plan=plan,
            chunks=[MagicMock()],
            extract_response={"name": "Ada", "age": "not-an-int"},
        )

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="invoice")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(valid=True, errors=[], warnings=[]),
        )

        result = pipeline.extract(document=str(doc), schema=schema)
        # validation is stored as a JSON-serializable dict for CLI/output consumers
        assert result.metadata["validation"]["valid"] is False
        assert result.data["name"] == "Ada"

    def test_validation_metadata_is_json_serializable(self, tmp_path: Path, monkeypatch):
        import json

        doc = tmp_path / "sample.txt"
        doc.write_text("invoice content")
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        pipeline = _pipeline_with_mocks(
            chunks=[MagicMock()],
            extract_response={"name": "Ada"},
        )
        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="invoice")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(valid=True, errors=[], warnings=[]),
        )

        result = pipeline.extract(document=str(doc), schema=schema)
        payload = {"data": result.data, "metadata": result.metadata}
        # Must not raise TypeError (ValidationResult is not JSON serializable).
        serialized = json.dumps(payload)
        assert '"validation"' in serialized
        assert isinstance(result.metadata["validation"], dict)


class TestMetadataNameAliases:
    """Result metadata must expose both canonical and alias name keys."""

    def test_provider_and_extractor_name_aliases(self, tmp_path: Path, monkeypatch):
        doc = tmp_path / "sample.txt"
        doc.write_text("invoice content")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        pipeline = _pipeline_with_mocks(
            chunks=[MagicMock()],
            extract_response={"name": "Ada", "age": 30},
        )

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.load_documents",
            lambda docs: [
                DocumentArtifact(source_path=str(doc), mime_type="text/plain", text="invoice")
            ],
        )
        monkeypatch.setattr(
            "nextract.ingest.DocumentValidator.validate",
            lambda self, artifact: ValidationResult(valid=True, errors=[], warnings=[]),
        )

        result = pipeline.extract(document=str(doc), schema=schema)
        assert result.metadata["provider"] == "openai"
        assert result.metadata["provider_name"] == "openai"
        assert result.metadata["extractor"] == "text"
        assert result.metadata["extractor_name"] == "text"


class TestBatchPipelineIsolation:
    """Batch workers must not share provider/extractor instances."""

    def test_batch_creates_pipeline_per_document(self, tmp_path: Path, monkeypatch):
        docs = []
        for i in range(3):
            p = tmp_path / f"doc{i}.txt"
            p.write_text(f"content {i}")
            docs.append(str(p))

        plan = _make_plan()
        constructed: list[object] = []
        extract_calls: list[str] = []

        class FakePipeline:
            def __init__(self, plan_arg):
                constructed.append(self)
                self.plan = plan_arg
                self.provider = MagicMock(name=f"provider-{len(constructed)}")
                self.extractor = MagicMock(name=f"extractor-{len(constructed)}")
                self.chunker = MagicMock(name=f"chunker-{len(constructed)}")

            def extract(self, document, schema, prompt=None, examples=None, include_extra=False):
                extract_calls.append(document)
                from nextract.core import ExtractionResult

                return ExtractionResult(
                    data={"doc": document},
                    metadata={"provider": "mock"},
                )

        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.ExtractionPipeline",
            FakePipeline,
        )
        monkeypatch.setattr(
            "nextract.pipeline.orchestrator.PlanValidator.raise_for_invalid",
            lambda plan: None,
        )

        batch = BatchPipeline(plan=plan, max_workers=3)
        result = batch.extract_batch(
            documents=docs,
            schema={"type": "object", "properties": {}},
        )

        assert len(constructed) == 3
        assert len({id(p) for p in constructed}) == 3
        assert set(extract_calls) == set(docs)
        assert set(result.results.keys()) == set(docs)
