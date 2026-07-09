"""Unit tests for SDK DX helpers on the package root API."""

from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest

import nextract
from nextract import (
    NextractError,
    PipelineError,
    PlanError,
    ProviderAuthError,
    ProviderRequestError,
    batch_extract,
    extract,
    extract_simple,
    get_available_chunkers,
)
from nextract.core.artifacts import ExtractionResult
from nextract.pipeline import BatchExtractionResult, BatchPipeline
from nextract import _mode_to_extractor_chunker


def test_extract_simple_return_annotation() -> None:
    hints = get_type_hints(extract_simple)
    assert hints["return"] is ExtractionResult


def test_extract_return_annotation() -> None:
    hints = get_type_hints(extract)
    assert hints["return"] is ExtractionResult


def test_batch_extract_return_annotation() -> None:
    hints = get_type_hints(batch_extract)
    assert hints["return"] is BatchExtractionResult


def test_extract_simple_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown mode 'magic'"):
        extract_simple(
            document="doc.pdf",
            schema={"type": "object"},
            provider="openai",
            mode="magic",
        )


def test_extract_simple_allowed_modes_in_error() -> None:
    with pytest.raises(ValueError, match="auto") as exc_info:
        extract_simple(
            document="doc.pdf",
            schema={"type": "object"},
            provider="openai",
            mode="nope",
        )
    message = str(exc_info.value)
    for mode in ("auto", "text", "visual", "ocr", "textract", "hybrid"):
        assert mode in message


@pytest.mark.parametrize(
    ("mode", "extractor", "chunker"),
    [
        ("text", "text", "semantic"),
        ("visual", "vlm", "page"),
        ("ocr", "ocr", "page"),
        ("textract", "textract", "page"),
        ("hybrid", "hybrid", "hybrid"),
    ],
)
def test_mode_to_extractor_chunker(mode: str, extractor: str, chunker: str) -> None:
    assert _mode_to_extractor_chunker(mode, "openai") == (extractor, chunker)


def test_mode_ocr_provider_overrides_text_mode() -> None:
    assert _mode_to_extractor_chunker("text", "tesseract") == ("ocr", "page")


def test_get_available_chunkers_unknown_extractor() -> None:
    with pytest.raises(ValueError, match="Unknown extractor 'not_a_real_extractor'"):
        get_available_chunkers("not_a_real_extractor")


def test_get_available_chunkers_known_extractor() -> None:
    chunkers = get_available_chunkers("text")
    assert isinstance(chunkers, list)
    assert "semantic" in chunkers


def test_package_exports_exceptions() -> None:
    for name, cls in [
        ("NextractError", NextractError),
        ("PipelineError", PipelineError),
        ("PlanError", PlanError),
        ("ProviderAuthError", ProviderAuthError),
        ("ProviderRequestError", ProviderRequestError),
    ]:
        assert name in nextract.__all__
        assert getattr(nextract, name) is cls


def test_core_exports_provider_auth_and_request_errors() -> None:
    from nextract import core

    assert core.ProviderAuthError is ProviderAuthError
    assert core.ProviderRequestError is ProviderRequestError
    assert "ProviderAuthError" in core.__all__
    assert "ProviderRequestError" in core.__all__


def test_batch_extract_passes_enable_suggestions() -> None:
    plan = MagicMock()
    schema = {"type": "object"}
    fake_result = MagicMock(spec=BatchExtractionResult)

    with patch("nextract.BatchPipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.extract_batch.return_value = fake_result
        mock_cls.return_value = mock_pipeline

        result = batch_extract(
            documents=["a.pdf"],
            schema=schema,
            plan=plan,
            max_workers=2,
            enable_suggestions=True,
        )

    mock_cls.assert_called_once_with(
        plan=plan,
        max_workers=2,
        enable_suggestions=True,
    )
    mock_pipeline.extract_batch.assert_called_once_with(
        documents=["a.pdf"],
        schema=schema,
        prompt=None,
        examples=None,
        include_extra=False,
    )
    assert result is fake_result


def test_batch_extract_enable_suggestions_default_false() -> None:
    sig = inspect.signature(batch_extract)
    assert sig.parameters["enable_suggestions"].default is False

    bp_sig = inspect.signature(BatchPipeline.__init__)
    assert bp_sig.parameters["enable_suggestions"].default is False
