"""Unit tests for HybridExtractor result sorting/merge with ChunkExtraction objects."""

from __future__ import annotations

from unittest.mock import MagicMock

from nextract.core import (
    ChunkExtraction,
    ExtractorConfig,
    ExtractorResult,
    ImageChunk,
    ProviderConfig,
    TextChunk,
)
from nextract.extractors.hybrid_extractor import HybridExtractor


def _chunk_extraction(
    chunk_id: str,
    response: dict | None = None,
    *,
    hybrid_order: int | None = None,
) -> ChunkExtraction:
    metadata = {}
    if hybrid_order is not None:
        metadata["hybrid_order"] = hybrid_order
    return ChunkExtraction(
        chunk_id=chunk_id,
        response=response or {"ok": True},
        metadata=metadata,
    )


def test_result_sort_key_uses_chunk_extraction_attributes():
    """Sort key must use ChunkExtraction attributes, not dict .get()."""
    with_order = _chunk_extraction("chunk_9", hybrid_order=2)
    by_id = _chunk_extraction("chunk_3")

    assert HybridExtractor._result_sort_key(with_order) == (0, 2, "chunk_9")
    assert HybridExtractor._result_sort_key(by_id) == (1, 3, "chunk_3")


def test_result_sort_key_handles_missing_numeric_chunk_id():
    result = _chunk_extraction("no-number")
    assert HybridExtractor._result_sort_key(result) == (1, 0, "no-number")


def test_hybrid_run_sorts_chunk_extractions_without_attribute_error(monkeypatch):
    """Hybrid merge of visual+text results must not AttributeError on .get()."""
    extractor = HybridExtractor()

    # Avoid real vision capability / provider init path.
    monkeypatch.setattr(extractor, "validate_config", lambda config: True)
    monkeypatch.setattr(extractor._vlm, "initialize", lambda config: None)
    monkeypatch.setattr(extractor._text, "initialize", lambda config: None)
    extractor.initialize(
        ExtractorConfig(
            name="hybrid",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        )
    )

    visual = ImageChunk(
        id="img_0",
        images=[b"fake"],
        source_path="doc.pdf",
        page_range=(0, 0),
        metadata={"hybrid_order": 0},
    )
    text = TextChunk(
        id="txt_1",
        text="hello",
        source_path="doc.pdf",
        metadata={"hybrid_order": 1},
    )

    # Sub-extractors return results out of hybrid order to force a sort.
    text_only = _chunk_extraction("txt_1", {"from": "text"}, hybrid_order=1)
    visual_only = _chunk_extraction("img_0", {"from": "vlm"}, hybrid_order=0)

    monkeypatch.setattr(
        extractor._vlm,
        "run",
        lambda *a, **k: ExtractorResult(
            name="vlm",
            provider_name="mock",
            results=[visual_only],
        ),
    )
    monkeypatch.setattr(
        extractor._text,
        "run",
        lambda *a, **k: ExtractorResult(
            name="text",
            provider_name="mock",
            results=[text_only],
        ),
    )

    provider = MagicMock()
    provider.get_name.return_value = "mock"

    result = extractor.run(
        [visual, text],
        provider=provider,
        prompt="extract",
        schema=None,
    )

    assert [r.chunk_id for r in result.results] == ["img_0", "txt_1"]
    assert result.metadata["visual_chunks"] == 1
    assert result.metadata["text_chunks"] == 1
    # Ensure objects remain ChunkExtraction, not dicts
    assert all(isinstance(r, ChunkExtraction) for r in result.results)


def test_hybrid_routes_by_hybrid_source_metadata(monkeypatch):
    """hybrid_source metadata must win over duck-typed attributes."""
    extractor = HybridExtractor()
    monkeypatch.setattr(extractor, "validate_config", lambda config: True)
    monkeypatch.setattr(extractor._vlm, "initialize", lambda config: None)
    monkeypatch.setattr(extractor._text, "initialize", lambda config: None)
    extractor.initialize(
        ExtractorConfig(
            name="hybrid",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        )
    )

    # TextChunk mis-tagged as visual must go to VLM, not text extractor.
    mis_tagged_visual = TextChunk(
        id="fake_visual",
        text="should still be treated as visual",
        source_path="doc.pdf",
        metadata={"hybrid_source": "visual", "hybrid_order": 0},
    )
    text = TextChunk(
        id="txt_1",
        text="hello",
        source_path="doc.pdf",
        metadata={"hybrid_source": "text", "hybrid_order": 1},
    )

    seen: dict[str, list] = {"vlm": [], "text": []}

    def vlm_run(chunks, **kwargs):
        seen["vlm"] = list(chunks)
        return ExtractorResult(
            name="vlm",
            provider_name="mock",
            results=[_chunk_extraction("fake_visual", hybrid_order=0)],
        )

    def text_run(chunks, **kwargs):
        seen["text"] = list(chunks)
        return ExtractorResult(
            name="text",
            provider_name="mock",
            results=[_chunk_extraction("txt_1", hybrid_order=1)],
        )

    monkeypatch.setattr(extractor._vlm, "run", vlm_run)
    monkeypatch.setattr(extractor._text, "run", text_run)

    provider = MagicMock()
    provider.get_name.return_value = "mock"
    extractor.run([mis_tagged_visual, text], provider=provider, prompt="extract")

    assert [c.id for c in seen["vlm"]] == ["fake_visual"]
    assert [c.id for c in seen["text"]] == ["txt_1"]


