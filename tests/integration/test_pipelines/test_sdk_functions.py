"""
Integration tests for SDK-level functions (extract_simple, extract, batch_extract).
"""

import pytest

from tests.integration.conftest import requires_openai

from nextract import extract_simple, extract, batch_extract
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig


@pytest.mark.integration
@requires_openai
class TestExtractSimple:
    """Tests for extract_simple function."""

    def test_extract_simple_basic(self, sample_pdf_path, simple_schema):
        """Test basic extract_simple usage."""
        result = extract_simple(
            document=str(sample_pdf_path),
            schema=simple_schema,
            provider="openai",
            model="gpt-4o-mini",
            prompt="Extract invoice details",
        )
        
        assert result is not None
        assert result.data is not None

    def test_extract_simple_with_default_model(self, sample_pdf_path, simple_schema):
        """Test extract_simple with default model selection."""
        result = extract_simple(
            document=str(sample_pdf_path),
            schema=simple_schema,
            provider="openai",
        )
        
        assert result is not None


@pytest.mark.integration
@requires_openai
class TestExtractFunction:
    """Tests for extract function with explicit plan."""

    def test_extract_with_plan(self, sample_pdf_path, simple_schema):
        """Test extract with fully specified plan."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
            plan=plan,
            prompt="Extract the invoice details",
        )
        
        assert result is not None
        assert result.data is not None

    def test_extract_with_examples(self, sample_pdf_path, simple_schema):
        """Test extract with example outputs."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        examples = [
            {"invoice_number": "INV-001", "total": 500.00},
            {"invoice_number": "INV-002", "total": 750.00},
        ]
        
        result = extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
            plan=plan,
            examples=examples,
        )
        
        assert result is not None


@pytest.mark.integration
@requires_openai
class TestBatchExtractFunction:
    """Tests for batch_extract function."""

    def test_batch_extract_single(self, sample_pdf_path, simple_schema):
        """Test batch_extract with single document."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = batch_extract(
            documents=[str(sample_pdf_path)],
            schema=simple_schema,
            plan=plan,
            max_workers=1,
        )
        
        assert result is not None
        assert len(result.results) == 1

    @pytest.mark.slow
    def test_batch_extract_multiple(self, tmp_path, simple_schema, sample_text_content):
        """Test batch_extract with multiple documents."""
        docs = []
        for i in range(2):
            doc_path = tmp_path / f"invoice_{i}.txt"
            doc_path.write_text(f"Invoice: INV-{i}\nTotal: ${100 * i}\n{sample_text_content[:100]}")
            docs.append(str(doc_path))
        
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = batch_extract(
            documents=docs,
            schema=simple_schema,
            plan=plan,
            max_workers=2,
        )
        
        assert len(result.results) == 2
