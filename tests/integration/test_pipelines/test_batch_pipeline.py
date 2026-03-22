"""
Integration tests for BatchPipeline.
"""

import pytest

from tests.integration.conftest import requires_openai

from nextract import BatchPipeline
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig


@pytest.mark.integration
class TestBatchPipelineUnit:
    """Unit tests for BatchPipeline."""

    def test_pipeline_initialization(self):
        """BatchPipeline initializes with valid plan."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = BatchPipeline(plan=plan, max_workers=4)
        assert pipeline is not None
        assert pipeline.max_workers == 4

    def test_pipeline_with_suggestions_enabled(self):
        """BatchPipeline with schema suggestions enabled."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = BatchPipeline(plan=plan, enable_suggestions=True)
        assert pipeline.enable_suggestions is True


@pytest.mark.integration
@requires_openai
class TestBatchPipelineLive:
    """Live tests requiring OpenAI credentials."""

    def test_extract_single_document(self, sample_pdf_path, simple_schema):
        """Test batch extraction with single document."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = BatchPipeline(plan=plan, max_workers=1)
        result = pipeline.extract_batch(
            documents=[str(sample_pdf_path)],
            schema=simple_schema,
        )
        
        assert result is not None
        assert len(result.results) == 1
        assert str(sample_pdf_path) in result.results

    @pytest.mark.slow
    def test_extract_multiple_documents(self, tmp_path, simple_schema, sample_text_content):
        """Test batch extraction with multiple documents."""
        docs = []
        for i in range(3):
            doc_path = tmp_path / f"doc_{i}.txt"
            doc_path.write_text(f"Invoice INV-{i:03d}\nTotal: ${100 * (i + 1)}\n{sample_text_content[:200]}")
            docs.append(str(doc_path))
        
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = BatchPipeline(plan=plan, max_workers=2)
        result = pipeline.extract_batch(
            documents=docs,
            schema=simple_schema,
        )
        
        assert len(result.results) == 3

    def test_extract_with_progress_callback(self, sample_pdf_path, simple_schema):
        """Test batch extraction with progress callback."""
        progress_values = []
        
        def progress_callback(pct):
            progress_values.append(pct)
        
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = BatchPipeline(plan=plan, max_workers=1, progress_callback=progress_callback)
        _ = pipeline.extract_batch(
            documents=[str(sample_pdf_path)],
            schema=simple_schema,
        )
        
        assert len(progress_values) > 0
        assert 100 in progress_values
