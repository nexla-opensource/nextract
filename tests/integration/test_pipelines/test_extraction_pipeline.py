"""
Integration tests for ExtractionPipeline.
"""

import pytest

from tests.integration.conftest import requires_openai

from nextract import ExtractionPipeline
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig


@pytest.mark.integration
class TestExtractionPipelineUnit:
    """Unit tests for ExtractionPipeline."""

    def test_pipeline_initialization(self):
        """Pipeline initializes with valid plan."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = ExtractionPipeline(plan)
        assert pipeline is not None
        assert pipeline.plan == plan

    def test_pipeline_with_invalid_plan(self):
        """Pipeline should reject invalid plan."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="vlm",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        with pytest.raises(Exception, match="not applicable to modality"):
            ExtractionPipeline(plan)

    def test_pipeline_with_multipass(self):
        """Pipeline with multi-pass configuration."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            num_passes=3,
        )
        
        pipeline = ExtractionPipeline(plan)
        assert pipeline.plan.num_passes == 3


@pytest.mark.integration
@requires_openai
class TestExtractionPipelineLive:
    """Live tests requiring OpenAI credentials."""

    def test_extract_from_text_file(self, sample_pdf_path, simple_schema):
        """Test extraction from a text file."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic", chunk_size=2000),
        )
        
        pipeline = ExtractionPipeline(plan)
        result = pipeline.extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
            prompt="Extract the invoice details",
        )
        
        assert result is not None
        assert result.data is not None
        assert "metadata" in dir(result)

    @pytest.mark.slow
    def test_extract_with_multipass(self, sample_pdf_path, simple_schema):
        """Test extraction with multiple passes."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            num_passes=2,
        )
        
        pipeline = ExtractionPipeline(plan)
        result = pipeline.extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
        )
        
        assert result is not None
        assert result.metadata.get("passes") == 2

    def test_extract_with_include_extra(self, sample_pdf_path, simple_schema):
        """Test extraction with include_extra flag."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = ExtractionPipeline(plan)
        result = pipeline.extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
            include_extra=True,
        )
        
        assert result is not None

    def test_extract_returns_usage_metadata(self, sample_pdf_path, simple_schema):
        """Extraction should return token usage in metadata."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        pipeline = ExtractionPipeline(plan)
        result = pipeline.extract(
            document=str(sample_pdf_path),
            schema=simple_schema,
        )
        
        assert "usage" in result.metadata
        usage = result.metadata["usage"]
        assert "input_tokens" in usage or "requests" in usage
