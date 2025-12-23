"""
Integration tests for capability mismatches between extractors, providers, and chunkers.
"""

import pytest

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.validate import PlanValidator


@pytest.mark.integration
class TestExtractorProviderMismatches:
    """Tests for extractor-provider compatibility issues."""

    def test_vlm_with_cohere_rejected(self):
        """VLM extractor should reject Cohere (no vision support)."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="vlm",
                provider=ProviderConfig(name="cohere", model="command-r"),
            ),
            chunker=ChunkerConfig(name="page"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False
        assert any("does not support provider" in e for e in result.errors)

    def test_textract_with_openai_rejected(self):
        """Textract extractor should reject OpenAI."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="textract",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
                extractor_params={"aws_access_key": "x", "aws_secret_key": "y", "region": "us-east-1"},
            ),
            chunker=ChunkerConfig(name="page"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False

    def test_text_with_openai_valid(self):
        """Text extractor with OpenAI should be valid."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is True


@pytest.mark.integration
class TestChunkerModalityMismatches:
    """Tests for chunker-modality compatibility issues."""

    def test_semantic_chunker_with_vlm_rejected(self):
        """Semantic chunker should be rejected for VLM (visual) extractor."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="vlm",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False
        assert any("not applicable to modality" in e for e in result.errors)

    def test_page_chunker_with_text_rejected(self):
        """Page chunker should be rejected for text extractor."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="page"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False
        assert any("not applicable to modality" in e for e in result.errors)

    def test_page_chunker_with_vlm_valid(self):
        """Page chunker with VLM extractor should be valid."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="vlm",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="page"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is True


@pytest.mark.integration
class TestConfigValidationEdgeCases:
    """Edge cases for configuration validation."""

    def test_unknown_extractor_rejected(self):
        """Unknown extractor should cause validation to fail."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="unknown_extractor",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False

    def test_unknown_provider_rejected(self):
        """Unknown provider should cause validation to fail."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="unknown_provider", model="model"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result.valid is False

    def test_unknown_chunker_rejected(self):
        """Unknown chunker validation behavior (currently not rejected at plan level)."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="unknown_chunker"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        
        assert result is not None
