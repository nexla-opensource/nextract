"""
Integration tests for ExtractionPlan validation.
"""

import pytest

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.validate import PlanValidator


@pytest.mark.integration
class TestPlanValidation:
    """Tests for plan validation."""

    def test_valid_text_plan(self):
        """Valid text extractor with semantic chunker."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_vlm_plan(self):
        """Valid VLM extractor with page chunker."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="vlm",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="page", pages_per_chunk=3),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True

    def test_invalid_chunker_modality(self):
        """VLM extractor with semantic chunker should fail."""
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

    def test_invalid_provider_for_extractor(self):
        """Textract extractor with OpenAI provider should fail."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="textract",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
                extractor_params={
                    "aws_access_key": "x",
                    "aws_secret_key": "y",
                    "region": "us-east-1",
                },
            ),
            chunker=ChunkerConfig(name="page"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is False
        assert any("does not support provider" in e for e in result.errors)


@pytest.mark.integration
class TestPlanValidationEdgeCases:
    """Edge case tests for plan validation."""

    def test_invalid_num_passes(self):
        """num_passes must be >= 1."""
        with pytest.raises(ValueError, match="num_passes must be >= 1"):
            plan = ExtractionPlan(
                extractor=ExtractorConfig(
                    name="text",
                    provider=ProviderConfig(name="openai", model="gpt-4o"),
                ),
                chunker=ChunkerConfig(name="semantic"),
                num_passes=0,
            )
            plan.validate()

    def test_invalid_backoff_factor(self):
        """backoff_factor must be >= 1."""
        with pytest.raises(ValueError, match="backoff_factor must be >= 1"):
            plan = ExtractionPlan(
                extractor=ExtractorConfig(
                    name="text",
                    provider=ProviderConfig(name="openai", model="gpt-4o"),
                ),
                chunker=ChunkerConfig(name="semantic"),
                backoff_factor=0.5,
            )
            plan.validate()

    def test_invalid_temperature(self):
        """Temperature must be between 0 and 2."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            config = ProviderConfig(name="openai", model="gpt-4o", temperature=3.0)
            config.validate()

    def test_fallback_provider_validation(self):
        """Fallback provider should also be validated."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
                fallback_provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        result = PlanValidator.validate_extraction_plan(plan)
        assert result.valid is True
