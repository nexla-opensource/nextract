"""
Integration tests for error recovery and retry handling.
"""

import pytest

from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig


@pytest.mark.integration
class TestRetryPolicy:
    """Tests for retry policy configuration."""

    def test_default_retry_settings(self):
        """Default retry settings should be sensible."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        assert plan.retry_on_failure is True
        assert plan.max_retries == 3
        assert plan.backoff_factor == 2.0

    def test_custom_retry_settings(self):
        """Custom retry settings should be respected."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            retry_on_failure=False,
            max_retries=5,
            backoff_factor=1.5,
        )
        
        assert plan.retry_on_failure is False
        assert plan.max_retries == 5
        assert plan.backoff_factor == 1.5

    def test_provider_timeout_settings(self):
        """Provider timeout should be configurable."""
        config = ProviderConfig(
            name="openai",
            model="gpt-4o",
            timeout=120,
            max_retries=5,
        )
        
        assert config.timeout == 120
        assert config.max_retries == 5


@pytest.mark.integration
class TestGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_fallback_provider_config(self):
        """Fallback provider should be configurable."""
        primary = ProviderConfig(name="openai", model="gpt-4o")
        fallback = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022")
        
        config = ExtractorConfig(
            name="text",
            provider=primary,
            fallback_provider=fallback,
        )
        
        assert config.fallback_provider is not None
        assert config.fallback_provider.name == "anthropic"

    def test_plan_with_fallback(self):
        """Plan with fallback provider should validate."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
                fallback_provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        assert plan.extractor.fallback_provider is not None


@pytest.mark.integration
class TestValidationErrors:
    """Tests for validation error handling."""

    def test_invalid_temperature_raises(self):
        """Invalid temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature"):
            config = ProviderConfig(name="openai", model="gpt-4o", temperature=5.0)
            config.validate()

    def test_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature"):
            config = ProviderConfig(name="openai", model="gpt-4o", temperature=-1.0)
            config.validate()

    def test_missing_provider_name_raises(self):
        """Missing provider name should raise ValueError."""
        with pytest.raises(ValueError, match="required"):
            config = ProviderConfig(name="", model="gpt-4o")
            config.validate()

    def test_missing_model_raises(self):
        """Missing model should raise ValueError."""
        with pytest.raises(ValueError, match="required"):
            config = ProviderConfig(name="openai", model="")
            config.validate()
