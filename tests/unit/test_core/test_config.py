"""
Unit tests for core configuration classes.

Tests:
- ProviderConfig validation
- ExtractorConfig validation
- ChunkerConfig validation
- ExtractionPlan validation
"""

import pytest

from nextract.core import (
    ChunkerConfig,
    ExtractionPlan,
    ExtractorConfig,
    Modality,
    ProviderConfig,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_basic_creation(self):
        """Create basic provider config."""
        config = ProviderConfig(name="openai", model="gpt-4o")
        assert config.name == "openai"
        assert config.model == "gpt-4o"

    def test_default_values(self):
        """Check default values."""
        config = ProviderConfig(name="openai", model="gpt-4o")
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.temperature == 0.0
        assert config.api_key is None
        assert config.api_base is None

    def test_custom_values(self):
        """Create config with custom values."""
        config = ProviderConfig(
            name="openai",
            model="gpt-4o",
            api_key="test-key",
            timeout=120,
            max_retries=5,
            temperature=0.7,
        )
        assert config.api_key == "test-key"
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.temperature == 0.7

    def test_validate_success(self):
        """Validation passes for valid config."""
        config = ProviderConfig(name="openai", model="gpt-4o")
        assert config.validate() is True

    def test_validate_missing_name(self):
        """Validation fails for missing name."""
        config = ProviderConfig(name="", model="gpt-4o")
        with pytest.raises(ValueError, match="required"):
            config.validate()

    def test_validate_missing_model(self):
        """Validation fails for missing model."""
        config = ProviderConfig(name="openai", model="")
        with pytest.raises(ValueError, match="required"):
            config.validate()

    def test_validate_invalid_temperature_high(self):
        """Validation fails for temperature > 2."""
        config = ProviderConfig(name="openai", model="gpt-4o", temperature=2.5)
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            config.validate()

    def test_validate_invalid_temperature_negative(self):
        """Validation fails for negative temperature."""
        config = ProviderConfig(name="openai", model="gpt-4o", temperature=-0.5)
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            config.validate()

    def test_extra_params(self):
        """Extra params are stored correctly."""
        config = ProviderConfig(
            name="openai",
            model="gpt-4o",
            extra_params={"custom_key": "custom_value"},
        )
        assert config.extra_params["custom_key"] == "custom_value"


class TestExtractorConfig:
    """Tests for ExtractorConfig."""

    def test_basic_creation(self):
        """Create basic extractor config."""
        provider = ProviderConfig(name="openai", model="gpt-4o")
        config = ExtractorConfig(name="text", provider=provider)
        assert config.name == "text"
        assert config.provider.name == "openai"

    def test_default_values(self):
        """Check default values."""
        provider = ProviderConfig(name="openai", model="gpt-4o")
        config = ExtractorConfig(name="text", provider=provider)
        assert config.enable_caching is True
        assert config.batch_size == 1
        assert config.fallback_provider is None

    def test_with_fallback_provider(self):
        """Config with fallback provider."""
        primary = ProviderConfig(name="openai", model="gpt-4o")
        fallback = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022")
        config = ExtractorConfig(name="text", provider=primary, fallback_provider=fallback)
        assert config.fallback_provider is not None
        assert config.fallback_provider.name == "anthropic"

    def test_extractor_params(self):
        """Extractor params are stored correctly."""
        provider = ProviderConfig(name="openai", model="gpt-4o")
        config = ExtractorConfig(
            name="text",
            provider=provider,
            extractor_params={"custom_param": "value"},
        )
        assert config.extractor_params["custom_param"] == "value"


class TestChunkerConfig:
    """Tests for ChunkerConfig."""

    def test_basic_creation(self):
        """Create basic chunker config."""
        config = ChunkerConfig(name="semantic")
        assert config.name == "semantic"

    def test_default_values(self):
        """Check default values."""
        config = ChunkerConfig(name="page")
        assert config.pages_per_chunk == 5
        assert config.page_overlap == 1
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 200

    def test_custom_page_chunker_values(self):
        """Custom values for page chunker."""
        config = ChunkerConfig(name="page", pages_per_chunk=3, page_overlap=1)
        assert config.pages_per_chunk == 3
        assert config.page_overlap == 1

    def test_custom_text_chunker_values(self):
        """Custom values for text chunker."""
        config = ChunkerConfig(name="semantic", chunk_size=1500, chunk_overlap=150)
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 150

    def test_validate_visual_modality(self):
        """Validation for visual modality."""
        config = ChunkerConfig(name="page", pages_per_chunk=3, page_overlap=1)
        assert config.validate(Modality.VISUAL) is True

    def test_validate_visual_invalid_pages(self):
        """Validation fails for pages_per_chunk < 1."""
        config = ChunkerConfig(name="page", pages_per_chunk=0)
        with pytest.raises(ValueError, match="pages_per_chunk must be >= 1"):
            config.validate(Modality.VISUAL)

    def test_validate_visual_invalid_overlap(self):
        """Validation fails when overlap >= pages_per_chunk."""
        config = ChunkerConfig(name="page", pages_per_chunk=3, page_overlap=3)
        with pytest.raises(ValueError, match="page_overlap must be < pages_per_chunk"):
            config.validate(Modality.VISUAL)

    def test_validate_text_invalid_chunk_size(self):
        """Validation fails for chunk_size < min_chunk_size."""
        config = ChunkerConfig(name="semantic", chunk_size=50, min_chunk_size=100)
        with pytest.raises(ValueError, match="chunk_size must be >="):
            config.validate(Modality.TEXT)

    def test_validate_text_invalid_overlap(self):
        """Validation fails when overlap >= chunk_size."""
        config = ChunkerConfig(name="semantic", chunk_size=200, chunk_overlap=200)
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            config.validate(Modality.TEXT)


class TestExtractionPlan:
    """Tests for ExtractionPlan."""

    def test_basic_creation(self):
        """Create basic extraction plan."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        assert plan.extractor.name == "text"
        assert plan.chunker.name == "semantic"

    def test_default_values(self):
        """Check default values."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        assert plan.num_passes == 1
        assert plan.include_confidence is True
        assert plan.include_citations is True
        assert plan.retry_on_failure is True
        assert plan.max_retries == 3
        assert plan.backoff_factor == 2.0

    def test_custom_values(self):
        """Create plan with custom values."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            num_passes=3,
            include_confidence=False,
            max_retries=5,
        )
        assert plan.num_passes == 3
        assert plan.include_confidence is False
        assert plan.max_retries == 5

    def test_validate_invalid_num_passes(self):
        """Validation fails for num_passes < 1."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            num_passes=0,
        )
        with pytest.raises(ValueError, match="num_passes must be >= 1"):
            plan.validate()

    def test_validate_invalid_backoff(self):
        """Validation fails for backoff_factor < 1."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
            backoff_factor=0.5,
        )
        with pytest.raises(ValueError, match="backoff_factor must be >= 1"):
            plan.validate()
