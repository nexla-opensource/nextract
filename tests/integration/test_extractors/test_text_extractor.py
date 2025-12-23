"""
Integration tests for TextExtractor.
"""

import pytest

from tests.integration.conftest import requires_openai

import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401
from nextract.core import ExtractorConfig, Modality, ProviderConfig, TextChunk
from nextract.registry import ExtractorRegistry, ProviderRegistry


@pytest.mark.integration
class TestTextExtractorUnit:
    """Unit tests for TextExtractor."""

    def test_extractor_registered(self):
        """TextExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "text" in registry.list_extractors()

    def test_modality_is_text(self):
        """TextExtractor should have TEXT modality."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        assert extractor_class is not None
        assert extractor_class.get_modality() == Modality.TEXT

    def test_supported_providers(self):
        """TextExtractor should support common LLM providers."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        supported = extractor_class.get_supported_providers()
        
        assert "openai" in supported
        assert "anthropic" in supported
        assert "google" in supported

    def test_initialization(self):
        """Extractor initializes correctly."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        )
        extractor.initialize(config)
        
        assert extractor.config is not None

    def test_validate_config(self):
        """Config validation should pass for valid config."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        )
        
        assert extractor.validate_config(config) is True


@pytest.mark.integration
@requires_openai
class TestTextExtractorLive:
    """Live tests requiring OpenAI credentials."""

    def test_run_extraction(self, simple_schema, sample_text_content):
        """Test extraction with real provider."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
        )
        extractor.initialize(config)
        
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(config.provider)
        
        chunks = [
            TextChunk(
                id="chunk_0",
                text=sample_text_content,
                source_path="test.txt",
                metadata={"page": 1},
            )
        ]
        
        result = extractor.run(
            input_data=chunks,
            provider=provider,
            prompt="Extract invoice details",
            schema=simple_schema,
        )
        
        assert result is not None
        assert result.name == "text"
        assert len(result.results) > 0
        assert result.results[0]["response"] is not None

    @pytest.mark.slow
    def test_run_with_multiple_chunks(self, simple_schema, sample_text_content):
        """Test extraction across multiple chunks."""
        extractor_class = ExtractorRegistry.get_instance().get("text")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
        )
        extractor.initialize(config)
        
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(config.provider)
        
        chunks = [
            TextChunk(
                id=f"chunk_{i}",
                text=f"Part {i}: {sample_text_content[:100]}",
                source_path="test.txt",
                metadata={"page": i},
            )
            for i in range(3)
        ]
        
        result = extractor.run(
            input_data=chunks,
            provider=provider,
            prompt="Extract any invoice details",
            schema=simple_schema,
        )
        
        assert len(result.results) == 3
        for r in result.results:
            assert "chunk_id" in r
            assert "usage" in r
