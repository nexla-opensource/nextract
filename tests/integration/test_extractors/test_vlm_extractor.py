"""
Integration tests for VLMExtractor.
"""

import pytest


import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401
from nextract.core import ExtractorConfig, Modality, ProviderConfig
from nextract.registry import ExtractorRegistry


@pytest.mark.integration
class TestVLMExtractorUnit:
    """Unit tests for VLMExtractor."""

    def test_extractor_registered(self):
        """VLMExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "vlm" in registry.list_extractors()

    def test_modality_is_visual(self):
        """VLMExtractor should have VISUAL modality."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        assert extractor_class is not None
        assert extractor_class.get_modality() == Modality.VISUAL

    def test_supported_providers(self):
        """VLMExtractor should support vision-capable providers."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        supported = extractor_class.get_supported_providers()
        
        assert "openai" in supported
        assert "anthropic" in supported
        assert "google" in supported

    def test_initialization(self):
        """Extractor initializes correctly."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        )
        extractor.initialize(config)
        
        assert extractor.config is not None

    def test_incompatible_provider_rejected(self):
        """Non-vision providers should be rejected."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        extractor = extractor_class()
        
        config = ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="cohere", model="command-r"),
        )
        
        with pytest.raises(ValueError, match="does not support"):
            extractor.initialize(config)


@pytest.mark.integration
class TestVLMExtractorModality:
    """Tests for VLM extractor modality requirements."""

    def test_vlm_requires_visual_modality(self):
        """VLM extractor should require visual modality."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        assert extractor_class.get_modality() == Modality.VISUAL

    def test_vlm_with_openai_valid(self):
        """VLM with OpenAI should be valid."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        assert "openai" in extractor_class.get_supported_providers()

    def test_vlm_with_anthropic_valid(self):
        """VLM with Anthropic should be valid."""
        extractor_class = ExtractorRegistry.get_instance().get("vlm")
        assert "anthropic" in extractor_class.get_supported_providers()
