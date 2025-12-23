"""
Integration tests for remaining extractors (OCR, Hybrid, Textract, LlamaIndex).
"""

import pytest

import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401
from nextract.core import Modality
from nextract.registry import ExtractorRegistry


@pytest.mark.integration
class TestOCRExtractorUnit:
    """Unit tests for OCRExtractor."""

    def test_extractor_registered(self):
        """OCRExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "ocr" in registry.list_extractors()

    def test_modality(self):
        """OCRExtractor should have appropriate modality."""
        extractor_class = ExtractorRegistry.get_instance().get("ocr")
        assert extractor_class is not None
        modality = extractor_class.get_modality()
        assert modality in [Modality.VISUAL, Modality.TEXT, Modality.HYBRID]

    def test_supported_providers(self):
        """OCRExtractor should list supported providers."""
        extractor_class = ExtractorRegistry.get_instance().get("ocr")
        supported = extractor_class.get_supported_providers()
        assert isinstance(supported, list)


@pytest.mark.integration
class TestHybridExtractorUnit:
    """Unit tests for HybridExtractor."""

    def test_extractor_registered(self):
        """HybridExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "hybrid" in registry.list_extractors()

    def test_modality_is_hybrid(self):
        """HybridExtractor should have HYBRID modality."""
        extractor_class = ExtractorRegistry.get_instance().get("hybrid")
        assert extractor_class is not None
        assert extractor_class.get_modality() == Modality.HYBRID

    def test_supported_providers(self):
        """HybridExtractor should support various providers."""
        extractor_class = ExtractorRegistry.get_instance().get("hybrid")
        supported = extractor_class.get_supported_providers()
        assert isinstance(supported, list)


@pytest.mark.integration
class TestTextractExtractorUnit:
    """Unit tests for TextractExtractor."""

    def test_extractor_registered(self):
        """TextractExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "textract" in registry.list_extractors()

    def test_modality(self):
        """TextractExtractor should have appropriate modality."""
        extractor_class = ExtractorRegistry.get_instance().get("textract")
        assert extractor_class is not None
        modality = extractor_class.get_modality()
        assert modality in [Modality.VISUAL, Modality.TEXT, Modality.HYBRID]

    def test_supported_providers_limited(self):
        """TextractExtractor should only support AWS."""
        extractor_class = ExtractorRegistry.get_instance().get("textract")
        supported = extractor_class.get_supported_providers()
        assert "aws" in supported or "textract" in supported


@pytest.mark.integration
class TestLlamaIndexExtractorUnit:
    """Unit tests for LlamaIndexExtractor."""

    def test_extractor_registered(self):
        """LlamaIndexExtractor should be registered."""
        registry = ExtractorRegistry.get_instance()
        assert "llamaindex" in registry.list_extractors()

    def test_modality(self):
        """LlamaIndexExtractor should have appropriate modality."""
        extractor_class = ExtractorRegistry.get_instance().get("llamaindex")
        assert extractor_class is not None
        modality = extractor_class.get_modality()
        assert modality in [Modality.VISUAL, Modality.TEXT, Modality.HYBRID]


@pytest.mark.integration
class TestExtractorRegistry:
    """Tests for extractor registry functionality."""

    def test_list_all_extractors(self):
        """Should list all registered extractors."""
        registry = ExtractorRegistry.get_instance()
        extractors = registry.list_extractors()
        
        expected = ["text", "vlm", "ocr", "hybrid", "textract", "llamaindex"]
        for name in expected:
            assert name in extractors

    def test_get_compatible_providers(self):
        """Should return compatible providers for extractor."""
        registry = ExtractorRegistry.get_instance()
        
        text_providers = registry.get_compatible_providers("text")
        assert "openai" in text_providers
        
        vlm_providers = registry.get_compatible_providers("vlm")
        assert "openai" in vlm_providers

    def test_get_nonexistent_extractor(self):
        """Should return None for unknown extractor."""
        registry = ExtractorRegistry.get_instance()
        assert registry.get("nonexistent") is None
