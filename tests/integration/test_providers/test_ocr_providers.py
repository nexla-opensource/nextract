"""
Integration tests for OCR providers (Tesseract, EasyOCR, PaddleOCR).
"""

import pytest

import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig
from nextract.registry import ProviderRegistry


@pytest.mark.integration
class TestTesseractProviderUnit:
    """Unit tests for Tesseract OCR provider."""

    def test_provider_registered(self):
        """Tesseract provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "tesseract" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("tesseract")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="tesseract", model="default")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
class TestEasyOCRProviderUnit:
    """Unit tests for EasyOCR provider."""

    def test_provider_registered(self):
        """EasyOCR provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "easyocr" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("easyocr")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="easyocr", model="default")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
class TestPaddleOCRProviderUnit:
    """Unit tests for PaddleOCR provider."""

    def test_provider_registered(self):
        """PaddleOCR provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "paddleocr" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("paddleocr")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="paddleocr", model="default")
        provider.initialize(config)
        
        assert provider.config is not None
