"""
Integration tests for Google provider.

Tests real API interactions when GOOGLE_API_KEY or GEMINI_API_KEY is available.
"""

import pytest

from tests.integration.conftest import requires_google

import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig, ProviderRequest
from nextract.registry import ProviderRegistry


@pytest.mark.integration
class TestGoogleProviderUnit:
    """Unit tests that don't require credentials."""

    def test_provider_registered(self):
        """Google provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "google" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("google")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="google", model="gemini-1.5-flash")
        provider.initialize(config)
        
        assert provider.config is not None

    def test_supports_vision(self):
        """Google provider should support vision."""
        provider_class = ProviderRegistry.get_instance().get("google")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="google", model="gemini-1.5-flash"))
        
        assert provider.supports_vision() is True

    def test_supports_structured_output(self):
        """Google provider should support structured output."""
        provider_class = ProviderRegistry.get_instance().get("google")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="google", model="gemini-1.5-flash"))
        
        assert provider.supports_structured_output() is True


@pytest.mark.integration
@requires_google
class TestGoogleProviderLive:
    """Live tests requiring Google API credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation with real API."""
        provider_class = ProviderRegistry.get_instance().get("google")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="google", model="gemini-1.5-flash"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data from the text."},
                {"role": "user", "content": [{"type": "text", "text": "Invoice INV-001, total $500"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        
        assert response is not None
        assert response.usage is not None or response.text or response.structured_output

    @pytest.mark.slow
    def test_generate_with_structured_output(self, simple_schema):
        """Test structured output extraction."""
        provider_class = ProviderRegistry.get_instance().get("google")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="google", model="gemini-1.5-flash"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract invoice details."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Invoice Number: INV-2024-001\nTotal Amount: $567.00",
                        }
                    ],
                },
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        
        assert response.structured_output is not None or response.text
