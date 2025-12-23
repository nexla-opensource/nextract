"""
Integration tests for Anthropic provider.

Tests real API interactions when ANTHROPIC_API_KEY is available.
"""

import pytest

from tests.integration.conftest import requires_anthropic

import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig, ProviderRequest
from nextract.registry import ProviderRegistry


@pytest.mark.integration
class TestAnthropicProviderUnit:
    """Unit tests that don't require credentials."""

    def test_provider_registered(self):
        """Anthropic provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "anthropic" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022")
        provider.initialize(config)
        
        assert provider.config is not None

    def test_supports_vision(self):
        """Anthropic provider should support vision."""
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022"))
        
        assert provider.supports_vision() is True

    def test_supports_structured_output(self):
        """Anthropic provider should support structured output."""
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022"))
        
        assert provider.supports_structured_output() is True


@pytest.mark.integration
@requires_anthropic
class TestAnthropicProviderLive:
    """Live tests requiring Anthropic API credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation with real API."""
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="anthropic", model="claude-3-5-haiku-20241022"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data from the text."},
                {"role": "user", "content": [{"type": "text", "text": "Invoice INV-001, total $500"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        
        assert response is not None
        assert response.usage is not None

    @pytest.mark.slow
    def test_generate_with_structured_output(self, simple_schema):
        """Test structured output extraction."""
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="anthropic", model="claude-3-5-haiku-20241022"))
        
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
