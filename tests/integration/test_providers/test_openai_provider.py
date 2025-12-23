"""
Integration tests for OpenAI provider.

Tests real API interactions when OPENAI_API_KEY is available.
"""

import pytest

from tests.integration.conftest import requires_openai

import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig, ProviderRequest
from nextract.registry import ProviderRegistry


@pytest.mark.integration
class TestOpenAIProviderUnit:
    """Unit tests that don't require credentials."""

    def test_provider_registered(self):
        """OpenAI provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "openai" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="openai", model="gpt-4o")
        provider.initialize(config)
        
        assert provider.config is not None
        assert provider.config.model == "gpt-4o"

    def test_supports_vision(self):
        """OpenAI provider should support vision."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o"))
        
        assert provider.supports_vision() is True

    def test_supports_structured_output(self):
        """OpenAI provider should support structured output."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o"))
        
        assert provider.supports_structured_output() is True

    def test_get_capabilities(self):
        """Capabilities dict should include expected keys."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o"))
        
        capabilities = provider.get_capabilities()
        assert "vision" in capabilities
        assert "structured_output" in capabilities
        assert capabilities["vision"] is True
        assert capabilities["structured_output"] is True


@pytest.mark.integration
@requires_openai
class TestOpenAIProviderLive:
    """Live tests requiring OpenAI API credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation with real API."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o-mini"))
        
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
        assert response.structured_output is not None or response.text

    @pytest.mark.slow
    def test_generate_with_structured_output(self, simple_schema):
        """Test structured output extraction."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o-mini"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract invoice details."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Invoice Number: INV-2024-001\nTotal Amount: $567.00\nDate: January 15, 2024",
                        }
                    ],
                },
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        
        assert response.structured_output is not None
        output = response.structured_output
        assert "invoice_number" in output or "total" in output

    def test_usage_tracking(self, simple_schema):
        """Token usage should be tracked."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="openai", model="gpt-4o-mini"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data."},
                {"role": "user", "content": [{"type": "text", "text": "Test"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        
        assert response.usage is not None
        assert "input_tokens" in response.usage
        assert "output_tokens" in response.usage
        assert response.usage["input_tokens"] > 0
