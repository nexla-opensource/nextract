"""
Consolidated provider tests for Azure, AWS, Cohere, and Local providers.
"""

import pytest

from tests.integration.conftest import requires_azure, requires_aws, requires_cohere

import nextract.providers  # noqa: F401
from nextract.core import ProviderConfig, ProviderRequest
from nextract.registry import ProviderRegistry


@pytest.mark.integration
class TestAzureProviderUnit:
    """Unit tests for Azure provider."""

    def test_provider_registered(self):
        """Azure provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "azure" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("azure")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="azure", model="gpt-4o")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
class TestAWSProviderUnit:
    """Unit tests for AWS provider."""

    def test_provider_registered(self):
        """AWS provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "aws" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("aws")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="aws", model="anthropic.claude-3-sonnet-20240229-v1:0")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
class TestCohereProviderUnit:
    """Unit tests for Cohere provider."""

    def test_provider_registered(self):
        """Cohere provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "cohere" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("cohere")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="cohere", model="command-r")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
class TestLocalProviderUnit:
    """Unit tests for Local (Ollama) provider."""

    def test_provider_registered(self):
        """Local provider should be registered."""
        registry = ProviderRegistry.get_instance()
        assert "local" in registry.list_providers()

    def test_provider_initialization(self):
        """Provider initializes with valid config."""
        provider_class = ProviderRegistry.get_instance().get("local")
        assert provider_class is not None
        
        provider = provider_class()
        config = ProviderConfig(name="local", model="llama3.2")
        provider.initialize(config)
        
        assert provider.config is not None


@pytest.mark.integration
@requires_azure
class TestAzureProviderLive:
    """Live tests requiring Azure credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation."""
        provider_class = ProviderRegistry.get_instance().get("azure")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="azure", model="gpt-4o"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data."},
                {"role": "user", "content": [{"type": "text", "text": "Invoice INV-001, total $500"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        assert response is not None


@pytest.mark.integration
@requires_aws
class TestAWSProviderLive:
    """Live tests requiring AWS credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation."""
        provider_class = ProviderRegistry.get_instance().get("aws")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="aws", model="anthropic.claude-3-haiku-20240307-v1:0"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data."},
                {"role": "user", "content": [{"type": "text", "text": "Invoice INV-001, total $500"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        assert response is not None


@pytest.mark.integration
@requires_cohere
class TestCohereProviderLive:
    """Live tests requiring Cohere credentials."""

    def test_generate_simple_text(self, simple_schema):
        """Test basic text generation."""
        provider_class = ProviderRegistry.get_instance().get("cohere")
        provider = provider_class()
        provider.initialize(ProviderConfig(name="cohere", model="command-r"))
        
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "Extract data."},
                {"role": "user", "content": [{"type": "text", "text": "Invoice INV-001, total $500"}]},
            ],
            schema=simple_schema,
        )
        
        response = provider.generate(request)
        assert response is not None
