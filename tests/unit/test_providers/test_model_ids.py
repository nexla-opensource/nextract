"""Unit tests for provider model ID resolution and credential wiring."""

from __future__ import annotations

import pytest

from nextract.core import ProviderConfig
from nextract.registry import ProviderRegistry
from nextract.registry.bootstrap import ensure_plugins_loaded
from nextract.providers.pydantic_ai_provider import PROVIDER_PREFIX_MAP, OCR_PROVIDERS

ensure_plugins_loaded()


class TestModelIdResolution:
    """Test that provider names map to correct Pydantic AI model ID prefixes."""

    @pytest.mark.parametrize(
        "provider_name,model,expected_prefix",
        [
            ("openai", "gpt-4o", "openai:gpt-4o"),
            ("anthropic", "claude-sonnet-4-20250514", "anthropic:claude-sonnet-4-20250514"),
            ("google", "gemini-2.0-flash", "google-gla:gemini-2.0-flash"),
            ("aws", "anthropic.claude-3", "bedrock:anthropic.claude-3"),
            ("bedrock", "anthropic.claude-3", "bedrock:anthropic.claude-3"),
            ("local", "llama3", "ollama:llama3"),
            ("azure", "gpt-4o", "azure:gpt-4o"),
            ("cohere", "command-r-plus", "cohere:command-r-plus"),
        ],
    )
    def test_model_id_prefix(self, provider_name: str, model: str, expected_prefix: str) -> None:
        """Each provider must resolve to the correct Pydantic AI model ID prefix."""
        provider_class = ProviderRegistry.get_instance().get(provider_name)
        assert provider_class is not None, f"Provider '{provider_name}' not registered"

        instance = provider_class()
        config = ProviderConfig(name=provider_name, model=model)
        instance.initialize(config)

        resolved_id = instance._resolve_model_id()
        assert resolved_id == expected_prefix, (
            f"Provider '{provider_name}' resolved to '{resolved_id}', "
            f"expected '{expected_prefix}'"
        )

    def test_google_vertex_backend_override(self) -> None:
        """Google provider with backend='google-vertex' must use google-vertex prefix."""
        provider_class = ProviderRegistry.get_instance().get("google")
        instance = provider_class()
        config = ProviderConfig(
            name="google",
            model="gemini-2.0-flash",
            extra_params={"backend": "google-vertex"},
        )
        instance.initialize(config)
        resolved_id = instance._resolve_model_id()
        assert resolved_id == "google-vertex:gemini-2.0-flash"

    def test_colon_in_model_passthrough(self) -> None:
        """If model already contains a colon, it should be used as-is."""
        provider_class = ProviderRegistry.get_instance().get("openai")
        instance = provider_class()
        config = ProviderConfig(name="openai", model="openai:gpt-4o-mini")
        instance.initialize(config)
        resolved_id = instance._resolve_model_id()
        assert resolved_id == "openai:gpt-4o-mini"


class TestProviderPrefixMap:
    """Test the static prefix map is correct."""

    def test_prefix_map_keys(self) -> None:
        expected_keys = {"openai", "anthropic", "google", "google-vertex", "azure", "cohere", "aws", "bedrock", "local"}
        assert expected_keys <= set(PROVIDER_PREFIX_MAP.keys())

    def test_prefix_map_values(self) -> None:
        assert PROVIDER_PREFIX_MAP["google"] == "google-gla"
        assert PROVIDER_PREFIX_MAP["aws"] == "bedrock"
        assert PROVIDER_PREFIX_MAP["local"] == "ollama"
        assert PROVIDER_PREFIX_MAP["bedrock"] == "bedrock"


class TestProviderRequiredEnvVars:
    """Test that providers report correct required env vars."""

    def test_openai_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("openai")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="openai", model="gpt-4o"))
        env_vars = instance.get_required_env_vars()
        assert "OPENAI_API_KEY" in env_vars

    def test_anthropic_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("anthropic")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="anthropic", model="claude-sonnet-4-20250514"))
        env_vars = instance.get_required_env_vars()
        assert "ANTHROPIC_API_KEY" in env_vars

    def test_google_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("google")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="google", model="gemini-2.0-flash"))
        env_vars = instance.get_required_env_vars()
        assert "GOOGLE_API_KEY" in env_vars

    def test_azure_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("azure")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="azure", model="gpt-4o"))
        env_vars = instance.get_required_env_vars()
        assert "AZURE_OPENAI_API_KEY" in env_vars
        assert "AZURE_OPENAI_ENDPOINT" in env_vars

    def test_cohere_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("cohere")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="cohere", model="command-r-plus"))
        env_vars = instance.get_required_env_vars()
        assert "CO_API_KEY" in env_vars

    def test_aws_bedrock_env(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("bedrock")
        instance = provider_class()
        instance.initialize(ProviderConfig(name="bedrock", model="anthropic.claude-3"))
        env_vars = instance.get_required_env_vars()
        assert "AWS_ACCESS_KEY_ID" in env_vars
        assert "AWS_SECRET_ACCESS_KEY" in env_vars


class TestOCRProviderVisionCapabilities:
    """Test that OCR providers correctly report vision capabilities."""

    @pytest.mark.parametrize("provider_name", ["tesseract", "easyocr", "paddleocr"])
    def test_ocr_provider_supports_vision(self, provider_name: str) -> None:
        provider_class = ProviderRegistry.get_instance().get(provider_name)
        assert provider_class is not None
        instance = provider_class()
        instance.initialize(ProviderConfig(name=provider_name, model="default"))
        assert instance.supports_vision() is True

    @pytest.mark.parametrize("provider_name", ["tesseract", "easyocr", "paddleocr"])
    def test_ocr_provider_no_structured_output(self, provider_name: str) -> None:
        provider_class = ProviderRegistry.get_instance().get(provider_name)
        assert provider_class is not None
        instance = provider_class()
        instance.initialize(ProviderConfig(name=provider_name, model="default"))
        assert instance.supports_structured_output() is False

    def test_textract_provider_supports_vision(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("textract")
        assert provider_class is not None
        instance = provider_class()
        instance.initialize(ProviderConfig(name="textract", model="default"))
        assert instance.supports_vision() is True


class TestAwsDeprecation:
    """Test that 'aws' provider is a deprecated alias for 'bedrock'."""

    def test_aws_resolves_to_bedrock_prefix(self) -> None:
        provider_class = ProviderRegistry.get_instance().get("aws")
        assert provider_class is not None
        instance = provider_class()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            instance.initialize(ProviderConfig(name="aws", model="anthropic.claude-3"))
        resolved_id = instance._resolve_model_id()
        assert resolved_id.startswith("bedrock:")
