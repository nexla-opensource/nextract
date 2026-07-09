"""Unit tests for FallbackMixin provider fallback eligibility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nextract.core import ExtractorConfig, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.core.exceptions import (
    ProviderAuthError,
    ProviderCapabilityError,
    ProviderRequestError,
)
from nextract.extractors.fallback_mixin import FallbackMixin


class _DummyExtractor(FallbackMixin):
    """Minimal host for FallbackMixin methods."""

    name = "dummy"

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config


def _request() -> ProviderRequest:
    return ProviderRequest(messages=[{"role": "user", "content": "hi"}])


def _config_with_fallback() -> ExtractorConfig:
    return ExtractorConfig(
        name="text",
        provider=ProviderConfig(name="openai", model="gpt-4o"),
        fallback_provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022"),
    )


def test_is_fallback_eligible_provider_request_error_retryable():
    host = _DummyExtractor()
    exc = ProviderRequestError("server error", provider="openai", retryable=True)
    assert host._is_fallback_eligible(exc) is True


def test_is_fallback_eligible_provider_request_error_not_retryable():
    host = _DummyExtractor()
    exc = ProviderRequestError("bad request", provider="openai", retryable=False)
    assert host._is_fallback_eligible(exc) is False


def test_is_fallback_eligible_auth_error_never():
    host = _DummyExtractor()
    exc = ProviderAuthError("invalid key", provider="openai", status_code=401)
    assert host._is_fallback_eligible(exc) is False


def test_is_fallback_eligible_capability_error_never():
    host = _DummyExtractor()
    exc = ProviderCapabilityError("no vision", provider="openai")
    assert host._is_fallback_eligible(exc) is False


def test_is_fallback_eligible_timeout_still_retryable():
    host = _DummyExtractor()
    assert host._is_fallback_eligible(TimeoutError("timed out")) is True


def test_safe_generate_falls_back_on_retryable_provider_request_error():
    host = _DummyExtractor(_config_with_fallback())
    primary = MagicMock()
    primary.generate.side_effect = ProviderRequestError(
        "upstream 503",
        provider="openai",
        status_code=503,
        retryable=True,
    )

    fallback_response = ProviderResponse(text="ok", structured_output={"x": 1})
    fallback_instance = MagicMock()
    fallback_instance.generate.return_value = fallback_response
    fallback_class = MagicMock(return_value=fallback_instance)

    with patch(
        "nextract.registry.ProviderRegistry.get_instance"
    ) as get_registry:
        registry = MagicMock()
        registry.get.return_value = fallback_class
        get_registry.return_value = registry

        response = host._safe_generate(primary, _request())

    assert response is fallback_response
    primary.generate.assert_called_once()
    fallback_instance.initialize.assert_called_once()
    fallback_instance.generate.assert_called_once()


def test_safe_generate_does_not_fall_back_on_auth_error():
    host = _DummyExtractor(_config_with_fallback())
    primary = MagicMock()
    auth_err = ProviderAuthError("bad creds", provider="openai", status_code=401)
    primary.generate.side_effect = auth_err

    with patch(
        "nextract.registry.ProviderRegistry.get_instance"
    ) as get_registry:
        with pytest.raises(ProviderAuthError):
            host._safe_generate(primary, _request())
        get_registry.assert_not_called()


def test_safe_generate_does_not_fall_back_on_non_retryable_request_error():
    host = _DummyExtractor(_config_with_fallback())
    primary = MagicMock()
    primary.generate.side_effect = ProviderRequestError(
        "client error",
        provider="openai",
        status_code=400,
        retryable=False,
    )

    with patch(
        "nextract.registry.ProviderRegistry.get_instance"
    ) as get_registry:
        with pytest.raises(ProviderRequestError):
            host._safe_generate(primary, _request())
        get_registry.assert_not_called()


def test_safe_generate_reraises_retryable_when_no_fallback_configured():
    host = _DummyExtractor(
        ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
            fallback_provider=None,
        )
    )
    primary = MagicMock()
    primary.generate.side_effect = ProviderRequestError(
        "upstream 503",
        provider="openai",
        retryable=True,
    )

    with pytest.raises(ProviderRequestError):
        host._safe_generate(primary, _request())
