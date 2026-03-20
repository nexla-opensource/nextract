"""Mixin classes for fallback-aware generation in extractors.

This module provides reusable mixins that handle provider fallback logic,
eliminating code duplication across VLM, Text, and OCR extractors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from nextract.core import ExtractorConfig, ProviderRequest

log = structlog.get_logger(__name__)

try:
    from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

    _LLM_RETRYABLE: tuple[type[Exception], ...] = (
        ModelHTTPError,
        UnexpectedModelBehavior,
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
    )
except ImportError:
    _LLM_RETRYABLE = (TimeoutError, ConnectionError, ConnectionResetError, BrokenPipeError)


class FallbackMixin:
    """Mixin providing fallback-aware generation for LLM extractors.

    This mixin handles the common pattern of trying a primary provider
    and falling back to a secondary provider on specific exceptions.

    Attributes:
        RETRYABLE_EXCEPTIONS: tuple of exception types that trigger fallback.
        config: ExtractorConfig instance (expected to be set by subclass).
    """

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = _LLM_RETRYABLE

    # Will be set by the extractor subclass
    config: "ExtractorConfig | None"

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        """Get retryable exceptions."""
        return cls.RETRYABLE_EXCEPTIONS

    def _safe_generate(self, provider: Any, request: "ProviderRequest") -> Any:
        """Generate with fallback provider support.

        Args:
            provider: The primary provider instance.
            request: The provider request to execute.

        Returns:
            ProviderResponse from the successful provider.

        Raises:
            The original exception if no fallback is configured or fallback fails.
        """
        try:
            return provider.generate(request)
        except self._get_retryable_exceptions() as exc:
            return self._try_fallback(request, exc)

    def _try_fallback(self, request: "ProviderRequest", exc: Exception) -> Any:
        """Attempt to use fallback provider if configured.

        Args:
            request: The provider request to execute.
            exc: The exception that triggered the fallback.

        Returns:
            ProviderResponse from the fallback provider.

        Raises:
            The original exception if no fallback is configured or available.
        """
        from nextract.registry import ProviderRegistry

        if not self.config or not self.config.fallback_provider:
            raise exc

        fallback_class = ProviderRegistry.get_instance().get(
            self.config.fallback_provider.name
        )
        if not fallback_class:
            raise exc

        fallback = fallback_class()
        fallback.initialize(self.config.fallback_provider)

        log.warning(
            "extractor_fallback_triggered",
            extractor=getattr(self, "name", self.__class__.__name__),
            error=str(exc),
            error_type=type(exc).__name__,
            fallback_provider=self.config.fallback_provider.name,
        )

        return fallback.generate(request)


class OCRFallbackMixin(FallbackMixin):
    """Fallback mixin specialized for OCR extractors.

    OCR providers are not pydantic-ai based, so they don't throw
    ModelHTTPError or UnexpectedModelBehavior exceptions.
    """

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
        RuntimeError,
    )

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        """Return OCR-specific retryable exceptions."""
        return cls.RETRYABLE_EXCEPTIONS
