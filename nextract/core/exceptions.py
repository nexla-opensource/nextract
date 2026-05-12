from __future__ import annotations

from typing import Any


class NextractError(Exception):
    """Base exception for nextract errors."""

    def __init__(self, message: str = "", *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class PlanError(NextractError):
    """Raised when an extraction plan is invalid."""


class ProviderError(NextractError):
    """Raised when a provider fails."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool = False,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "retryable": retryable,
        }
        if status_code is not None:
            full_context["status_code"] = status_code
        if request_id is not None:
            full_context["request_id"] = request_id
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        self.cause = cause


class ExtractorError(NextractError):
    """Raised when an extractor fails."""

    def __init__(
        self,
        message: str = "",
        *,
        extractor: str = "",
        chunk_id: str | None = None,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {"extractor": extractor}
        if chunk_id is not None:
            full_context["chunk_id"] = chunk_id
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.extractor = extractor
        self.chunk_id = chunk_id
        self.cause = cause


class ChunkerError(NextractError):
    """Raised when a chunker fails."""

    def __init__(
        self,
        message: str = "",
        *,
        chunker: str = "",
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {"chunker": chunker}
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.chunker = chunker
        self.cause = cause


class ValidationError(NextractError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str = "",
        *,
        field: str | None = None,
        schema_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {}
        if field is not None:
            full_context["field"] = field
        if schema_path is not None:
            full_context["schema_path"] = schema_path
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.field = field
        self.schema_path = schema_path


class PipelineError(NextractError):
    """Raised when pipeline orchestration fails."""

    def __init__(
        self,
        message: str = "",
        *,
        stage: str = "",
        document: str | None = None,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {"stage": stage}
        if document is not None:
            full_context["document"] = document
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.stage = stage
        self.document = document
        self.cause = cause


class ProviderAuthError(ProviderError):
    """Raised when provider authentication fails (invalid or missing credentials)."""


class ProviderRequestError(ProviderError):
    """Raised when a provider request fails due to invalid input or server error."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        retryable: bool = False,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            retryable=retryable,
            cause=cause,
            context=context,
        )


class ProviderCapabilityError(ProviderError):
    """Raised when a provider lacks a required capability (e.g., vision, structured output)."""


class DocumentTooLargeError(NextractError):
    """Raised when a document exceeds size limits for the target service."""

    def __init__(
        self,
        message: str = "",
        *,
        service: str = "",
        size_bytes: int | None = None,
        max_bytes: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {"service": service}
        if size_bytes is not None:
            full_context["size_bytes"] = size_bytes
        if max_bytes is not None:
            full_context["max_bytes"] = max_bytes
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.service = service
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class UnsupportedDocumentError(NextractError):
    """Raised when a document format is not supported by the target service."""

    def __init__(
        self,
        message: str = "",
        *,
        service: str = "",
        format: str = "",
        supported_formats: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        full_context: dict[str, Any] = {"service": service, "format": format}
        if supported_formats is not None:
            full_context["supported_formats"] = supported_formats
        if context:
            full_context.update(context)
        super().__init__(message, context=full_context)
        self.service = service
        self.format = format
        self.supported_formats = supported_formats
