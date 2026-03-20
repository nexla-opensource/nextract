from __future__ import annotations


class NextractError(Exception):
    """Base exception for nextract errors."""


class PlanError(NextractError):
    """Raised when an extraction plan is invalid."""


class ProviderError(NextractError):
    """Raised when a provider fails."""


class ExtractorError(NextractError):
    """Raised when an extractor fails."""


class ChunkerError(NextractError):
    """Raised when a chunker fails."""


class ValidationError(NextractError):
    """Raised when validation fails."""


class PipelineError(NextractError):
    """Raised when pipeline orchestration fails."""
