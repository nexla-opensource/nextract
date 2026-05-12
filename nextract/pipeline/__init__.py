"""Pipeline orchestration layer."""

from .orchestrator import BatchExtractionResult, BatchPipeline, ExtractionPipeline

__all__ = [
    "BatchExtractionResult",
    "BatchPipeline",
    "ExtractionPipeline",
]


def __getattr__(name: str):
    """Lazy-load deprecated exports with warnings."""
    if name == "PipelineRouter":
        import warnings
        warnings.warn(
            "PipelineRouter is a stub and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .router import PipelineRouter
        return PipelineRouter
    if name == "RetryPolicy":
        import warnings
        warnings.warn(
            "RetryPolicy is unused and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .retries import RetryPolicy
        return RetryPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
