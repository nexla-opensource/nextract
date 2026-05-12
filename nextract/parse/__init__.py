"""Parse layer: OCR and layout parsing."""

from .text_parser import extract_text
from .ocr_parser import OCRParser

__all__ = ["extract_text", "OCRParser"]


def __getattr__(name: str):
    """Lazy-load deprecated exports with warnings."""
    if name == "LayoutParser":
        import warnings
        warnings.warn(
            "LayoutParser is not implemented and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .layout_parser import LayoutParser
        return LayoutParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
