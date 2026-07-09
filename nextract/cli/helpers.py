"""Shared CLI helpers."""

from __future__ import annotations

# Extractor → default chunker (modality-compatible).
_EXTRACTOR_DEFAULT_CHUNKERS: dict[str, str] = {
    "vlm": "page",
    "ocr": "page",
    "textract": "page",
    "text": "semantic",
    "llamaindex": "semantic",
    "hybrid": "hybrid",
    # "auto" is resolved by the extract command before chunker lookup.
}

_FALLBACK_DEFAULT_CHUNKER = "semantic"


def default_chunker_for_extractor(extractor: str) -> str:
    """Return the modality-compatible default chunker for an extractor name.

    Mapping:
      - vlm / ocr / textract → page
      - text / llamaindex → semantic
      - hybrid → hybrid
      - unknown → semantic
    """
    return _EXTRACTOR_DEFAULT_CHUNKERS.get(extractor.lower().strip(), _FALLBACK_DEFAULT_CHUNKER)
