"""Shared Textract utilities to avoid circular imports.

These functions are used by both TextractExtractor and TextractProvider,
and live in a separate module to break the circular dependency between
nextract.extractors and nextract.providers.
"""

from __future__ import annotations

import os
from typing import Any

from nextract.core.exceptions import DocumentTooLargeError, UnsupportedDocumentError

# Textract sync Bytes API limits
TEXTRACT_SYNC_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def validate_textract_sync_input(data: bytes) -> None:
    """Validate that bytes are suitable for Textract sync API."""
    if len(data) > TEXTRACT_SYNC_MAX_BYTES:
        raise DocumentTooLargeError(
            f"Document is {len(data)} bytes, exceeding Textract sync API "
            f"limit of {TEXTRACT_SYNC_MAX_BYTES} bytes. "
            f"Use S3 + StartDocumentAnalysis for larger documents.",
            service="textract",
            size_bytes=len(data),
            max_bytes=TEXTRACT_SYNC_MAX_BYTES,
        )

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return  # PNG OK
    if data[:2] == b"\xff\xd8":
        return  # JPEG OK

    raise UnsupportedDocumentError(
        "Textract sync API only supports PNG and JPEG. "
        "For PDF/TIFF/multi-page documents, use S3 + StartDocumentAnalysis.",
        service="textract",
        format="unknown",
        supported_formats=["image/png", "image/jpeg"],
    )


def resolve_textract_client_kwargs(extractor_params: dict[str, Any]) -> dict[str, str]:
    """Resolve AWS client kwargs from config params and env vars."""
    client_kwargs: dict[str, str] = {}
    value_sources = {
        "aws_access_key_id": (extractor_params.get("aws_access_key"), os.environ.get("AWS_ACCESS_KEY_ID")),
        "aws_secret_access_key": (
            extractor_params.get("aws_secret_key"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
        ),
        "region_name": (
            extractor_params.get("region"),
            os.environ.get("AWS_DEFAULT_REGION"),
            os.environ.get("AWS_REGION"),
        ),
    }

    for client_key, candidates in value_sources.items():
        value = next((candidate for candidate in candidates if candidate), None)
        if value is not None:
            client_kwargs[client_key] = value

    return client_kwargs


def extract_text_from_textract_response(response: dict[str, Any]) -> str:
    """Extract text content from Textract analyze_document response."""
    blocks = response.get("Blocks", [])
    text_lines: list[str] = []
    for block in blocks:
        if block.get("BlockType") == "LINE" and block.get("Text"):
            text_lines.append(block["Text"])
    return "\n".join(text_lines)
