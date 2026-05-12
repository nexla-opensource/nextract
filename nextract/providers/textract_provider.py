from __future__ import annotations

from typing import Any

import structlog

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.core.exceptions import ProviderError
from nextract.extractors.textract_utils import (
    extract_text_from_textract_response,
    resolve_textract_client_kwargs,
    validate_textract_sync_input,
)
from nextract.providers.ocr_utils import decode_images, encode_image_to_bytes
from nextract.registry import register_provider

log = structlog.get_logger(__name__)

TEXTRACT_SYNC_SUPPORTED_FORMATS = {"image/png", "image/jpeg", "image/jpg"}


@register_provider("textract")
class TextractProvider(BaseProvider):
    """AWS Textract provider for OCR-based extraction.

    This is NOT a Pydantic AI provider. It wraps boto3's Textract client
    and is used by TextractExtractor for document OCR.

    Validation and client-kwargs logic are shared via
    nextract.extractors.textract_extractor module-level functions.
    """

    def __init__(self) -> None:
        self.name: str = "textract"
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 required for AWS Textract. Install with: pip install boto3"
            ) from exc

        client = boto3.client("textract", **self._resolve_client_kwargs())

        ocr_dpi = int(request.options.get("ocr_dpi", 300))
        images = decode_images(request.images, ocr_dpi=ocr_dpi)

        if not images:
            return ProviderResponse(text="", raw=None)

        text_parts: list[str] = []
        for image in images:
            image_bytes = encode_image_to_bytes(image)
            if image_bytes is None:
                continue

            validate_textract_sync_input(image_bytes)

            try:
                response = client.analyze_document(
                    Document={"Bytes": image_bytes},
                    FeatureTypes=["TABLES", "FORMS"],
                )
                text_parts.append(extract_text_from_textract_response(response))
            except Exception as exc:
                raise ProviderError(
                    f"Textract analyze_document failed: {exc}",
                    provider="textract",
                    retryable=False,
                ) from exc

        return ProviderResponse(text="\n\n".join(text_parts), raw=None)

    def supports_vision(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return False

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "vision": True,
            "structured_output": False,
            "ocr": True,
            "textract": True,
        }

    def _resolve_client_kwargs(self) -> dict[str, str]:
        if not self.config:
            return {}
        return resolve_textract_client_kwargs(self.config.extra_params or {})
