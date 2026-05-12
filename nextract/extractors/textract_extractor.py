from __future__ import annotations

import os
from typing import Any

import structlog

from nextract.core import BaseExtractor, ChunkExtraction, ExtractorConfig, ExtractorResult, Modality
from nextract.core.exceptions import DocumentTooLargeError, ProviderError, UnsupportedDocumentError
from nextract.extractors.textract_utils import validate_textract_sync_input
from nextract.providers.ocr_utils import encode_image_to_bytes
from nextract.registry import register_extractor

log = structlog.get_logger(__name__)


@register_extractor("textract")
class TextractExtractor(BaseExtractor):
    """Extractor using AWS Textract.

    Delegates to TextractProvider for the actual API call, avoiding
    code duplication between extractor and provider.
    """

    SUPPORTED_PROVIDERS = ["textract", "aws"]

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.VISUAL

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        params = config.extractor_params
        required = {
            "aws_access_key": ("AWS_ACCESS_KEY_ID",),
            "aws_secret_key": ("AWS_SECRET_ACCESS_KEY",),
            "region": ("AWS_DEFAULT_REGION", "AWS_REGION"),
        }
        missing = []
        for param_key, env_keys in required.items():
            if param_key not in params and not any(os.environ.get(env_key) for env_key in env_keys):
                missing.append(param_key)
        if missing:
            raise ValueError(
                f"Textract requires: {', '.join(missing)} "
                "(via extractor_params or environment variables)"
            )
        return True

    def run(self, input_data: Any, provider: Any, **kwargs: Any) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        results: list[ChunkExtraction] = []

        for idx, chunk in enumerate(input_data):
            document_bytes = self._get_chunk_bytes(chunk)
            metadata = getattr(chunk, "metadata", {})

            if not document_bytes:
                results.append(
                    ChunkExtraction(
                        chunk_id=getattr(chunk, "id", f"chunk_{idx}"),
                        response={},
                        metadata=metadata,
                        error="Unsupported chunk type for Textract",
                    )
                )
                continue

            validate_textract_sync_input(document_bytes)

            # Delegate to the TextractProvider for the actual API call
            from nextract.core import ProviderRequest
            import base64

            images_b64 = [base64.b64encode(document_bytes).decode("ascii")]
            request = ProviderRequest(
                messages=[],
                images=images_b64,
                schema=None,
                options={},
            )
            try:
                response = provider.generate(request)
                results.append(
                    ChunkExtraction(
                        chunk_id=getattr(chunk, "id", f"chunk_{idx}"),
                        response={"text": response.text},
                        metadata=metadata,
                    )
                )
            except (DocumentTooLargeError, UnsupportedDocumentError):
                raise
            except Exception as exc:
                raise ProviderError(
                    f"Textract analyze_document failed: {exc}",
                    provider="textract",
                    retryable=False,
                ) from exc

        return ExtractorResult(
            name="textract",
            provider_name=provider.get_name(),
            results=results,
            metadata={"modality": "visual", "num_chunks": len(input_data)},
        )

    def _get_chunk_bytes(self, chunk: Any) -> bytes | None:
        if hasattr(chunk, "content") and isinstance(chunk.content, (bytes, bytearray)):
            return bytes(chunk.content)
        if hasattr(chunk, "images") and getattr(chunk, "images", None):
            image = chunk.images[0]
            return encode_image_to_bytes(image)
        return None
