from __future__ import annotations

import os
from typing import Any

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality
from nextract.registry import register_extractor


@register_extractor("textract")
class TextractExtractor(BaseExtractor):
    """Extractor using AWS Textract."""

    SUPPORTED_PROVIDERS = ["aws"]

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

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("boto3 required for Textract. Install with: pip install boto3") from exc

        client = boto3.client(
            "textract",
            **self._resolve_client_kwargs(),
        )

        results: list[dict[str, Any]] = []

        for idx, chunk in enumerate(input_data):
            document_bytes = self._get_chunk_bytes(chunk)
            metadata = getattr(chunk, "metadata", {})

            if not document_bytes:
                results.append(
                    {
                        "chunk_id": getattr(chunk, "id", f"chunk_{idx}"),
                        "response": {"error": "Unsupported chunk type for Textract"},
                        "metadata": metadata,
                    }
                )
                continue

            try:
                response = client.analyze_document(
                    Document={"Bytes": document_bytes},
                    FeatureTypes=["TABLES", "FORMS"],
                )
            except Exception as exc:  # noqa: BLE001
                response = {"error": str(exc)}

            results.append(
                {
                    "chunk_id": getattr(chunk, "id", f"chunk_{idx}"),
                    "response": response,
                    "metadata": metadata,
                }
            )

        provider_name = getattr(provider, "config", None)
        return ExtractorResult(
            name="textract",
            provider_name=provider_name.name if provider_name else "aws",
            results=results,
            metadata={"modality": "visual", "num_chunks": len(input_data)},
        )

    def _resolve_client_kwargs(self) -> dict[str, str]:
        if not self.config:
            raise ValueError("Extractor not initialized")

        params = self.config.extractor_params
        client_kwargs: dict[str, str] = {}
        value_sources = {
            "aws_access_key_id": (params.get("aws_access_key"), os.environ.get("AWS_ACCESS_KEY_ID")),
            "aws_secret_access_key": (
                params.get("aws_secret_key"),
                os.environ.get("AWS_SECRET_ACCESS_KEY"),
            ),
            "region_name": (
                params.get("region"),
                os.environ.get("AWS_DEFAULT_REGION"),
                os.environ.get("AWS_REGION"),
            ),
        }

        for client_key, candidates in value_sources.items():
            value = next((candidate for candidate in candidates if candidate), None)
            if value is not None:
                client_kwargs[client_key] = value

        return client_kwargs

    def _get_chunk_bytes(self, chunk: Any) -> bytes | None:
        if hasattr(chunk, "content") and isinstance(chunk.content, (bytes, bytearray)):
            return bytes(chunk.content)
        if hasattr(chunk, "images") and getattr(chunk, "images", None):
            image = chunk.images[0]
            if isinstance(image, (bytes, bytearray)):
                return bytes(image)
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Pillow required for Textract image handling. Install with: pip install pillow"
                ) from exc
            if isinstance(image, Image.Image):
                from io import BytesIO

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue()
        return None
