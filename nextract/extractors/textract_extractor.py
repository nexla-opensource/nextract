from __future__ import annotations

from typing import Any, Dict, List, Optional

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality
from nextract.registry import register_extractor


@register_extractor("textract")
class TextractExtractor(BaseExtractor):
    """Extractor using AWS Textract."""

    SUPPORTED_PROVIDERS = ["aws"]

    def __init__(self) -> None:
        self.config: Optional[ExtractorConfig] = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config
        self.validate_config(config)

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.VISUAL

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        required_params = ["aws_access_key", "aws_secret_key", "region"]
        for param in required_params:
            if param not in config.extractor_params:
                raise ValueError(f"Textract requires '{param}' in extractor_params")
        return True

    def run(self, input_data: Any, provider: Any, **kwargs: Any) -> ExtractorResult:
        if not self.config:
            raise ValueError("Extractor not initialized")

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("boto3 required for Textract. Install with: pip install boto3") from exc

        params = self.config.extractor_params
        client = boto3.client(
            "textract",
            aws_access_key_id=params["aws_access_key"],
            aws_secret_access_key=params["aws_secret_key"],
            region_name=params["region"],
        )

        results: List[Dict[str, Any]] = []

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

    def _get_chunk_bytes(self, chunk: Any) -> Optional[bytes]:
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
