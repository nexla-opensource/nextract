from __future__ import annotations

from typing import Any

from nextract.core import BaseExtractor, ExtractorConfig, ExtractorResult, Modality


class CustomExtractorTemplate(BaseExtractor):
    """Template for custom extractor implementations."""

    SUPPORTED_PROVIDERS: list[str] = []

    def __init__(self) -> None:
        self.config: ExtractorConfig | None = None

    def initialize(self, config: ExtractorConfig) -> None:
        self.config = config

    @classmethod
    def get_modality(cls) -> Modality:
        return Modality.TEXT

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        return cls.SUPPORTED_PROVIDERS

    def validate_config(self, config: ExtractorConfig) -> bool:
        return True

    def run(self, input_data: Any, provider: Any, **kwargs: Any) -> ExtractorResult:
        raise NotImplementedError
