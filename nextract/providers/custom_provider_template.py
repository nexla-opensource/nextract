from __future__ import annotations

from typing import Any

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse


class CustomProviderTemplate(BaseProvider):
    """Template for custom provider implementations."""

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    def supports_vision(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return True

    def get_capabilities(self) -> dict[str, Any]:
        return {"vision": self.supports_vision(), "structured_output": True}
