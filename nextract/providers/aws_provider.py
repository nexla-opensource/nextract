from __future__ import annotations

from nextract.providers.pydantic_ai_provider import PydanticAIProvider
from nextract.registry import register_provider


@register_provider("aws")
class AWSProvider(PydanticAIProvider):
    """AWS provider wrapper (Bedrock, Textract)."""

    def supports_vision(self) -> bool:
        if not self.config:
            return False
        model = self.config.model.lower()
        if "textract" in model:
            return True
        return super().supports_vision()

    def get_capabilities(self) -> dict[str, object]:
        capabilities = super().get_capabilities()
        if self.config and "textract" in self.config.model.lower():
            capabilities["textract"] = True
        return capabilities
