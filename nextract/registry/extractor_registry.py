from __future__ import annotations

from typing import Dict, List, Optional, Type

from nextract.core.base import BaseExtractor


class ExtractorRegistry:
    """Registry for all extractor implementations."""

    _instance: Optional["ExtractorRegistry"] = None
    _extractors: Dict[str, Type[BaseExtractor]] = {}

    @classmethod
    def get_instance(cls) -> "ExtractorRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, extractor_class: Type[BaseExtractor]) -> None:
        self._extractors[name] = extractor_class

    def get(self, name: str) -> Optional[Type[BaseExtractor]]:
        return self._extractors.get(name)

    def list_extractors(self) -> List[str]:
        return list(self._extractors.keys())

    def get_compatible_providers(self, extractor_name: str) -> List[str]:
        extractor_class = self.get(extractor_name)
        if extractor_class:
            return extractor_class.get_supported_providers()
        return []


def register_extractor(name: str):
    """Decorator to register an extractor."""

    def decorator(cls: Type[BaseExtractor]):
        ExtractorRegistry.get_instance().register(name, cls)
        return cls

    return decorator
