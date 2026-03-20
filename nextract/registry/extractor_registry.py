from __future__ import annotations

import threading

from nextract.core.base import BaseExtractor


class ExtractorRegistry:
    """Registry for all extractor implementations."""

    _instance: ExtractorRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._extractors: dict[str, type[BaseExtractor]] = {}

    @classmethod
    def get_instance(cls) -> "ExtractorRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, extractor_class: type[BaseExtractor]) -> None:
        if name in self._extractors:
            import warnings
            warnings.warn(f"Extractor '{name}' already registered; overwriting.", stacklevel=3)
        self._extractors[name] = extractor_class

    def get(self, name: str) -> type[BaseExtractor] | None:
        return self._extractors.get(name)

    def list_extractors(self) -> list[str]:
        return list(self._extractors.keys())

    def get_compatible_providers(self, extractor_name: str) -> list[str]:
        extractor_class = self.get(extractor_name)
        if extractor_class:
            return extractor_class.get_supported_providers()
        return []


def register_extractor(name: str):
    """Decorator to register an extractor."""

    def decorator(cls: type[BaseExtractor]):
        ExtractorRegistry.get_instance().register(name, cls)
        return cls

    return decorator
