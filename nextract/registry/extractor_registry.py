from __future__ import annotations

from nextract.core.base import BaseExtractor
from nextract.registry.base import Registry


class ExtractorRegistry(Registry[BaseExtractor]):
    """Registry for all extractor implementations."""

    def __init__(self) -> None:
        super().__init__(kind="extractor")

    def list_extractors(self) -> list[str]:
        return self.list_items()

    def get_compatible_providers(self, extractor_name: str) -> list[str]:
        extractor_class = self.get(extractor_name)
        if extractor_class:
            return extractor_class.get_supported_providers()
        return []


def register_extractor(name: str, *, overwrite: bool = False):
    """Decorator to register an extractor.

    Args:
        name: Unique name for the extractor.
        overwrite: If True, silently replace any existing registration.
            Defaults to False — duplicates raise RegistryError.
    """

    def decorator(cls: type[BaseExtractor]):
        ExtractorRegistry.get_instance().register(name, cls, overwrite=overwrite)
        return cls

    return decorator
