from __future__ import annotations

from typing import Dict, Optional, Type

from nextract.core.base import BaseProvider


class ProviderRegistry:
    """Registry for all provider implementations."""

    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def get_instance(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, provider_class: Type[BaseProvider]) -> None:
        self._providers[name] = provider_class

    def get(self, name: str) -> Optional[Type[BaseProvider]]:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


def register_provider(name: str):
    """Decorator to register a provider."""

    def decorator(cls: Type[BaseProvider]):
        ProviderRegistry.get_instance().register(name, cls)
        return cls

    return decorator
