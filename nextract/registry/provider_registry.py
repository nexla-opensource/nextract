from __future__ import annotations

import threading

from nextract.core.base import BaseProvider


class ProviderRegistry:
    """Registry for all provider implementations."""

    _instance: ProviderRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseProvider]] = {}

    @classmethod
    def get_instance(cls) -> "ProviderRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, provider_class: type[BaseProvider]) -> None:
        if name in self._providers:
            import warnings
            warnings.warn(f"Provider '{name}' already registered; overwriting.", stacklevel=3)
        self._providers[name] = provider_class

    def get(self, name: str) -> type[BaseProvider] | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


def register_provider(name: str):
    """Decorator to register a provider."""

    def decorator(cls: type[BaseProvider]):
        ProviderRegistry.get_instance().register(name, cls)
        return cls

    return decorator
