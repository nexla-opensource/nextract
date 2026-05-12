from __future__ import annotations

from nextract.core.base import BaseProvider
from nextract.registry.base import Registry


class ProviderRegistry(Registry[BaseProvider]):
    """Registry for all provider implementations."""

    def __init__(self) -> None:
        super().__init__(kind="provider")

    def list_providers(self) -> list[str]:
        return self.list_items()


def register_provider(name: str, *, overwrite: bool = False):
    """Decorator to register a provider.

    Args:
        name: Unique name for the provider.
        overwrite: If True, silently replace any existing registration.
            Defaults to False — duplicates raise RegistryError.
    """

    def decorator(cls: type[BaseProvider]):
        ProviderRegistry.get_instance().register(name, cls, overwrite=overwrite)
        return cls

    return decorator
