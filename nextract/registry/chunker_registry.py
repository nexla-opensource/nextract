from __future__ import annotations

from nextract.core.base import BaseChunker
from nextract.core.types import Modality
from nextract.registry.base import Registry


class ChunkerRegistry(Registry[BaseChunker]):
    """Registry for chunkers."""

    def __init__(self) -> None:
        super().__init__(kind="chunker")

    def get_chunkers_for_modality(self, modality: Modality) -> list[str]:
        applicable = []
        for name in self.list_items():
            chunker_class = self.get(name)
            if chunker_class and modality in chunker_class.get_applicable_modalities():
                applicable.append(name)
        return applicable


def register_chunker(name: str, *, overwrite: bool = False):
    """Decorator to register a chunker.

    Args:
        name: Unique name for the chunker.
        overwrite: If True, silently replace any existing registration.
            Defaults to False — duplicates raise RegistryError.
    """

    def decorator(cls: type[BaseChunker]):
        ChunkerRegistry.get_instance().register(name, cls, overwrite=overwrite)
        return cls

    return decorator
