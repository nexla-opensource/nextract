from __future__ import annotations

from typing import Dict, List, Optional, Type

from nextract.core.base import BaseChunker, Modality


class ChunkerRegistry:
    """Registry for chunkers."""

    _instance: Optional["ChunkerRegistry"] = None
    _chunkers: Dict[str, Type[BaseChunker]] = {}

    @classmethod
    def get_instance(cls) -> "ChunkerRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, chunker_class: Type[BaseChunker]) -> None:
        self._chunkers[name] = chunker_class

    def get(self, name: str) -> Optional[Type[BaseChunker]]:
        return self._chunkers.get(name)

    def get_chunkers_for_modality(self, modality: Modality) -> List[str]:
        applicable = []
        for name, chunker_class in self._chunkers.items():
            if modality in chunker_class.get_applicable_modalities():
                applicable.append(name)
        return applicable


def register_chunker(name: str):
    """Decorator to register a chunker."""

    def decorator(cls: Type[BaseChunker]):
        ChunkerRegistry.get_instance().register(name, cls)
        return cls

    return decorator
