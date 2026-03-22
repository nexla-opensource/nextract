from __future__ import annotations

import threading

from nextract.core.base import BaseChunker, Modality


class ChunkerRegistry:
    """Registry for chunkers."""

    _instance: ChunkerRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._chunkers: dict[str, type[BaseChunker]] = {}

    @classmethod
    def get_instance(cls) -> "ChunkerRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, chunker_class: type[BaseChunker]) -> None:
        if name in self._chunkers:
            import warnings
            warnings.warn(f"Chunker '{name}' already registered; overwriting.", stacklevel=3)
        self._chunkers[name] = chunker_class

    def get(self, name: str) -> type[BaseChunker] | None:
        return self._chunkers.get(name)

    def get_chunkers_for_modality(self, modality: Modality) -> list[str]:
        applicable = []
        for name, chunker_class in self._chunkers.items():
            if modality in chunker_class.get_applicable_modalities():
                applicable.append(name)
        return applicable


def register_chunker(name: str):
    """Decorator to register a chunker."""

    def decorator(cls: type[BaseChunker]):
        ChunkerRegistry.get_instance().register(name, cls)
        return cls

    return decorator
