"""Generic registry base for plugin discovery."""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")

# Per-subclass singleton storage: maps subclass type -> instance
_instances: dict[type[Registry], Registry] = {}  # type: ignore[type-arg]
_instances_lock = threading.Lock()


class RegistryError(Exception):
    """Raised on registry conflicts or lookup failures."""


class Registry(Generic[T]):
    """Thread-safe generic registry for named plugin classes.

    Provides singleton access, type-safe registration, lookup,
    and test isolation via snapshot/restore.

    Each concrete subclass gets its own singleton instance.
    """

    def __init__(self, kind: str = "item") -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}
        self._mutation_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> Registry[T]:
        """Return the singleton instance for this specific subclass."""
        if cls not in _instances:
            with _instances_lock:
                if cls not in _instances:
                    _instances[cls] = cls()
        return _instances[cls]

    def register(self, name: str, item_class: type[T], *, overwrite: bool = False) -> None:
        with self._mutation_lock:
            if name in self._items and not overwrite:
                raise RegistryError(
                    f"{self._kind.capitalize()} '{name}' already registered. "
                    f"Use overwrite=True to replace."
                )
            self._items[name] = item_class

    def get(self, name: str) -> type[T] | None:
        return self._items.get(name)

    def list_items(self) -> list[str]:
        with self._mutation_lock:
            return sorted(self._items)

    def snapshot(self) -> dict[str, type[T]]:
        """Return a shallow copy of current registrations for test restore."""
        with self._mutation_lock:
            return dict(self._items)

    def restore(self, snapshot: dict[str, type[T]]) -> None:
        """Restore registrations from a prior snapshot."""
        with self._mutation_lock:
            self._items = dict(snapshot)

    def clear(self) -> None:
        """Remove all registrations. Primarily for test isolation."""
        with self._mutation_lock:
            self._items.clear()
