"""Centralized plugin bootstrap for registry population.

Import this module instead of scattering side-effect imports across the codebase.
Ensures all extractors, providers, and chunkers are registered before use.
"""

from __future__ import annotations

import threading

_loaded = False
_lock = threading.Lock()


def ensure_plugins_loaded() -> None:
    """Import all first-party plugin packages to trigger registration.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        import nextract.extractors  # noqa: F401
        import nextract.providers  # noqa: F401
        import nextract.chunking  # noqa: F401
        _loaded = True


def reset_bootstrap() -> None:
    """Reset the bootstrap state. For test isolation only."""
    global _loaded
    with _lock:
        _loaded = False
