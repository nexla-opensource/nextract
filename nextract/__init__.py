try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("nextract") if version else "unknown"
except PackageNotFoundError:
    __version__ = "unknown"

from .core import extract, batch_extract

__all__ = ["extract", "batch_extract", "__version__"]
