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

__all__ = ["extract", "batch_extract", "RagDocumentChunker", "__version__"]


def __getattr__(name):
    # Lazy export: rag_chunking pulls heavy deps (pandas, pdfplumber, google-genai),
    # so only import it when actually requested.
    if name == "RagDocumentChunker":
        from .rag_chunking import RagDocumentChunker

        return RagDocumentChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Keep dir()/tab-completion aware of the lazy export.
    return sorted(set(globals()) | set(__all__))
