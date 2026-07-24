"""RAG chunking: multi-format (PDF/CSV/Excel/image) LLM-driven document chunker.

Ported verbatim from the Nexla production chunking pipeline (see PROVENANCE.md
for the module-by-module source mapping). :class:`RagDocumentChunker` wraps the
internal ``DocumentPipeline``, which profiles each file, routes it to a
format-specific handler, and uses Gemini for planning, chunking, metadata
enrichment, and verification. Unlike :class:`nextract.chunking.DocumentChunker`
(which splits large documents so schema extraction fits in context), this
produces retrieval-ready chunks with rich metadata.

The ``use_vertex`` flag selects how the API key is interpreted: ``True`` (the
default, matching production) means a Vertex Agent-Platform express-mode key;
pass ``False`` for a plain Gemini Developer API key.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import pandas as pd
import structlog
from pydantic import BaseModel

from .config import Config
from .pipeline import DocumentPipeline

__all__ = ["RagDocumentChunker", "Chunk", "Config"]

log = structlog.get_logger(__name__)


class Chunk(BaseModel):
    """A retrieval-ready chunk: text plus flat metadata.

    Vendored from the ai-chunking package's ``Chunk`` model so this subpackage
    stands alone.
    """

    text: str
    metadata: dict[str, Any]


def _ensure_no_running_loop(method_name: str) -> None:
    """Raise if called from inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — safe to proceed with the sync API.
        return
    raise RuntimeError(
        f"RagDocumentChunker.{method_name}() is a synchronous API and cannot be "
        "called from inside a running asyncio event loop: the underlying pipeline "
        "handlers call asyncio.run() internally. Call it from synchronous code "
        "(e.g. via asyncio.to_thread or a worker thread)."
    )


class RagDocumentChunker:
    """Multi-format (PDF/CSV/Excel/image) LLM-driven chunker. Gemini-backed."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        custom_instructions: Optional[str] = None,
        use_vertex: bool = True,
    ) -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "RagDocumentChunker requires a Gemini API key. Pass api_key=... or "
                "set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        self._pipeline = DocumentPipeline(
            config or Config(), api_key, custom_instructions, use_vertex=use_vertex
        )

    def chunk_document(
        self, file_path: str, extra_metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """Process a single file and return its chunks.

        Args:
            file_path: Path to the file (PDF/CSV/Excel/image) to process.
            extra_metadata: Optional dict merged into every chunk's metadata.

        Returns:
            List of chunks in (output_name, row-order) order.
        """
        _ensure_no_running_loop("chunk_document")
        meta = {"tags": {"display_path": str(file_path)}}
        results = self._pipeline.process_file(str(file_path), meta)

        chunks: list[Chunk] = []
        for output_name, df in results.items():
            for _, row in df.iterrows():
                row_dict = dict(row)
                text = str(row_dict.pop("chunk_text")) if "chunk_text" in row_dict else ""
                metadata: dict[str, Any] = {}
                for key, value in row_dict.items():
                    try:
                        if pd.isna(value):
                            value = None
                    except (TypeError, ValueError):
                        # Non-scalar values (dict/list/ndarray) pass through unchanged.
                        pass
                    metadata[key] = value
                metadata["output_name"] = output_name
                if extra_metadata:
                    metadata.update(extra_metadata)
                chunks.append(Chunk(text=text, metadata=metadata))
        return chunks

    def chunk_documents(self, file_paths: list[str]) -> list[Chunk]:
        """Process multiple files and return all chunks concatenated."""
        all_chunks: list[Chunk] = []
        for file_path in file_paths:
            all_chunks.extend(self.chunk_document(file_path))
        return all_chunks

    def process_to_dataframes(self, file_path: str) -> dict:
        """Process a file and return the raw {output_name: DataFrame} mapping.

        This exposes the production pipeline's native output surface for
        platform compatibility and parity testing.
        """
        _ensure_no_running_loop("process_to_dataframes")
        meta = {"tags": {"display_path": str(file_path)}}
        return self._pipeline.process_file(str(file_path), meta)
