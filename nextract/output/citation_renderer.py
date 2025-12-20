from __future__ import annotations

from typing import Iterable

from nextract.core import Citation


def render_citations(citations: Iterable[Citation]) -> str:
    """Render citations into a human-readable string."""
    lines = []
    for citation in citations:
        span = ""
        if citation.span:
            span = f"[{citation.span.start_pos}:{citation.span.end_pos}]"
        snippet = f" - {citation.snippet}" if citation.snippet else ""
        lines.append(f"{citation.source_path}:{citation.chunk_id}{span}{snippet}")
    return "\n".join(lines)
