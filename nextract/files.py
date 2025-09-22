from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from pydantic_ai import BinaryContent
from .mimetypes_map import guess_mime, is_textual, is_pdf, is_image, is_zip, is_office_binary

TMP_ROOT = Path("/tmp")

@dataclass
class PreparedPart:
    """Represents a part to be passed into Agent.run/run_sync:
       - either text (string) or a BinaryContent.
    """
    text: str | None = None
    binary: BinaryContent | None = None
    source_path: Path | None = None

def _read_text_file(path: Path) -> str:
    # "Read-as-is" with minimal decoding assumptions
    # We DO NOT parse/transform; just push raw text bytes to UTF-8 (lossy ok).
    b = path.read_bytes()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")

def _wrap_text_payload(path: Path, text: str, mime: str) -> str:
    header = f"\n--- BEGIN FILE: {path.name} ({mime}) ---\n"
    footer = f"\n--- END FILE: {path.name} ---\n"
    return header + text + footer

def _prepare_single_file(path: Path) -> List[PreparedPart]:
    parts: List[PreparedPart] = []
    mime = guess_mime(path)

    if is_textual(path):
        text = _read_text_file(path)
        parts.append(PreparedPart(text=_wrap_text_payload(path, text, mime), source_path=path))
        return parts

    if is_image(path) or is_pdf(path):
        bc = BinaryContent(data=path.read_bytes(), media_type=mime)
        parts.append(PreparedPart(binary=bc, source_path=path))
        return parts

    if is_office_binary(path):
        # Explicitly *not processing* Office docs for now; treat as unsupported.
        # We attach as binary so vision-capable models that support these formats (if any) may still attempt.
        bc = BinaryContent(data=path.read_bytes(), media_type=mime)
        parts.append(PreparedPart(binary=bc, source_path=path))
        return parts

    # Default: attach as binary blob
    bc = BinaryContent(data=path.read_bytes(), media_type=mime)
    parts.append(PreparedPart(binary=bc, source_path=path))
    return parts

def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a zip to dest_dir, avoiding traversal. Returns extracted file paths."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # prevent directory traversal
            member_path = Path(member.filename)
            if member.is_dir():
                continue
            target = dest_dir / member_path.name
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted

def prepare_parts(file_paths: Sequence[str | os.PathLike[str] | Path]) -> list[PreparedPart]:
    """Prepare Agent message parts from the provided file paths."""
    parts: list[PreparedPart] = []
    for p in file_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if is_zip(path):
            # Extract and treat each contained file individually.
            out_dir = TMP_ROOT / f"nextract-zip-{path.stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fp in _safe_extract_zip(path, out_dir):
                parts.extend(_prepare_single_file(fp))
        else:
            parts.extend(_prepare_single_file(path))

    return parts

def flatten_for_agent(parts: Iterable[PreparedPart]) -> list[str | BinaryContent]:
    """Agent.run accepts a list of content parts (strings or BinaryContent)."""
    out: list[str | BinaryContent] = []
    for p in parts:
        if p.text is not None:
            out.append(p.text)
        elif p.binary is not None:
            out.append(p.binary)
    return out
