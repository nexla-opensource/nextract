from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
import subprocess
import tempfile
import shutil
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from pydantic_ai import BinaryContent
from .mimetypes_map import guess_mime, is_textual, is_pdf, is_image, is_zip, is_office_binary

import structlog

log = structlog.get_logger(__name__)

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


def _which(*candidates: str) -> str | None:
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
    return None


def _convert_office_to_pdf(path: Path) -> Path | None:
    """Convert .doc/.docx/.ppt/.pptx to PDF using available CLI tools.

    Tries LibreOffice/soffice first, then unoconv. Writes output to a temp dir under /tmp.
    Returns the PDF path on success, or None on failure.
    """
    # Prepare temp output directory per file
    out_dir = Path(tempfile.mkdtemp(prefix=f"nextract-officepdf-{path.stem}-", dir=str(TMP_ROOT)))
    target_pdf = out_dir / f"{path.stem}.pdf"

    soffice = _which("soffice", "libreoffice")
    if soffice:
        try:
            # --headless convert avoids loading file into memory in Python
            cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if target_pdf.exists():
                return target_pdf
            else:
                log.warning(
                    "office_pdf_conversion_no_output",
                    tool="soffice",
                    returncode=res.returncode,
                    stdout=res.stdout.decode(errors="ignore"),
                    stderr=res.stderr.decode(errors="ignore"),
                    file=str(path),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("office_pdf_conversion_exception", tool="soffice", error=str(e), file=str(path))

    unoconv = _which("unoconv")
    if unoconv:
        try:
            cmd = [unoconv, "-f", "pdf", "-o", str(out_dir), str(path)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if target_pdf.exists():
                return target_pdf
            else:
                log.warning(
                    "office_pdf_conversion_no_output",
                    tool="unoconv",
                    returncode=res.returncode,
                    stdout=res.stdout.decode(errors="ignore"),
                    stderr=res.stderr.decode(errors="ignore"),
                    file=str(path),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("office_pdf_conversion_exception", tool="unoconv", error=str(e), file=str(path))

    # Failed all methods
    return None


def _xlsx_to_text(path: Path) -> str:
    """Lightweight XLSX to TSV text extractor using zip + XML parsing.

    - Extracts shared strings
    - Emits each worksheet as tab-separated rows
    - Best-effort only (styles, dates, formulas are rendered as their value text when possible)
    """
    def col_to_index(col: str) -> int:
        idx = 0
        for ch in col:
            if not ch.isalpha():
                break
            idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
        return idx

    def cell_ref_to_coords(ref: str) -> tuple[int, int]:
        m = re.match(r"([A-Za-z]+)([0-9]+)", ref)
        if not m:
            return 1, 1
        col, row = m.group(1), m.group(2)
        return col_to_index(col), int(row)

    with zipfile.ZipFile(path, "r") as zf:
        # shared strings
        shared_strings: list[str] = []
        try:
            with zf.open("xl/sharedStrings.xml") as f:
                root = ET.fromstring(f.read())
                # Namespace agnostic: tags end with 'si' and 't'
                for si in root.iter():
                    if si.tag.endswith("si"):
                        text_parts: list[str] = []
                        for t in si.iter():
                            if t.tag.endswith("t") and (t.text is not None):
                                text_parts.append(t.text)
                        if text_parts:
                            shared_strings.append("".join(text_parts))
        except KeyError:
            pass  # no shared strings

        # Sheet names (best-effort)
        sheet_names: list[str] = []
        try:
            with zf.open("xl/workbook.xml") as f:
                wb = ET.fromstring(f.read())
                for s in wb.iter():
                    if s.tag.endswith("sheet"):
                        nm = s.attrib.get("name")
                        if nm:
                            sheet_names.append(nm)
        except KeyError:
            pass

        out_blocks: list[str] = []
        # Iterate known sheet files in order; names if available
        i = 1
        while True:
            sheet_path = f"xl/worksheets/sheet{i}.xml"
            try:
                with zf.open(sheet_path) as f:
                    root = ET.fromstring(f.read())
            except KeyError:
                break

            name = sheet_names[i - 1] if i - 1 < len(sheet_names) else f"Sheet{i}"
            out_lines: list[str] = [f"### {name}"]

            for row in root.iter():
                if row.tag.endswith("row"):
                    # Build row with sparse cells
                    cells: dict[int, str] = {}
                    max_col = 0
                    for c in row.iter():
                        if c.tag.endswith("c"):
                            r = c.attrib.get("r", "A1")
                            t = c.attrib.get("t")  # s=shared, b=bool, str=str, e=error
                            v_text = ""
                            v = None
                            for child in c:
                                if child.tag.endswith("v"):
                                    v = child.text
                                    break
                            if v is None:
                                # inlineStr case
                                is_text = None
                                for child in c:
                                    if child.tag.endswith("is"):
                                        # concatenated <t>
                                        parts: list[str] = []
                                        for sub in child.iter():
                                            if sub.tag.endswith("t") and (sub.text is not None):
                                                parts.append(sub.text)
                                        is_text = "".join(parts) if parts else ""
                                        break
                                v_text = is_text or ""
                            else:
                                if t == "s":
                                    # shared string index
                                    try:
                                        idx = int(v)
                                        v_text = shared_strings[idx] if 0 <= idx < len(shared_strings) else str(v)
                                    except Exception:
                                        v_text = str(v)
                                else:
                                    v_text = str(v)

                            col_idx, _ = cell_ref_to_coords(r)
                            max_col = max(max_col, col_idx)
                            cells[col_idx] = v_text

                    if max_col > 0:
                        row_vals = [cells.get(ci, "") for ci in range(1, max_col + 1)]
                        out_lines.append("\t".join(row_vals))

            out_blocks.append("\n".join(out_lines))
            i += 1

        return "\n\n".join(out_blocks) if out_blocks else ""


def _xls_to_text_via_cli(path: Path) -> str | None:
    """Attempt to convert legacy .xls to CSV via CLI tools and return text.

    Prefers LibreOffice/soffice; falls back to unoconv. Returns CSV text for the first sheet.
    Returns None if conversion failed.
    """
    out_dir = Path(tempfile.mkdtemp(prefix=f"nextract-xls2csv-{path.stem}-", dir=str(TMP_ROOT)))
    soffice = _which("soffice", "libreoffice")
    if soffice:
        try:
            # Use CSV export filter; defaults to active sheet. We still accept it.
            cmd = [
                soffice,
                "--headless",
                "--convert-to",
                "csv",
                "--outdir",
                str(out_dir),
                str(path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            # Find produced CSV (same stem or similar)
            produced = list(out_dir.glob("*.csv"))
            if produced:
                try:
                    return produced[0].read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    return produced[0].read_text(encoding="latin-1", errors="replace")
            else:
                log.warning(
                    "xls_csv_no_output",
                    tool="soffice",
                    returncode=res.returncode,
                    stdout=res.stdout.decode(errors="ignore"),
                    stderr=res.stderr.decode(errors="ignore"),
                    file=str(path),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("xls_csv_exception", tool="soffice", error=str(e), file=str(path))

    unoconv = _which("unoconv")
    if unoconv:
        try:
            cmd = [unoconv, "-f", "csv", "-o", str(out_dir), str(path)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            produced = list(out_dir.glob("*.csv"))
            if produced:
                try:
                    return produced[0].read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    return produced[0].read_text(encoding="latin-1", errors="replace")
            else:
                log.warning(
                    "xls_csv_no_output",
                    tool="unoconv",
                    returncode=res.returncode,
                    stdout=res.stdout.decode(errors="ignore"),
                    stderr=res.stderr.decode(errors="ignore"),
                    file=str(path),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("xls_csv_exception", tool="unoconv", error=str(e), file=str(path))
    return None

def _prepare_single_file(path: Path) -> List[PreparedPart]:
    parts: List[PreparedPart] = []
    mime = guess_mime(path)

    if is_textual(path):
        # Special handling for Excel files to extract textual content
        if path.suffix.lower() in {".xlsx"}:
            try:
                text = _xlsx_to_text(path)
            except Exception as e:  # noqa: BLE001
                log.warning("xlsx_text_extraction_failed", file=str(path), error=str(e))
                # Fallback to raw bytes decode (not ideal)
                text = _read_text_file(path)
            parts.append(PreparedPart(text=_wrap_text_payload(path, text, "text/tab-separated-values"), source_path=path))
            return parts
        elif path.suffix.lower() in {".xls"}:
            # Prefer CLI-based CSV extraction; fallback to raw decode
            text_cli = _xls_to_text_via_cli(path)
            if text_cli is not None:
                parts.append(PreparedPart(text=_wrap_text_payload(path, text_cli, "text/csv"), source_path=path))
                return parts
            else:
                # Last resort: attempt a lossy decode
                log.warning("xls_text_extraction_fallback_raw_decode", file=str(path))
                text = _read_text_file(path)
                parts.append(PreparedPart(text=_wrap_text_payload(path, text, "text/plain"), source_path=path))
                return parts

        text = _read_text_file(path)
        parts.append(PreparedPart(text=_wrap_text_payload(path, text, mime), source_path=path))
        return parts

    if is_image(path) or is_pdf(path):
        bc = BinaryContent(data=path.read_bytes(), media_type=mime)
        parts.append(PreparedPart(binary=bc, source_path=path))
        return parts

    if is_office_binary(path):
        # Convert Office docs to PDF, then attach PDF as binary for LLMs
        pdf_path = _convert_office_to_pdf(path)
        if pdf_path and pdf_path.exists():
            try:
                pdf_bytes = pdf_path.read_bytes()
                bc = BinaryContent(data=pdf_bytes, media_type="application/pdf")
                parts.append(PreparedPart(binary=bc, source_path=path))
                return parts
            except Exception as e:  # noqa: BLE001
                log.warning("office_pdf_read_failed", file=str(pdf_path), error=str(e))
        # Fallback: attach original binary if conversion failed
        log.warning("office_pdf_conversion_failed_fallback_binary", file=str(path))
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
