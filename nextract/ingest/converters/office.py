"""Public Office-to-PDF conversion utility."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import structlog

log = structlog.get_logger(__name__)

TMP_ROOT = Path(tempfile.gettempdir())

DEFAULT_CONVERSION_TIMEOUT = 120  # seconds


def convert_office_to_pdf(
    path: str | Path,
    timeout: int = DEFAULT_CONVERSION_TIMEOUT,
) -> tuple[Path | None, Path | None]:
    """Convert an Office document (.doc/.docx/.ppt/.pptx) to PDF.

    Uses LibreOffice/soffice or unoconv for the conversion.
    Writes output to a temp directory.

    Args:
        path: Path to the Office document.
        timeout: Maximum seconds for the subprocess conversion (default 120).

    Returns:
        A tuple of (pdf_path, temp_dir) where temp_dir should be cleaned up
        after use, or (None, None) on failure.
    """
    file_path = Path(path)
    out_dir = Path(tempfile.mkdtemp(prefix=f"nextract-officepdf-{file_path.stem}-", dir=str(TMP_ROOT)))
    target_pdf = out_dir / f"{file_path.stem}.pdf"

    soffice = _which("soffice", "libreoffice")
    if soffice:
        try:
            cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(file_path)]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
            if target_pdf.exists():
                return target_pdf, out_dir
            log.debug(
                "office_pdf_conversion_no_output",
                tool="soffice",
                returncode=res.returncode,
                stdout=res.stdout.decode(errors="ignore"),
                stderr=res.stderr.decode(errors="ignore"),
                file=str(file_path),
            )
        except subprocess.TimeoutExpired:
            log.warning("office_pdf_conversion_timeout", tool="soffice", timeout=timeout, file=str(file_path))
        except Exception as e:
            log.debug("office_pdf_conversion_exception", tool="soffice", error=str(e), file=str(file_path))

    unoconv = _which("unoconv")
    if unoconv:
        try:
            cmd = [unoconv, "-f", "pdf", "-o", str(out_dir), str(file_path)]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
            if target_pdf.exists():
                return target_pdf, out_dir
            log.debug(
                "office_pdf_conversion_no_output",
                tool="unoconv",
                returncode=res.returncode,
                stdout=res.stdout.decode(errors="ignore"),
                stderr=res.stderr.decode(errors="ignore"),
                file=str(file_path),
            )
        except subprocess.TimeoutExpired:
            log.warning("office_pdf_conversion_timeout", tool="unoconv", timeout=timeout, file=str(file_path))
        except Exception as e:
            log.debug("office_pdf_conversion_exception", tool="unoconv", error=str(e), file=str(file_path))

    # Failed all methods - clean up temp directory before returning
    shutil.rmtree(out_dir, ignore_errors=True)
    return None, None


def _which(*candidates: str) -> str | None:
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
    return None
