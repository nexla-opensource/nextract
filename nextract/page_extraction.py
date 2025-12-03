"""
Page-specific PDF extraction for provenance-guided retry.

Extracts specific page ranges from PDFs for focused extraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import structlog

log = structlog.get_logger(__name__)


def extract_pdf_pages(
    pdf_path: str | Path,
    page_range: tuple[int, int],
    include_page_numbers: bool = True
) -> str:
    """
    Extract text from specific pages of a PDF.

    For scanned PDFs, this will perform OCR on the specified pages.

    Args:
        pdf_path: Path to PDF file
        page_range: Tuple of (start_page, end_page) - 1-indexed, inclusive
        include_page_numbers: Include page markers in extracted text

    Returns:
        Extracted text from the specified pages

    Example:
        # Extract pages 165-167
        text = extract_pdf_pages("document.pdf", (165, 167))
    """
    path = Path(pdf_path).expanduser().resolve()
    start_page, end_page = page_range

    log.info(
        "extracting_pdf_pages",
        file=str(path),
        page_range=f"{start_page}-{end_page}",
        num_pages=end_page - start_page + 1
    )

    # Create a temporary PDF with just the specified pages
    # Then use PDFTextExtractor which handles OCR for scanned PDFs
    temp_pdf_path = create_temp_pdf_from_pages(pdf_path, page_range)

    try:
        from .pdf_extractor import extract_pdf_text

        # Extract text with OCR support
        extracted_text, analysis = extract_pdf_text(
            temp_pdf_path,
            enable_ocr=True,
            include_page_numbers=include_page_numbers
        )

        log.info(
            "pdf_pages_extracted",
            file=str(path),
            page_range=f"{start_page}-{end_page}",
            text_length=len(extracted_text),
            pdf_type=analysis.pdf_type.value,
            method=analysis.recommended_method.value
        )

        return extracted_text

    finally:
        # Clean up temporary PDF
        try:
            Path(temp_pdf_path).unlink()
        except Exception as e:
            log.warning("temp_pdf_cleanup_failed", temp_file=temp_pdf_path, error=str(e))


def create_temp_pdf_from_pages(
    pdf_path: str | Path,
    page_range: tuple[int, int]
) -> str:
    """
    Create a temporary PDF file containing only specified pages.
    
    Args:
        pdf_path: Path to source PDF file
        page_range: Tuple of (start_page, end_page) - 1-indexed, inclusive
    
    Returns:
        Path to temporary PDF file (caller must delete)
    
    Example:
        temp_pdf = create_temp_pdf_from_pages("document.pdf", (165, 167))
        # Use temp_pdf...
        os.unlink(temp_pdf)  # Clean up
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")
    
    path = Path(pdf_path).expanduser().resolve()
    start_page, end_page = page_range
    
    # Convert to 0-indexed
    start_idx = start_page - 1
    end_idx = end_page - 1
    
    log.debug(
        "creating_temp_pdf",
        source=str(path),
        page_range=f"{start_page}-{end_page}"
    )
    
    # Open source PDF
    source_doc = fitz.open(path)
    
    # Validate page range
    if start_idx < 0 or end_idx >= len(source_doc):
        source_doc.close()
        raise ValueError(
            f"Invalid page range {start_page}-{end_page}. "
            f"PDF has {len(source_doc)} pages."
        )
    
    # Create new PDF with selected pages
    temp_doc = fitz.open()
    temp_doc.insert_pdf(source_doc, from_page=start_idx, to_page=end_idx)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        temp_doc.save(f.name)
        temp_path = f.name
    
    temp_doc.close()
    source_doc.close()
    
    log.debug(
        "temp_pdf_created",
        source=str(path),
        temp_file=temp_path,
        page_range=f"{start_page}-{end_page}"
    )
    
    return temp_path

