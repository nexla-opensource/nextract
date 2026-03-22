"""
Adaptive PDF text extraction using multiple methods.

Automatically selects the best extraction method based on PDF analysis:
- PyMuPDF for text-rich PDFs (fast, accurate)
- Tesseract OCR for scanned PDFs (slow, necessary)
- Hybrid approach for mixed documents
"""

from __future__ import annotations

import io
from pathlib import Path

import structlog

from .pdf_analyzer import PDFAnalyzer, PDFAnalysis, ExtractionMethod

log = structlog.get_logger(__name__)


class PDFTextExtractor:
    """
    Adaptive PDF text extractor that automatically chooses the best method.
    """
    
    def __init__(
        self,
        analyzer: PDFAnalyzer | None = None,
        enable_ocr: bool = True,
        ocr_dpi: int = 300,
        include_page_numbers: bool = True,
        max_workers: int = 10
    ):
        """
        Initialize PDF text extractor.

        Args:
            analyzer: PDF analyzer instance (creates default if None)
            enable_ocr: Enable Tesseract OCR for scanned pages
            ocr_dpi: DPI for OCR image conversion
            include_page_numbers: Include page markers in extracted text
            max_workers: Maximum number of parallel workers for OCR (default: 10)
        """
        self.analyzer = analyzer or PDFAnalyzer()
        self.enable_ocr = enable_ocr
        self.ocr_dpi = ocr_dpi
        self.include_page_numbers = include_page_numbers
        self.max_workers = max_workers

        # Check OCR availability
        self.ocr_available = self._check_ocr_available()
    
    def extract(self, pdf_path: str | Path) -> tuple[str, PDFAnalysis]:
        """
        Extract text from PDF using the best method.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            (extracted_text, analysis)
        """
        path = Path(pdf_path).expanduser().resolve()
        
        # Analyze PDF to determine best method
        analysis = self.analyzer.analyze(path)
        
        log.info(
            "pdf_extraction_started",
            file=str(path),
            method=analysis.recommended_method.value,
            pdf_type=analysis.pdf_type.value
        )
        
        # Extract using recommended method
        if analysis.recommended_method == ExtractionMethod.PYMUPDF:
            text = self._extract_with_pymupdf(path)
        
        elif analysis.recommended_method == ExtractionMethod.TESSERACT:
            if self.enable_ocr and self.ocr_available:
                text = self._extract_with_tesseract(path)
            else:
                log.warning(
                    "ocr_not_available_fallback_pymupdf",
                    message="Tesseract recommended but not available, using PyMuPDF"
                )
                text = self._extract_with_pymupdf(path)
        
        elif analysis.recommended_method == ExtractionMethod.HYBRID:
            if self.enable_ocr and self.ocr_available:
                text = self._extract_hybrid(path)
            else:
                log.warning(
                    "ocr_not_available_fallback_pymupdf",
                    message="Hybrid method requires OCR, using PyMuPDF only"
                )
                text = self._extract_with_pymupdf(path)
        
        else:
            # Default to PyMuPDF
            text = self._extract_with_pymupdf(path)
        
        log.info(
            "pdf_extraction_complete",
            file=str(path),
            method=analysis.recommended_method.value,
            text_length=len(text),
            char_count=len(text)
        )
        
        return text, analysis
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fast, for text-based PDFs)."""
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")
        
        log.debug("extracting_with_pymupdf", file=str(pdf_path))

        doc = fitz.open(pdf_path)
        try:
            text_parts = []

            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()

                if page_text.strip():
                    if self.include_page_numbers:
                        text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")
                    else:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts)
        finally:
            doc.close()
    
    def _extract_with_tesseract(self, pdf_path: Path) -> str:
        """Extract text using Tesseract OCR (slow, for scanned PDFs)."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "Tesseract OCR dependencies required. Install with: "
                "pip install pytesseract pdf2image pillow"
            )

        log.debug("extracting_with_tesseract", file=str(pdf_path), dpi=self.ocr_dpi)

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.ocr_dpi)

        log.info(
            "ocr_conversion_complete",
            file=str(pdf_path),
            num_pages=len(images),
            dpi=self.ocr_dpi
        )

        # Parallel OCR processing
        if len(images) > 1:
            text_parts = self._ocr_images_parallel(images)
        else:
            text_parts = self._ocr_images_sequential(images)

        return "\n\n".join(text_parts)

    def _ocr_images_sequential(self, images: list) -> list[str]:
        """OCR images sequentially (for small documents)."""
        import pytesseract

        text_parts = []
        for page_num, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(image)

            if page_text.strip():
                if self.include_page_numbers:
                    text_parts.append(f"--- PAGE {page_num} (OCR) ---\n{page_text}")
                else:
                    text_parts.append(page_text)

        return text_parts

    def _ocr_images_parallel(self, images: list) -> list[str]:
        """OCR images in parallel using ThreadPoolExecutor."""
        import pytesseract
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log.info(
            "ocr_parallel_processing_started",
            num_pages=len(images),
            max_workers=self.max_workers
        )

        # Create a function to OCR a single page
        def ocr_single_page(page_data: tuple) -> tuple[int, str]:
            """OCR a single page and return (page_num, text)."""
            page_num, image = page_data
            try:
                page_text = pytesseract.image_to_string(image)
                return (page_num, page_text)
            except Exception as e:
                log.warning(
                    "ocr_page_failed",
                    page=page_num,
                    error=str(e)
                )
                return (page_num, "")

        # Process pages in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pages
            future_to_page = {
                executor.submit(ocr_single_page, (page_num, image)): page_num
                for page_num, image in enumerate(images, 1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num, page_text = future.result()
                results[page_num] = page_text

                # Log progress every 10 pages
                if len(results) % 10 == 0:
                    log.info(
                        "ocr_progress",
                        completed=len(results),
                        total=len(images),
                        percentage=f"{len(results)/len(images)*100:.1f}%"
                    )

        log.info(
            "ocr_parallel_processing_complete",
            num_pages=len(images),
            successful=len([t for t in results.values() if t.strip()])
        )

        # Reconstruct text parts in order
        text_parts = []
        for page_num in sorted(results.keys()):
            page_text = results[page_num]
            if page_text.strip():
                if self.include_page_numbers:
                    text_parts.append(f"--- PAGE {page_num} (OCR) ---\n{page_text}")
                else:
                    text_parts.append(page_text)

        return text_parts
    
    def _extract_hybrid(self, pdf_path: Path) -> str:
        """
        Extract text using hybrid approach.
        
        Uses PyMuPDF for text-rich pages, Tesseract for scanned pages.
        """
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                f"Hybrid extraction requires PyMuPDF and Tesseract: {e}"
            )
        
        log.debug("extracting_with_hybrid", file=str(pdf_path))

        doc = fitz.open(pdf_path)
        try:
            text_parts = []

            for page_num, page in enumerate(doc, 1):
                # Try PyMuPDF first
                page_text = page.get_text().strip()

                # If page has sufficient text, use it
                if len(page_text) >= self.analyzer.min_chars_per_page:
                    if self.include_page_numbers:
                        text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")
                    else:
                        text_parts.append(page_text)

                # Otherwise, use OCR
                else:
                    try:
                        # Convert page to image
                        pix = page.get_pixmap(dpi=self.ocr_dpi)
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))

                        # OCR the image
                        ocr_text = pytesseract.image_to_string(image)

                        if ocr_text.strip():
                            if self.include_page_numbers:
                                text_parts.append(f"--- PAGE {page_num} (OCR) ---\n{ocr_text}")
                            else:
                                text_parts.append(ocr_text)
                        else:
                            log.warning("page_no_text", file=str(pdf_path), page=page_num)
                            if self.include_page_numbers:
                                text_parts.append(f"--- PAGE {page_num} ---\n[No text extracted]")

                    except Exception as e:
                        log.warning(
                            "page_ocr_failed",
                            file=str(pdf_path),
                            page=page_num,
                            error=str(e)
                        )
                        if self.include_page_numbers:
                            text_parts.append(f"--- PAGE {page_num} ---\n[OCR failed]")

            return "\n\n".join(text_parts)
        finally:
            doc.close()
    
    def _check_ocr_available(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract

            # Try to get Tesseract version
            pytesseract.get_tesseract_version()
            return True
        
        except ImportError:
            log.debug(
                "ocr_dependencies_not_installed",
                message="Install with: pip install pytesseract pdf2image pillow"
            )
            return False
        
        except Exception as e:
            log.debug(
                "tesseract_not_available",
                error=str(e),
                message="Tesseract binary not found. Install from: https://github.com/tesseract-ocr/tesseract"
            )
            return False


def extract_pdf_text(
    pdf_path: str | Path,
    enable_ocr: bool = True,
    include_page_numbers: bool = True,
    max_workers: int = 10
) -> tuple[str, PDFAnalysis]:
    """
    Convenience function to extract text from PDF.

    Args:
        pdf_path: Path to PDF file
        enable_ocr: Enable OCR for scanned pages
        include_page_numbers: Include page markers in text
        max_workers: Maximum number of parallel workers for OCR (default: 10)

    Returns:
        (extracted_text, analysis)

    Example:
        text, analysis = extract_pdf_text("document.pdf")
        print(f"PDF Type: {analysis.pdf_type.value}")
        print(f"Method: {analysis.recommended_method.value}")
        print(f"Text: {text[:500]}...")
    """
    extractor = PDFTextExtractor(
        enable_ocr=enable_ocr,
        include_page_numbers=include_page_numbers,
        max_workers=max_workers
    )
    return extractor.extract(pdf_path)

