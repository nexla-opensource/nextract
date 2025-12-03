"""
Intelligent PDF analysis and adaptive text extraction.

This module analyzes PDFs to determine the best extraction strategy:
- Text-rich PDFs: Use PyMuPDF (fast, accurate)
- Scanned PDFs: Use Tesseract OCR (slow, necessary)
- Hybrid PDFs: Mixed approach (page-by-page detection)
- Image-heavy PDFs: Use vision API
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


class PDFType(Enum):
    """PDF document type classification"""
    TEXT_RICH = "text_rich"  # Digital PDF with embedded text
    SCANNED = "scanned"  # Scanned document (image-based)
    HYBRID = "hybrid"  # Mix of text and scanned pages
    IMAGE_HEAVY = "image_heavy"  # Contains many images/charts
    UNKNOWN = "unknown"  # Unable to determine


class ExtractionMethod(Enum):
    """Text extraction method"""
    PYMUPDF = "pymupdf"  # Fast text extraction
    TESSERACT = "tesseract"  # OCR extraction
    VISION_API = "vision_api"  # LLM vision API
    HYBRID = "hybrid"  # Mixed approach


@dataclass
class PDFAnalysis:
    """Analysis result for a PDF document"""
    pdf_type: PDFType
    total_pages: int
    text_pages: int  # Pages with extractable text
    scanned_pages: int  # Pages requiring OCR
    image_pages: int  # Pages with significant images
    avg_chars_per_page: float
    text_quality_score: float  # 0.0 to 1.0
    recommended_method: ExtractionMethod
    confidence: float  # 0.0 to 1.0
    details: dict  # Additional metadata


class PDFAnalyzer:
    """
    Intelligent PDF analyzer that detects document type and recommends
    the best extraction strategy.
    """
    
    def __init__(
        self,
        sample_pages: int = 10,
        min_chars_per_page: int = 100,
        text_quality_threshold: float = 0.7
    ):
        """
        Initialize PDF analyzer.
        
        Args:
            sample_pages: Number of pages to sample for analysis
            min_chars_per_page: Minimum characters to consider page as text-rich
            text_quality_threshold: Threshold for text quality (0.0-1.0)
        """
        self.sample_pages = sample_pages
        self.min_chars_per_page = min_chars_per_page
        self.text_quality_threshold = text_quality_threshold
    
    def analyze(self, pdf_path: str | Path) -> PDFAnalysis:
        """
        Analyze PDF and determine best extraction strategy.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFAnalysis with recommended extraction method
        """
        path = Path(pdf_path).expanduser().resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
            log.error(
                "pymupdf_not_installed",
                message="PyMuPDF required for PDF analysis. Install with: pip install PyMuPDF"
            )
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")
        
        log.info("pdf_analysis_started", file=str(path))
        
        try:
            doc = fitz.open(path)
            total_pages = len(doc)
            
            # Sample pages for analysis
            sample_size = min(self.sample_pages, total_pages)
            sample_indices = self._get_sample_indices(total_pages, sample_size)
            
            # Analyze each sampled page
            text_pages = 0
            scanned_pages = 0
            image_pages = 0
            total_chars = 0
            
            for page_idx in sample_indices:
                page = doc[page_idx]
                
                # Extract text
                text = page.get_text().strip()
                char_count = len(text)
                total_chars += char_count
                
                # Count images
                image_list = page.get_images()
                has_images = len(image_list) > 0
                
                # Classify page
                if char_count >= self.min_chars_per_page:
                    text_pages += 1
                    if has_images and len(image_list) > 2:
                        image_pages += 1
                else:
                    scanned_pages += 1
            
            doc.close()
            
            # Calculate metrics
            avg_chars_per_page = total_chars / sample_size if sample_size > 0 else 0
            text_ratio = text_pages / sample_size if sample_size > 0 else 0
            
            # Calculate text quality score (0.0 to 1.0)
            text_quality_score = self._calculate_text_quality(
                text_ratio=text_ratio,
                avg_chars_per_page=avg_chars_per_page
            )
            
            # Determine PDF type
            pdf_type = self._classify_pdf_type(
                text_pages=text_pages,
                scanned_pages=scanned_pages,
                image_pages=image_pages,
                sample_size=sample_size
            )
            
            # Recommend extraction method
            recommended_method, confidence = self._recommend_method(
                pdf_type=pdf_type,
                text_quality_score=text_quality_score,
                total_pages=total_pages
            )
            
            analysis = PDFAnalysis(
                pdf_type=pdf_type,
                total_pages=total_pages,
                text_pages=text_pages,
                scanned_pages=scanned_pages,
                image_pages=image_pages,
                avg_chars_per_page=avg_chars_per_page,
                text_quality_score=text_quality_score,
                recommended_method=recommended_method,
                confidence=confidence,
                details={
                    "sample_size": sample_size,
                    "text_ratio": text_ratio,
                    "sampled_pages": sample_indices
                }
            )
            
            log.info(
                "pdf_analysis_complete",
                file=str(path),
                pdf_type=pdf_type.value,
                recommended_method=recommended_method.value,
                text_quality_score=f"{text_quality_score:.2f}",
                confidence=f"{confidence:.2f}"
            )
            
            return analysis
            
        except Exception as e:
            log.error("pdf_analysis_failed", file=str(path), error=str(e))
            raise
    
    def _get_sample_indices(self, total_pages: int, sample_size: int) -> list[int]:
        """
        Get evenly distributed sample page indices.
        
        Samples from beginning, middle, and end of document.
        """
        if total_pages <= sample_size:
            return list(range(total_pages))
        
        # Sample evenly across document
        step = total_pages / sample_size
        indices = [int(i * step) for i in range(sample_size)]
        
        return indices
    
    def _calculate_text_quality(
        self,
        text_ratio: float,
        avg_chars_per_page: float
    ) -> float:
        """
        Calculate text quality score (0.0 to 1.0).
        
        Higher score = better text extraction quality
        """
        # Weight text ratio (70%) and character density (30%)
        ratio_score = text_ratio
        
        # Normalize character count (assume 500 chars/page is good)
        char_score = min(1.0, avg_chars_per_page / 500)
        
        quality_score = (ratio_score * 0.7) + (char_score * 0.3)
        
        return quality_score
    
    def _classify_pdf_type(
        self,
        text_pages: int,
        scanned_pages: int,
        image_pages: int,
        sample_size: int
    ) -> PDFType:
        """Classify PDF type based on page analysis."""
        
        if sample_size == 0:
            return PDFType.UNKNOWN
        
        text_ratio = text_pages / sample_size
        scanned_ratio = scanned_pages / sample_size
        image_ratio = image_pages / sample_size
        
        # Text-rich: >80% pages have text
        if text_ratio > 0.8:
            if image_ratio > 0.3:
                return PDFType.IMAGE_HEAVY
            return PDFType.TEXT_RICH
        
        # Scanned: >80% pages have no text
        elif scanned_ratio > 0.8:
            return PDFType.SCANNED
        
        # Hybrid: mix of text and scanned
        else:
            return PDFType.HYBRID
    
    def _recommend_method(
        self,
        pdf_type: PDFType,
        text_quality_score: float,
        total_pages: int
    ) -> tuple[ExtractionMethod, float]:
        """
        Recommend extraction method based on analysis.
        
        Returns:
            (recommended_method, confidence)
        """
        
        # Text-rich PDFs: Use PyMuPDF
        if pdf_type == PDFType.TEXT_RICH:
            if text_quality_score >= self.text_quality_threshold:
                return ExtractionMethod.PYMUPDF, 0.95
            else:
                return ExtractionMethod.PYMUPDF, 0.75
        
        # Scanned PDFs: Use Tesseract OCR
        elif pdf_type == PDFType.SCANNED:
            return ExtractionMethod.TESSERACT, 0.90
        
        # Hybrid PDFs: Use mixed approach
        elif pdf_type == PDFType.HYBRID:
            return ExtractionMethod.HYBRID, 0.80
        
        # Image-heavy PDFs: Use vision API for best results
        elif pdf_type == PDFType.IMAGE_HEAVY:
            # For small docs, vision API is fine
            if total_pages <= 20:
                return ExtractionMethod.VISION_API, 0.85
            # For large docs, use hybrid (text + selective vision)
            else:
                return ExtractionMethod.HYBRID, 0.75
        
        # Unknown: Default to PyMuPDF with low confidence
        else:
            return ExtractionMethod.PYMUPDF, 0.50

