"""
Provenance tracking for Nextract extractions.

Tracks where each extracted field came from (page, chunk, file) with
confidence scores and text citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import structlog

log = structlog.get_logger(__name__)


@dataclass
class FieldProvenance:
    """
    Provenance information for a single extracted field.
    
    Tracks where a field value came from in the source documents,
    including page number, chunk ID, confidence score, and text citation.
    
    Attributes:
        field_name: Name of the field in the schema
        field_value: The extracted value
        source_page: Page number where value was found (1-indexed)
        source_chunk: Chunk ID where value was found
        source_file: Source file path
        confidence: Confidence score (0.0 to 1.0)
        citation: Text snippet showing context around the value
        extraction_method: Method used for extraction (e.g., "text", "ocr", "table")
        metadata: Additional metadata
    """
    field_name: str
    field_value: Any
    source_page: int | None = None
    source_chunk: int | None = None
    source_file: str | None = None
    confidence: float = 1.0
    citation: str | None = None
    extraction_method: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "field_name": self.field_name,
            "field_value": self.field_value,
            "source_page": self.source_page,
            "source_chunk": self.source_chunk,
            "source_file": self.source_file,
            "confidence": self.confidence,
            "citation": self.citation,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> FieldProvenance:
        """Create from dictionary"""
        return cls(**data)


class ProvenanceTracker:
    """
    Track provenance for all fields in an extraction.
    
    Maintains a mapping of field names to their provenance information,
    including source location, confidence, and citations.
    
    Example:
        tracker = ProvenanceTracker()
        
        # Track a field
        tracker.track_field(
            field_name="invoice_number",
            value="INV-12345",
            chunk=chunk,
            confidence=0.95
        )
        
        # Get provenance
        prov = tracker.get_provenance("invoice_number")
        print(f"Found on page {prov.source_page}")
        print(f"Citation: {prov.citation}")
        
        # Get all provenance
        all_prov = tracker.get_all_provenance()
    """
    
    def __init__(self):
        """Initialize provenance tracker"""
        self.provenance_map: dict[str, FieldProvenance] = {}
        log.debug("provenance_tracker_initialized")
    
    def track_field(
        self,
        field_name: str,
        value: Any,
        chunk: Any = None,  # DocumentChunk
        confidence: float = 1.0,
        extraction_method: str = "text",
        metadata: dict[str, Any] | None = None
    ) -> FieldProvenance:
        """
        Track provenance for a single field.

        Args:
            field_name: Name of the field
            value: Extracted value
            chunk: DocumentChunk where value was found
            confidence: Confidence score (0.0 to 1.0)
            extraction_method: Method used for extraction
            metadata: Additional metadata
        
        Returns:
            FieldProvenance object
        
        Raises:
            ValueError: If confidence is not between 0.0 and 1.0
        """
        # Extract chunk information
        source_page = None
        source_chunk = None
        source_file = None
        chunk_content = None
        
        if chunk is not None:
            source_chunk = getattr(chunk, 'chunk_id', None)
            source_file = getattr(chunk, 'source_file', None)
            chunk_content = getattr(chunk, 'content', None)
            
            # Try to get page from metadata
            if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                # Handle page_range (e.g., [1, 3])
                if 'page_range' in chunk.metadata:
                    page_range = chunk.metadata['page_range']
                    if isinstance(page_range, (list, tuple)) and len(page_range) > 0:
                        source_page = page_range[0]  # Use first page
                # Handle single page
                elif 'page' in chunk.metadata:
                    source_page = chunk.metadata['page']
        
        # Generate citation
        citation = self._generate_citation(chunk_content, value)
        
        # Create provenance
        provenance = FieldProvenance(
            field_name=field_name,
            field_value=value,
            source_page=source_page,
            source_chunk=source_chunk,
            source_file=source_file,
            confidence=confidence,
            citation=citation,
            extraction_method=extraction_method,
            metadata=metadata or {}
        )
        
        # Store in map
        self.provenance_map[field_name] = provenance
        
        log.debug(
            "field_provenance_tracked",
            field_name=field_name,
            source_page=source_page,
            source_chunk=source_chunk,
            confidence=confidence
        )
        
        return provenance
    
    def track_nested_field(
        self,
        field_path: str,
        value: Any,
        chunk: Any = None,
        confidence: float = 1.0,
        extraction_method: str = "text",
        metadata: dict[str, Any] | None = None
    ) -> FieldProvenance:
        """
        Track provenance for a nested field (e.g., "address.city").
        
        Args:
            field_path: Dot-separated path to field (e.g., "address.city")
            value: Extracted value
            chunk: DocumentChunk where value was found
            confidence: Confidence score
            extraction_method: Method used for extraction
            metadata: Additional metadata
        
        Returns:
            FieldProvenance object
        """
        return self.track_field(
            field_name=field_path,
            value=value,
            chunk=chunk,
            confidence=confidence,
            extraction_method=extraction_method,
            metadata=metadata
        )
    
    def get_provenance(self, field_name: str) -> FieldProvenance | None:
        """
        Get provenance for a specific field.
        
        Args:
            field_name: Name of the field
        
        Returns:
            FieldProvenance object or None if not tracked
        """
        return self.provenance_map.get(field_name)
    
    def get_all_provenance(self) -> dict[str, FieldProvenance]:
        """
        Get provenance for all tracked fields.
        
        Returns:
            Dictionary mapping field names to FieldProvenance objects
        """
        return self.provenance_map.copy()
    
    def to_dict(self) -> dict[str, dict]:
        """
        Convert all provenance to dictionary format.
        
        Returns:
            Dictionary mapping field names to provenance dicts
        """
        return {
            field_name: prov.to_dict()
            for field_name, prov in self.provenance_map.items()
        }
    
    def merge(self, other: ProvenanceTracker, strategy: str = "highest_confidence") -> None:
        """
        Merge provenance from another tracker.
        
        Args:
            other: Another ProvenanceTracker
            strategy: Merge strategy ("highest_confidence", "first", "last")
        
        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in ["highest_confidence", "first", "last"]:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        for field_name, other_prov in other.provenance_map.items():
            if field_name not in self.provenance_map:
                # New field, just add it
                self.provenance_map[field_name] = other_prov
            else:
                # Field exists, apply merge strategy
                existing_prov = self.provenance_map[field_name]
                
                if strategy == "highest_confidence":
                    if other_prov.confidence > existing_prov.confidence:
                        self.provenance_map[field_name] = other_prov
                elif strategy == "first":
                    # Keep existing (first)
                    pass
                elif strategy == "last":
                    # Use new (last)
                    self.provenance_map[field_name] = other_prov
        
        log.debug(
            "provenance_merged",
            strategy=strategy,
            total_fields=len(self.provenance_map)
        )
    
    def _generate_citation(self, content: Any, value: Any, context_chars: int = 50) -> str | None:
        """
        Generate a text citation showing context around the value.
        
        Args:
            content: Chunk content (str or bytes)
            value: The extracted value
            context_chars: Number of characters to show before/after value
        
        Returns:
            Citation string or None if value not found
        """
        if content is None or value is None:
            return None
        
        # Convert content to string
        if isinstance(content, bytes):
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception:
                return None
        else:
            text = str(content)
        
        # Convert value to string
        value_str = str(value)
        
        # Find value in text (case-insensitive)
        text_lower = text.lower()
        value_lower = value_str.lower()
        
        idx = text_lower.find(value_lower)
        
        if idx == -1:
            # Value not found in text
            return None
        
        # Extract context
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(value_str) + context_chars)
        
        # Build citation with ellipsis
        citation_parts = []
        
        if start > 0:
            citation_parts.append("...")
        
        citation_parts.append(text[start:end])
        
        if end < len(text):
            citation_parts.append("...")
        
        citation = "".join(citation_parts)
        
        # Clean up whitespace
        citation = " ".join(citation.split())
        
        return citation

