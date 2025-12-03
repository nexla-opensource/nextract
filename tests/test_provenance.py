"""
Tests for provenance tracking functionality.

Tests:
- FieldProvenance creation and validation
- ProvenanceTracker tracking and retrieval
- Citation generation
- Provenance merging
"""

import pytest
from dataclasses import dataclass
from nextract.provenance import FieldProvenance, ProvenanceTracker


@dataclass
class MockChunk:
    """Mock DocumentChunk for testing"""
    chunk_id: int
    content: str
    source_file: str
    metadata: dict


class TestFieldProvenance:
    """Test FieldProvenance dataclass"""
    
    def test_creation(self):
        """Test creating FieldProvenance"""
        prov = FieldProvenance(
            field_name="invoice_number",
            field_value="INV-12345",
            source_page=1,
            source_chunk=0,
            source_file="invoice.pdf",
            confidence=0.95
        )
        
        assert prov.field_name == "invoice_number"
        assert prov.field_value == "INV-12345"
        assert prov.source_page == 1
        assert prov.confidence == 0.95
    
    def test_confidence_validation(self):
        """Test confidence score validation"""
        # Valid confidence
        prov = FieldProvenance(
            field_name="test",
            field_value="value",
            confidence=0.5
        )
        assert prov.confidence == 0.5
        
        # Invalid confidence (too high)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            FieldProvenance(
                field_name="test",
                field_value="value",
                confidence=1.5
            )
        
        # Invalid confidence (negative)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            FieldProvenance(
                field_name="test",
                field_value="value",
                confidence=-0.1
            )
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        prov = FieldProvenance(
            field_name="amount",
            field_value=1234.56,
            source_page=2,
            confidence=0.98,
            citation="...Total: $1,234.56..."
        )
        
        prov_dict = prov.to_dict()
        
        assert prov_dict["field_name"] == "amount"
        assert prov_dict["field_value"] == 1234.56
        assert prov_dict["source_page"] == 2
        assert prov_dict["confidence"] == 0.98
        assert prov_dict["citation"] == "...Total: $1,234.56..."
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            "field_name": "vendor",
            "field_value": "Acme Corp",
            "source_page": 1,
            "source_chunk": 0,
            "source_file": "invoice.pdf",
            "confidence": 0.99,
            "citation": "...Vendor: Acme Corp...",
            "extraction_method": "text",
            "metadata": {}
        }
        
        prov = FieldProvenance.from_dict(data)
        
        assert prov.field_name == "vendor"
        assert prov.field_value == "Acme Corp"
        assert prov.confidence == 0.99


class TestProvenanceTracker:
    """Test ProvenanceTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ProvenanceTracker()
        assert len(tracker.provenance_map) == 0
    
    def test_track_field_basic(self):
        """Test tracking a field"""
        tracker = ProvenanceTracker()
        
        chunk = MockChunk(
            chunk_id=0,
            content="Invoice Number: INV-12345",
            source_file="invoice.pdf",
            metadata={"page": 1}
        )
        
        prov = tracker.track_field(
            field_name="invoice_number",
            value="INV-12345",
            chunk=chunk,
            confidence=0.95
        )
        
        assert prov.field_name == "invoice_number"
        assert prov.field_value == "INV-12345"
        assert prov.source_page == 1
        assert prov.source_chunk == 0
        assert prov.source_file == "invoice.pdf"
        assert prov.confidence == 0.95
    
    def test_track_field_with_page_range(self):
        """Test tracking field with page_range metadata"""
        tracker = ProvenanceTracker()
        
        chunk = MockChunk(
            chunk_id=1,
            content="Some content",
            source_file="doc.pdf",
            metadata={"page_range": [3, 5]}
        )
        
        prov = tracker.track_field(
            field_name="test_field",
            value="test_value",
            chunk=chunk
        )
        
        # Should use first page of range
        assert prov.source_page == 3
    
    def test_track_field_without_chunk(self):
        """Test tracking field without chunk"""
        tracker = ProvenanceTracker()
        
        prov = tracker.track_field(
            field_name="manual_field",
            value="manual_value",
            chunk=None,
            confidence=1.0
        )
        
        assert prov.field_name == "manual_field"
        assert prov.source_page is None
        assert prov.source_chunk is None
        assert prov.source_file is None
    
    def test_get_provenance(self):
        """Test retrieving provenance"""
        tracker = ProvenanceTracker()
        
        tracker.track_field(
            field_name="field1",
            value="value1",
            chunk=None
        )
        
        prov = tracker.get_provenance("field1")
        assert prov is not None
        assert prov.field_name == "field1"
        
        # Non-existent field
        prov = tracker.get_provenance("nonexistent")
        assert prov is None
    
    def test_get_all_provenance(self):
        """Test getting all provenance"""
        tracker = ProvenanceTracker()
        
        tracker.track_field("field1", "value1", chunk=None)
        tracker.track_field("field2", "value2", chunk=None)
        tracker.track_field("field3", "value3", chunk=None)
        
        all_prov = tracker.get_all_provenance()
        
        assert len(all_prov) == 3
        assert "field1" in all_prov
        assert "field2" in all_prov
        assert "field3" in all_prov
    
    def test_to_dict(self):
        """Test converting all provenance to dict"""
        tracker = ProvenanceTracker()
        
        tracker.track_field("field1", "value1", chunk=None, confidence=0.9)
        tracker.track_field("field2", "value2", chunk=None, confidence=0.8)
        
        prov_dict = tracker.to_dict()
        
        assert len(prov_dict) == 2
        assert prov_dict["field1"]["field_name"] == "field1"
        assert prov_dict["field1"]["confidence"] == 0.9
        assert prov_dict["field2"]["confidence"] == 0.8
    
    def test_merge_highest_confidence(self):
        """Test merging with highest_confidence strategy"""
        tracker1 = ProvenanceTracker()
        tracker1.track_field("field1", "value1", chunk=None, confidence=0.8)
        tracker1.track_field("field2", "value2", chunk=None, confidence=0.9)
        
        tracker2 = ProvenanceTracker()
        tracker2.track_field("field1", "value1_new", chunk=None, confidence=0.95)  # Higher
        tracker2.track_field("field3", "value3", chunk=None, confidence=0.7)
        
        tracker1.merge(tracker2, strategy="highest_confidence")
        
        # field1 should be updated (higher confidence)
        assert tracker1.get_provenance("field1").confidence == 0.95
        
        # field2 should remain (not in tracker2)
        assert tracker1.get_provenance("field2").confidence == 0.9
        
        # field3 should be added
        assert tracker1.get_provenance("field3").confidence == 0.7
    
    def test_merge_first_strategy(self):
        """Test merging with first strategy"""
        tracker1 = ProvenanceTracker()
        tracker1.track_field("field1", "value1", chunk=None, confidence=0.8)
        
        tracker2 = ProvenanceTracker()
        tracker2.track_field("field1", "value1_new", chunk=None, confidence=0.95)
        
        tracker1.merge(tracker2, strategy="first")
        
        # Should keep first (existing)
        assert tracker1.get_provenance("field1").confidence == 0.8
    
    def test_merge_last_strategy(self):
        """Test merging with last strategy"""
        tracker1 = ProvenanceTracker()
        tracker1.track_field("field1", "value1", chunk=None, confidence=0.8)
        
        tracker2 = ProvenanceTracker()
        tracker2.track_field("field1", "value1_new", chunk=None, confidence=0.95)
        
        tracker1.merge(tracker2, strategy="last")
        
        # Should use last (new)
        assert tracker1.get_provenance("field1").confidence == 0.95
    
    def test_merge_invalid_strategy(self):
        """Test merging with invalid strategy"""
        tracker1 = ProvenanceTracker()
        tracker2 = ProvenanceTracker()
        
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            tracker1.merge(tracker2, strategy="invalid")
    
    @pytest.mark.skip(reason="Implementation pending: citation generation")
    def test_citation_generation(self):
        """Test citation text generation"""
        tracker = ProvenanceTracker()
        
        chunk = MockChunk(
            chunk_id=0,
            content="This is a test document. The invoice number is INV-12345. Please process it.",
            source_file="test.pdf",
            metadata={}
        )
        
        prov = tracker.track_field(
            field_name="invoice_number",
            value="INV-12345",
            chunk=chunk
        )
        
        # Should have citation with context
        assert prov.citation is not None
        assert "INV-12345" in prov.citation
        assert "..." in prov.citation  # Should have ellipsis
    
    def test_citation_value_not_found(self):
        """Test citation when value not in content"""
        tracker = ProvenanceTracker()
        
        chunk = MockChunk(
            chunk_id=0,
            content="This document has no invoice number",
            source_file="test.pdf",
            metadata={}
        )
        
        prov = tracker.track_field(
            field_name="invoice_number",
            value="INV-12345",
            chunk=chunk
        )
        
        # Citation should be None if value not found
        assert prov.citation is None
    
    def test_citation_case_insensitive(self):
        """Test citation generation is case-insensitive"""
        tracker = ProvenanceTracker()
        
        chunk = MockChunk(
            chunk_id=0,
            content="The INVOICE NUMBER is inv-12345",
            source_file="test.pdf",
            metadata={}
        )
        
        prov = tracker.track_field(
            field_name="invoice_number",
            value="INV-12345",  # Different case
            chunk=chunk
        )
        
        # Should still find it
        assert prov.citation is not None
        assert "inv-12345" in prov.citation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

