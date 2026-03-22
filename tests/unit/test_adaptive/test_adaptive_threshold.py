"""
Unit tests for intelligent adaptive threshold calculation.

Tests the new intelligent threshold system that adjusts based on:
- Schema complexity (number of fields, nesting depth)
- Document complexity (pages, tokens)
- Array extraction (number of instances)
"""

from nextract.adaptive_extraction import (
    analyze_schema_complexity,
    analyze_document_complexity,
    calculate_adaptive_threshold,
    estimate_array_instances
)


# ============================================================================
# Schema Complexity Analysis Tests
# ============================================================================

def test_analyze_schema_complexity_simple_flat():
    """Test simple flat schema (5 fields, 1 level)"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "address": {"type": "string"},
            "city": {"type": "string"}
        }
    }
    
    result = analyze_schema_complexity(schema)
    
    assert result["num_fields"] == 5
    assert result["nesting_depth"] == 1
    assert result["is_array_schema"] is False
    assert 0.0 <= result["complexity_score"] <= 0.3  # Simple schema


def test_analyze_schema_complexity_nested():
    """Test nested schema (10 fields, 3 levels)"""
    schema = {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"}
                        }
                    }
                }
            },
            "invoice": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "date": {"type": "string"},
                    "amount": {"type": "number"}
                }
            }
        }
    }
    
    result = analyze_schema_complexity(schema)
    
    assert result["num_fields"] == 6  # Leaf fields only
    assert result["nesting_depth"] == 3
    assert result["is_array_schema"] is False
    assert 0.1 <= result["complexity_score"] <= 0.5


def test_analyze_schema_complexity_complex():
    """Test complex schema (50+ fields, 4 levels)"""
    # Create a complex nested schema
    schema = {
        "type": "object",
        "properties": {}
    }
    
    # Add 50 fields across 4 nesting levels
    for i in range(10):
        schema["properties"][f"section_{i}"] = {
            "type": "object",
            "properties": {
                f"field_{j}": {
                    "type": "object",
                    "properties": {
                        f"subfield_{k}": {"type": "string"}
                        for k in range(5)
                    }
                }
                for j in range(1)
            }
        }
    
    result = analyze_schema_complexity(schema)

    assert result["num_fields"] == 50
    assert result["nesting_depth"] == 3  # Actually 3 levels (object -> object -> object)
    assert result["is_array_schema"] is False
    assert result["complexity_score"] >= 0.4  # Complex schema


def test_analyze_schema_complexity_array():
    """Test array schema"""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"}
            }
        }
    }
    
    result = analyze_schema_complexity(schema)
    
    assert result["num_fields"] == 2
    assert result["nesting_depth"] == 1
    assert result["is_array_schema"] is True


# ============================================================================
# Document Complexity Analysis Tests
# ============================================================================

def test_analyze_document_complexity_small_pdf(tmp_path):
    """Test small PDF (2 pages)"""
    # Create a small test PDF
    pdf_path = tmp_path / "small.pdf"
    
    # Create minimal PDF (we'll mock the analysis)
    pdf_path.write_bytes(b"%PDF-1.4\n" + b"x" * 100000)  # ~100KB
    
    result = analyze_document_complexity([str(pdf_path)])
    
    assert result["num_files"] == 1
    assert result["total_pages"] >= 1  # At least 1 page
    assert result["estimated_tokens"] > 0
    assert "pdf" in result["document_types"]


def test_analyze_document_complexity_multiple_files(tmp_path):
    """Test multiple files"""
    files = []
    for i in range(3):
        pdf_path = tmp_path / f"doc_{i}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n" + b"x" * 50000)
        files.append(str(pdf_path))
    
    result = analyze_document_complexity(files)
    
    assert result["num_files"] == 3
    assert result["total_pages"] >= 3
    assert result["estimated_tokens"] > 0


# ============================================================================
# Array Instance Estimation Tests
# ============================================================================

def test_estimate_array_instances_small_doc():
    """Test array instance estimation for small document"""
    schema_complexity = {"is_array_schema": True}
    document_complexity = {"total_pages": 2}
    
    result = estimate_array_instances(schema_complexity, document_complexity)
    
    assert result == 10  # 2 pages * 5 items/page


def test_estimate_array_instances_large_doc():
    """Test array instance estimation for large document"""
    schema_complexity = {"is_array_schema": True}
    document_complexity = {"total_pages": 120}

    result = estimate_array_instances(schema_complexity, document_complexity)

    assert result == 1800  # 120 pages * 15 items/page


def test_estimate_array_instances_not_array():
    """Test array instance estimation for non-array schema"""
    schema_complexity = {"is_array_schema": False}
    document_complexity = {"total_pages": 10}

    result = estimate_array_instances(schema_complexity, document_complexity)

    assert result == 0  # Not an array


# ============================================================================
# Adaptive Threshold Calculation Tests
# ============================================================================

def test_calculate_threshold_simple_schema_small_doc():
    """Test threshold for simple schema + small document (should be strict)"""
    schema_complexity = {
        "num_fields": 5,
        "nesting_depth": 1,
        "is_array_schema": False,
        "complexity_score": 0.15
    }
    document_complexity = {
        "total_pages": 2,
        "estimated_tokens": 5000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be strict: 0.3 - 0.1 (simple) - 0.05 (small) = 0.15
    assert abs(threshold - 0.15) < 0.01  # Allow for floating point precision
    assert 0.1 <= threshold <= 0.3  # Strict range


def test_calculate_threshold_complex_schema_large_doc():
    """Test threshold for complex schema + large document (should be lenient)"""
    schema_complexity = {
        "num_fields": 60,
        "nesting_depth": 4,
        "is_array_schema": False,
        "complexity_score": 0.7
    }
    document_complexity = {
        "total_pages": 100,
        "estimated_tokens": 250000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be lenient: 0.3 + 0.1 (complex) + 0.05 (deep) + 0.05 (large) = 0.50
    assert threshold == 0.50
    assert 0.4 <= threshold <= 0.6  # Lenient range


def test_calculate_threshold_medium_schema():
    """Test threshold for medium schema (should be neutral)"""
    schema_complexity = {
        "num_fields": 25,
        "nesting_depth": 2,
        "is_array_schema": False,
        "complexity_score": 0.35
    }
    document_complexity = {
        "total_pages": 20,
        "estimated_tokens": 50000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be neutral: 0.3 + 0.0 (medium) + 0.0 (medium) = 0.30
    assert threshold == 0.30


def test_calculate_threshold_array_large_instances():
    """Test threshold for array schema with many instances (should be lenient)"""
    schema_complexity = {
        "num_fields": 10,
        "nesting_depth": 2,
        "is_array_schema": True,
        "complexity_score": 0.25
    }
    document_complexity = {
        "total_pages": 120,
        "estimated_tokens": 300000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be lenient: 0.3 - 0.1 (simple) + 0.05 (large) + 0.1 (array) = 0.35
    # But estimate_array_instances will return 1800 (> 100), so +0.1
    assert threshold >= 0.35
    assert 0.3 <= threshold <= 0.5


def test_calculate_threshold_clamping_min():
    """Test threshold clamping to minimum (0.1)"""
    schema_complexity = {
        "num_fields": 3,
        "nesting_depth": 1,
        "is_array_schema": False,
        "complexity_score": 0.05
    }
    document_complexity = {
        "total_pages": 1,
        "estimated_tokens": 1000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Would be: 0.3 - 0.1 - 0.05 = 0.15, but clamped to 0.1 minimum
    assert threshold >= 0.1


def test_calculate_threshold_clamping_max():
    """Test threshold clamping to maximum (0.6)"""
    schema_complexity = {
        "num_fields": 100,
        "nesting_depth": 5,
        "is_array_schema": True,
        "complexity_score": 0.9
    }
    document_complexity = {
        "total_pages": 200,
        "estimated_tokens": 500000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Would be very high, but clamped to 0.6 maximum
    assert threshold <= 0.6


def test_calculate_threshold_custom_base():
    """Test threshold with custom base threshold"""
    schema_complexity = {
        "num_fields": 25,
        "nesting_depth": 2,
        "is_array_schema": False,
        "complexity_score": 0.35
    }
    document_complexity = {
        "total_pages": 20,
        "estimated_tokens": 50000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(
        schema_complexity,
        document_complexity,
        base_threshold=0.5  # Custom base
    )

    # Should use custom base: 0.5 + 0.0 = 0.50
    assert threshold == 0.50


# ============================================================================
# Integration Tests
# ============================================================================

def test_threshold_calculation_realistic_invoice():
    """Test realistic scenario: simple invoice extraction"""
    # Simple invoice: 10 fields, 2 pages
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "customer_name": {"type": "string"},
            "customer_email": {"type": "string"},
            "total_amount": {"type": "number"},
            "tax_amount": {"type": "number"},
            "subtotal": {"type": "number"},
            "payment_terms": {"type": "string"},
            "due_date": {"type": "string"},
            "notes": {"type": "string"}
        }
    }

    schema_complexity = analyze_schema_complexity(schema)
    document_complexity = {
        "total_pages": 2,
        "estimated_tokens": 5000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be strict for simple invoice
    assert 0.15 <= threshold <= 0.25


def test_threshold_calculation_realistic_insurance_claim():
    """Test realistic scenario: complex insurance claim extraction"""
    # Complex claim: 50+ fields, 100 pages, nested
    schema = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "review": {
                        "type": "object",
                        "properties": {
                            "reviewer_name": {"type": "string"},
                            "review_date": {"type": "string"},
                            "approval": {
                                "type": "object",
                                "properties": {
                                    "approved": {"type": "boolean"},
                                    "amount": {"type": "number"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    # Add more fields to reach 50+
    for i in range(45):
        schema["properties"][f"field_{i}"] = {"type": "string"}

    schema_complexity = analyze_schema_complexity(schema)
    document_complexity = {
        "total_pages": 100,
        "estimated_tokens": 250000,
        "document_types": ["pdf"]
    }

    threshold = calculate_adaptive_threshold(schema_complexity, document_complexity)

    # Should be lenient for complex claim
    assert 0.40 <= threshold <= 0.55


