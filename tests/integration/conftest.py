"""
Shared fixtures and configuration for integration tests.

This module provides:
- Provider credential detection and skip logic
- Sample document fixtures
- Schema fixtures
- Common test utilities
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCUMENTS_DIR = FIXTURES_DIR / "documents"
SCHEMAS_DIR = FIXTURES_DIR / "schemas"


PROVIDER_CREDENTIALS: Dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
    "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "cohere": ["CO_API_KEY"],
    "local": [],
}

# Providers that require ALL listed env vars (vs any one of them)
PROVIDER_CREDENTIALS_ALL: set[str] = {"azure", "aws"}


def has_provider_credentials(provider: str) -> bool:
    """Check if required credentials for a provider are available.

    For providers in PROVIDER_CREDENTIALS_ALL, ALL env vars must be set.
    For others, ANY one env var is sufficient.
    """
    required = PROVIDER_CREDENTIALS.get(provider, [])
    if not required:
        return True
    if provider in PROVIDER_CREDENTIALS_ALL:
        return all(os.getenv(key) for key in required)
    return any(os.getenv(key) for key in required)


def skip_without_credentials(provider: str):
    """Pytest marker to skip tests when provider credentials are missing."""
    return pytest.mark.skipif(
        not has_provider_credentials(provider),
        reason=f"Missing credentials for provider '{provider}'",
    )


requires_openai = skip_without_credentials("openai")
requires_anthropic = skip_without_credentials("anthropic")
requires_google = skip_without_credentials("google")
requires_azure = skip_without_credentials("azure")
requires_aws = skip_without_credentials("aws")
requires_cohere = skip_without_credentials("cohere")


@pytest.fixture
def simple_schema() -> Dict[str, Any]:
    """Simple flat schema for basic extraction tests."""
    return {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "description": "The invoice number"},
            "total": {"type": "number", "description": "Total amount"},
            "date": {"type": "string", "description": "Invoice date"},
        },
        "required": ["invoice_number", "total"],
    }


@pytest.fixture
def nested_schema() -> Dict[str, Any]:
    """Nested schema with complex structure."""
    return {
        "type": "object",
        "properties": {
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "tax_id": {"type": "string"},
                },
                "required": ["name"],
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"},
                    },
                    "required": ["description", "quantity"],
                },
            },
            "totals": {
                "type": "object",
                "properties": {
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "grand_total": {"type": "number"},
                },
            },
        },
        "required": ["vendor", "line_items"],
    }


@pytest.fixture
def array_schema() -> Dict[str, Any]:
    """Schema for extracting arrays of items."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["name"],
        },
    }


@pytest.fixture
def invalid_schema() -> Dict[str, Any]:
    """Invalid schema for error testing."""
    return {
        "type": "invalid_type",
        "properties": "not_a_dict",
    }


@pytest.fixture
def sample_text_content() -> str:
    """Sample text content for text-based extraction."""
    return """
    INVOICE
    
    Invoice Number: INV-2024-001
    Date: January 15, 2024
    
    Bill To:
    Acme Corporation
    123 Business Street
    New York, NY 10001
    
    Items:
    1. Widget A - Qty: 10 - $25.00 each - $250.00
    2. Widget B - Qty: 5 - $45.00 each - $225.00
    3. Service Fee - $50.00
    
    Subtotal: $525.00
    Tax (8%): $42.00
    Total: $567.00
    
    Payment due within 30 days.
    """


@pytest.fixture
def sample_pdf_path(tmp_path: Path, sample_text_content: str) -> Path:
    """Create a simple PDF file simulating a document for testing."""
    doc_path = tmp_path / "sample_invoice.pdf"
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), sample_text_content)
        doc.save(str(doc_path))
        doc.close()
    except ImportError:
        # Fallback: create a minimal valid PDF
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
        )
        doc_path.write_bytes(pdf_bytes)
    return doc_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Create an empty file for edge case testing."""
    empty_path = tmp_path / "empty_file.txt"
    empty_path.touch()
    return empty_path


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    """Create a file with random bytes simulating corruption."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 100)
    return corrupt_path


@pytest.fixture
def schema_file(tmp_path: Path, simple_schema: Dict[str, Any]) -> Path:
    """Write simple schema to a JSON file."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(simple_schema, indent=2))
    return schema_path


@pytest.fixture
def plan_config_file(tmp_path: Path) -> Path:
    """Create a valid plan configuration file."""
    plan = {
        "extractor": {
            "name": "text",
            "provider": {"name": "openai", "model": "gpt-4o"},
        },
        "chunker": {"name": "semantic", "chunk_size": 2000},
        "num_passes": 1,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path


@pytest.fixture
def invalid_plan_config_file(tmp_path: Path) -> Path:
    """Create an invalid plan configuration file."""
    plan = {
        "extractor": {
            "name": "textract",
            "provider": {"name": "openai", "model": "gpt-4o"},
        },
        "chunker": {"name": "semantic"},
    }
    plan_path = tmp_path / "invalid_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path
