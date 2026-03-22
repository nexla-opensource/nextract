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
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
    "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "local": [],
}


def has_provider_credentials(provider: str) -> bool:
    """Check if required credentials for a provider are available."""
    required = PROVIDER_CREDENTIALS.get(provider, [])
    if not required:
        return True
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
    """Create a simple text file simulating a document for testing."""
    doc_path = tmp_path / "sample_invoice.txt"
    doc_path.write_text(sample_text_content)
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
