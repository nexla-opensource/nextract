"""
Shared fixtures for unit tests.

This module provides:
- Mock schemas (simple, nested, array)
- Mock chunks and documents
- Mock extraction results
- Common test utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


@dataclass
class MockChunk:
    """Mock DocumentChunk for testing."""
    chunk_id: int
    content: str
    source_file: str
    metadata: dict
    chunk_type: str = "text"


@dataclass
class MockDocumentArtifact:
    """Mock DocumentArtifact for testing."""
    source_path: str
    mime_type: str
    content: bytes = b""
    text: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def simple_schema() -> Dict[str, Any]:
    """Simple flat schema for basic tests."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name field"},
            "age": {"type": "integer", "description": "Age field"},
            "city": {"type": "string", "description": "City field"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def nested_schema() -> Dict[str, Any]:
    """Nested schema with complex structure."""
    return {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                        },
                    },
                },
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
            },
        },
        "required": ["person"],
    }


@pytest.fixture
def array_schema() -> Dict[str, Any]:
    """Schema for array extraction."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
                "quantity": {"type": "integer"},
                "price": {"type": "number"},
            },
            "required": ["item_name"],
        },
    }


@pytest.fixture
def mock_chunk() -> MockChunk:
    """Single mock chunk."""
    return MockChunk(
        chunk_id=0,
        content="This is test content with Invoice Number INV-12345 and total $500.",
        source_file="test.pdf",
        metadata={"page": 1},
    )


@pytest.fixture
def mock_chunks() -> List[MockChunk]:
    """Multiple mock chunks."""
    return [
        MockChunk(
            chunk_id=i,
            content=f"Content for chunk {i}",
            source_file="test.pdf",
            metadata={"page": i + 1},
        )
        for i in range(5)
    ]


@pytest.fixture
def complete_extraction_result() -> Dict[str, Any]:
    """Complete extraction result with all fields populated."""
    return {
        "name": "John Doe",
        "age": 30,
        "city": "New York",
        "email": "john@example.com",
    }


@pytest.fixture
def partial_extraction_result() -> Dict[str, Any]:
    """Partial extraction result with some empty fields."""
    return {
        "name": "John Doe",
        "age": None,
        "city": "",
        "email": "N/A",
    }


@pytest.fixture
def sample_text_document() -> str:
    """Sample text content for chunking tests."""
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
