"""
Integration tests for schema suggestion.
"""

import pytest

from tests.integration.conftest import requires_openai

from nextract.core import ProviderConfig
from nextract.schema import SchemaGenerator


@pytest.mark.integration
class TestSchemaGeneratorUnit:
    """Unit tests for SchemaGenerator."""

    def test_generator_initialization(self):
        """SchemaGenerator initializes with provider config."""
        config = ProviderConfig(name="openai", model="gpt-4o")
        generator = SchemaGenerator(provider=config)
        
        assert generator is not None


@pytest.mark.integration
@requires_openai
class TestSchemaGeneratorLive:
    """Live tests requiring OpenAI credentials."""

    @pytest.mark.slow
    def test_suggest_schema_from_text(self, sample_pdf_path):
        """Test schema suggestion from sample document."""
        config = ProviderConfig(name="openai", model="gpt-4o-mini")
        generator = SchemaGenerator(provider=config)
        
        schema = generator.suggest_schema(
            sample_documents=[str(sample_pdf_path)],
            prompt="Extract invoice information including invoice number, date, and total",
        )
        
        assert schema is not None
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema

    @pytest.mark.slow
    def test_suggest_schema_with_examples(self, sample_pdf_path, tmp_path):
        """Test schema suggestion with example outputs."""
        config = ProviderConfig(name="openai", model="gpt-4o-mini")
        generator = SchemaGenerator(provider=config)
        
        examples = [
            {"invoice_number": "INV-001", "total": 500.00, "date": "2024-01-15"},
        ]
        
        schema = generator.suggest_schema(
            sample_documents=[str(sample_pdf_path)],
            prompt="Extract invoice data",
            examples=examples,
        )
        
        assert schema is not None
        assert "properties" in schema

    def test_save_schema(self, tmp_path):
        """Test saving generated schema to file."""
        config = ProviderConfig(name="openai", model="gpt-4o")
        generator = SchemaGenerator(provider=config)
        
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }
        
        output_path = tmp_path / "output_schema.json"
        generator.save_schema(schema, output_path)
        
        assert output_path.exists()
        import json
        saved = json.loads(output_path.read_text())
        assert saved == schema
