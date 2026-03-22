"""
Integration tests for schema validation.
"""

import pytest

from nextract.validate import SchemaValidator


@pytest.mark.integration
class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_valid_data(self, simple_schema):
        """Valid data should pass validation."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001", "total": 500.00}
        
        result = validator.validate(data, simple_schema)
        
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_missing_required_field(self, simple_schema):
        """Missing required field should fail validation."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001"}
        
        result = validator.validate(data, simple_schema)
        
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_wrong_type(self, simple_schema):
        """Wrong type should fail validation."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001", "total": "not_a_number"}
        
        result = validator.validate(data, simple_schema)
        
        assert result.valid is False

    def test_validate_extra_fields_allowed(self, simple_schema):
        """Extra fields should be allowed by default."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001", "total": 500.00, "extra_field": "value"}
        
        result = validator.validate(data, simple_schema)
        
        assert result.valid is True

    def test_completeness_score_all_filled(self, simple_schema):
        """Completeness should be 1.0 when all required fields are filled."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001", "total": 500.00}
        
        result = validator.validate(data, simple_schema)
        
        assert result.metadata.get("completeness") == 1.0

    def test_completeness_score_partial(self, simple_schema):
        """Completeness should reflect partial fill."""
        validator = SchemaValidator()
        data = {"invoice_number": "INV-001", "total": None}
        
        result = validator.validate(data, simple_schema)
        
        completeness = result.metadata.get("completeness")
        assert completeness < 1.0


@pytest.mark.integration
class TestNestedSchemaValidation:
    """Tests for nested schema validation."""

    def test_validate_nested_valid(self, nested_schema):
        """Valid nested data should pass."""
        validator = SchemaValidator()
        data = {
            "vendor": {"name": "Acme Corp", "address": "123 Main St"},
            "line_items": [
                {"description": "Widget", "quantity": 10, "unit_price": 25.00}
            ],
        }
        
        result = validator.validate(data, nested_schema)
        
        assert result.valid is True

    def test_validate_nested_missing_required(self, nested_schema):
        """Missing nested required field should fail."""
        validator = SchemaValidator()
        data = {
            "vendor": {"address": "123 Main St"},
            "line_items": [{"description": "Widget", "quantity": 10}],
        }
        
        result = validator.validate(data, nested_schema)
        
        assert result.valid is False


@pytest.mark.integration
class TestArraySchemaValidation:
    """Tests for array schema validation."""

    def test_validate_array_valid(self, array_schema):
        """Valid array should pass."""
        validator = SchemaValidator()
        data = [
            {"name": "Item 1", "value": 100},
            {"name": "Item 2", "value": 200},
        ]
        
        result = validator.validate(data, array_schema)
        
        assert result.valid is True

    def test_validate_array_invalid_item(self, array_schema):
        """Array with invalid item should fail."""
        validator = SchemaValidator()
        data = [
            {"name": "Item 1", "value": 100},
            {"value": 200},
        ]
        
        result = validator.validate(data, array_schema)
        
        assert result.valid is False

    def test_validate_empty_array(self, array_schema):
        """Empty array should be valid."""
        validator = SchemaValidator()
        data = []
        
        result = validator.validate(data, array_schema)
        
        assert result.valid is True
