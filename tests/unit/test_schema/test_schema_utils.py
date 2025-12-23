"""
Unit tests for schema utilities.

Tests:
- validate_json_schema
- augment_schema_with_extra
- build_output_type
- is_pydantic_model
- to_json_schema
"""

import pytest
from pydantic import BaseModel

from nextract.schema import (
    augment_schema_with_extra,
    build_output_type,
    is_pydantic_model,
    to_json_schema,
    validate_json_schema,
)


class TestValidateJsonSchema:
    """Tests for validate_json_schema."""

    def test_valid_schema(self, simple_schema):
        """Valid schema should not raise."""
        validate_json_schema(simple_schema)

    def test_non_dict_raises(self):
        """Non-dict schema should raise."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_json_schema("not a dict")

    def test_non_object_type_raises(self):
        """Non-object type should raise."""
        schema = {"type": "string", "properties": {}}
        with pytest.raises(ValueError, match="must have 'type': 'object'"):
            validate_json_schema(schema)

    def test_missing_type_warns(self, simple_schema):
        """Missing type should not raise but log warning."""
        schema = {"properties": {"field": {"type": "string"}}}
        validate_json_schema(schema)

    def test_missing_properties_warns(self):
        """Missing properties should not raise but log warning."""
        schema = {"type": "object"}
        validate_json_schema(schema)


class TestAugmentSchemaWithExtra:
    """Tests for augment_schema_with_extra."""

    def test_no_augment_when_false(self, simple_schema):
        """Schema unchanged when include_extra=False."""
        result = augment_schema_with_extra(simple_schema, include_extra=False)
        assert result == simple_schema

    def test_adds_extra_field_when_true(self, simple_schema):
        """Adds 'extra' field when include_extra=True."""
        result = augment_schema_with_extra(simple_schema, include_extra=True)
        assert "extra" in result["properties"]
        assert result["properties"]["extra"]["type"] == "object"
        assert result["properties"]["extra"]["additionalProperties"] is True

    def test_preserves_original_properties(self, simple_schema):
        """Original properties are preserved."""
        result = augment_schema_with_extra(simple_schema, include_extra=True)
        assert "name" in result["properties"]
        assert "age" in result["properties"]

    def test_preserves_required(self, simple_schema):
        """Required fields are preserved."""
        result = augment_schema_with_extra(simple_schema, include_extra=True)
        assert result["required"] == simple_schema["required"]


class TestBuildOutputType:
    """Tests for build_output_type."""

    def test_returns_pydantic_model_unchanged(self):
        """Pydantic model should be returned unchanged."""

        class TestModel(BaseModel):
            name: str
            age: int

        result = build_output_type(TestModel, include_extra=False)
        assert result is TestModel

    def test_returns_structured_dict_for_json_schema(self, simple_schema):
        """JSON schema should return StructuredDict."""
        result = build_output_type(simple_schema, include_extra=False)
        assert result is not None


class TestIsPydanticModel:
    """Tests for is_pydantic_model."""

    def test_pydantic_model_returns_true(self):
        """Pydantic model should return True."""

        class TestModel(BaseModel):
            name: str

        assert is_pydantic_model(TestModel) is True

    def test_dict_returns_false(self):
        """Dict should return False."""
        assert is_pydantic_model({"type": "object"}) is False

    def test_string_returns_false(self):
        """String should return False."""
        assert is_pydantic_model("not a model") is False

    def test_none_returns_false(self):
        """None should return False."""
        assert is_pydantic_model(None) is False


class TestToJsonSchema:
    """Tests for to_json_schema."""

    def test_dict_returns_unchanged(self, simple_schema):
        """Dict schema should be returned unchanged."""
        result = to_json_schema(simple_schema)
        assert result == simple_schema

    def test_pydantic_model_converted(self):
        """Pydantic model should be converted to JSON schema."""

        class TestModel(BaseModel):
            name: str
            age: int

        result = to_json_schema(TestModel)
        assert isinstance(result, dict)
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]
