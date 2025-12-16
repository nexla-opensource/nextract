"""
Tests for nested schema support in adaptive extraction.

Tests cover:
- Helper functions (count_leaf_fields, get_nested_value, set_nested_value)
- Recursive missing field identification
- Focused schema creation from dot paths
- Deep merge of nested results
- Focused prompt generation
"""

import pytest
from nextract.adaptive_extraction import (
    count_leaf_fields,
    get_nested_value,
    set_nested_value,
    is_empty_value,
    identify_missing_fields,
    create_focused_schema,
    merge_extraction_results,
    create_focused_prompt,
)


class TestHelperFunctions:
    """Test helper functions for nested operations"""
    
    def test_count_leaf_fields_flat_schema(self):
        """Flat schema should count all fields"""
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"},
                "field3": {"type": "boolean"}
            }
        }
        assert count_leaf_fields(schema) == 3
    
    def test_count_leaf_fields_nested_schema(self):
        """Nested schema should count only leaf fields"""
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
                                "review_date": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        # Should count: claim_id, reviewer_name, review_date = 3
        assert count_leaf_fields(schema) == 3
    
    def test_count_leaf_fields_deeply_nested(self):
        """Deeply nested schema (4 levels)"""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "level4": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert count_leaf_fields(schema) == 1
    
    def test_get_nested_value_simple(self):
        """Get value from simple nested dict"""
        data = {"claim": {"review": {"reviewer_name": "John"}}}
        assert get_nested_value(data, "claim.review.reviewer_name") == "John"
    
    def test_get_nested_value_missing(self):
        """Get value from missing path returns None"""
        data = {"claim": {"review": {}}}
        assert get_nested_value(data, "claim.review.reviewer_name") is None
    
    def test_get_nested_value_partial_path(self):
        """Get value when intermediate key is missing"""
        data = {"claim": {}}
        assert get_nested_value(data, "claim.review.reviewer_name") is None
    
    def test_get_nested_value_non_dict(self):
        """Get value when intermediate value is not a dict"""
        data = {"claim": "not a dict"}
        assert get_nested_value(data, "claim.review.reviewer_name") is None
    
    def test_set_nested_value_creates_structure(self):
        """Set value creates nested structure if needed"""
        data = {}
        set_nested_value(data, "claim.review.reviewer_name", "John")
        assert data == {"claim": {"review": {"reviewer_name": "John"}}}
    
    def test_set_nested_value_preserves_existing(self):
        """Set value preserves existing fields"""
        data = {"claim": {"claim_id": "CLM-123", "review": {}}}
        set_nested_value(data, "claim.review.reviewer_name", "John")
        assert data["claim"]["claim_id"] == "CLM-123"
        assert data["claim"]["review"]["reviewer_name"] == "John"
    
    def test_set_nested_value_overwrites_non_dict(self):
        """Set value overwrites non-dict intermediate values"""
        data = {"claim": "not a dict"}
        set_nested_value(data, "claim.review.reviewer_name", "John")
        assert data == {"claim": {"review": {"reviewer_name": "John"}}}
    
    def test_is_empty_value_none(self):
        """None is empty"""
        assert is_empty_value(None) is True
    
    def test_is_empty_value_empty_string(self):
        """Empty string is empty"""
        assert is_empty_value("") is True
    
    def test_is_empty_value_empty_list(self):
        """Empty list is empty"""
        assert is_empty_value([]) is True
    
    def test_is_empty_value_empty_dict(self):
        """Empty dict is empty"""
        assert is_empty_value({}) is True
    
    def test_is_empty_value_placeholder(self):
        """Placeholder strings are empty"""
        assert is_empty_value("N/A") is True
        assert is_empty_value("n/a") is True
        assert is_empty_value("None") is True
        assert is_empty_value("Unknown") is True
        assert is_empty_value("Not Found") is True
    
    def test_is_empty_value_valid(self):
        """Valid values are not empty"""
        assert is_empty_value("John") is False
        assert is_empty_value(123) is False
        assert is_empty_value(["item"]) is False
        assert is_empty_value({"key": "value"}) is False


class TestRecursiveFieldCounting:
    """Test recursive missing field identification"""
    
    def test_identify_missing_fields_flat_schema(self):
        """Flat schema should identify missing fields"""
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"}
            }
        }
        result = {"field1": "value1", "field2": None, "field3": ""}
        
        missing = identify_missing_fields(result, schema, empty_threshold=0.0)
        assert set(missing) == {"field2", "field3"}
    
    def test_identify_missing_fields_nested_schema(self):
        """Nested schema should return dot-notation paths"""
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
                                "review_date": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        result = {
            "claim": {
                "claim_id": "CLM-123",
                "review": {
                    "reviewer_name": "John",
                    "review_date": None
                }
            }
        }
        
        missing = identify_missing_fields(result, schema, empty_threshold=0.0)
        assert missing == ["claim.review.review_date"]
    
    def test_identify_missing_fields_multiple_nested(self):
        """Multiple missing fields in different nested sections"""
        schema = {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                },
                "claimant": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }
            }
        }
        result = {
            "claim": {"claim_id": "CLM-123", "status": None},
            "claimant": {"name": "Alice", "email": ""}
        }
        
        missing = identify_missing_fields(result, schema, empty_threshold=0.0)
        assert set(missing) == {"claim.status", "claimant.email"}
    
    def test_identify_missing_fields_respects_threshold(self):
        """Should return empty list if below threshold"""
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"},
                "field4": {"type": "string"}
            }
        }
        result = {"field1": "value1", "field2": "value2", "field3": "value3", "field4": None}
        
        # 1/4 = 25% empty, threshold = 30%
        missing = identify_missing_fields(result, schema, empty_threshold=0.3)
        assert missing == []  # Below threshold
    
    def test_identify_missing_fields_above_threshold(self):
        """Should return fields if above threshold"""
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"}
            }
        }
        result = {"field1": "value1", "field2": None, "field3": None}
        
        # 2/3 = 66% empty, threshold = 30%
        missing = identify_missing_fields(result, schema, empty_threshold=0.3)
        assert set(missing) == {"field2", "field3"}


class TestFocusedSchemaCreation:
    """Test focused schema creation from dot paths"""
    
    def test_create_focused_schema_single_path(self):
        """Create schema for single dot path"""
        original = {
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
                                "review_date": {"type": "string", "description": "Review date"}
                            }
                        }
                    }
                }
            }
        }
        
        focused = create_focused_schema(original, ["claim.review.review_date"])
        
        # Should have nested structure
        assert "claim" in focused["properties"]
        assert "review" in focused["properties"]["claim"]["properties"]
        assert "review_date" in focused["properties"]["claim"]["properties"]["review"]["properties"]
        
        # Should preserve field schema
        field = focused["properties"]["claim"]["properties"]["review"]["properties"]["review_date"]
        assert field["type"] == "string"
        assert field["description"] == "Review date"
        
        # Should NOT include other fields
        assert "claim_id" not in focused["properties"]["claim"]["properties"]
        assert "reviewer_name" not in focused["properties"]["claim"]["properties"]["review"]["properties"]

    def test_create_focused_schema_multiple_paths_same_section(self):
        """Create schema for multiple paths in same section"""
        original = {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "object",
                    "properties": {
                        "review": {
                            "type": "object",
                            "properties": {
                                "review_date": {"type": "string"},
                                "review_comments": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

        focused = create_focused_schema(original, [
            "claim.review.review_date",
            "claim.review.review_comments"
        ])

        review_props = focused["properties"]["claim"]["properties"]["review"]["properties"]
        assert "review_date" in review_props
        assert "review_comments" in review_props

    def test_create_focused_schema_multiple_sections(self):
        """Create schema for paths in different sections"""
        original = {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"}
                    }
                },
                "claimant": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        focused = create_focused_schema(original, [
            "claim.claim_id",
            "claimant.name"
        ])

        assert "claim" in focused["properties"]
        assert "claimant" in focused["properties"]
        assert "claim_id" in focused["properties"]["claim"]["properties"]
        assert "name" in focused["properties"]["claimant"]["properties"]

    def test_create_focused_schema_flat_paths(self):
        """Create schema for flat (non-nested) paths"""
        original = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"}
            }
        }

        focused = create_focused_schema(original, ["field1"])

        assert "field1" in focused["properties"]
        assert "field2" not in focused["properties"]


class TestDeepMerge:
    """Test deep merge of nested results"""

    def test_merge_simple_nested(self):
        """Merge simple nested structure"""
        pass1 = {
            "claim": {
                "claim_id": "CLM-123",
                "review": {
                    "reviewer_name": "John",
                    "review_date": None
                }
            }
        }

        pass2 = {
            "claim": {
                "review": {
                    "review_date": "2025-01-15"
                }
            }
        }

        merged = merge_extraction_results(pass1, pass2, ["claim.review.review_date"])

        assert merged["claim"]["claim_id"] == "CLM-123"  # Preserved from Pass 1
        assert merged["claim"]["review"]["reviewer_name"] == "John"  # Preserved from Pass 1
        assert merged["claim"]["review"]["review_date"] == "2025-01-15"  # Updated from Pass 2

    def test_merge_preserves_non_target_fields(self):
        """Merge should not touch non-target fields"""
        pass1 = {
            "claim": {"claim_id": "CLM-123", "status": "Pending"},
            "claimant": {"name": "Alice"}
        }

        pass2 = {
            "claim": {"status": "Approved"}  # Different value
        }

        # Only merge claim.status
        merged = merge_extraction_results(pass1, pass2, ["claim.status"])

        assert merged["claim"]["claim_id"] == "CLM-123"  # Preserved
        assert merged["claim"]["status"] == "Approved"  # Updated
        assert merged["claimant"]["name"] == "Alice"  # Preserved

    def test_merge_ignores_empty_pass2_values(self):
        """Merge should not use empty values from Pass 2"""
        pass1 = {
            "claim": {"status": "Pending"}
        }

        pass2 = {
            "claim": {"status": None}  # Empty
        }

        merged = merge_extraction_results(pass1, pass2, ["claim.status"])

        # Should keep Pass 1 value since Pass 2 is empty
        assert merged["claim"]["status"] == "Pending"

    def test_merge_multiple_nested_paths(self):
        """Merge multiple nested paths"""
        pass1 = {
            "claim": {
                "review": {"review_date": None, "reviewer_name": "John"},
                "audit": {"audit_date": None, "audit_officer": "Jane"}
            }
        }

        pass2 = {
            "claim": {
                "review": {"review_date": "2025-01-15"},
                "audit": {"audit_date": "2025-01-20"}
            }
        }

        merged = merge_extraction_results(pass1, pass2, [
            "claim.review.review_date",
            "claim.audit.audit_date"
        ])

        assert merged["claim"]["review"]["review_date"] == "2025-01-15"
        assert merged["claim"]["review"]["reviewer_name"] == "John"  # Preserved
        assert merged["claim"]["audit"]["audit_date"] == "2025-01-20"
        assert merged["claim"]["audit"]["audit_officer"] == "Jane"  # Preserved

    def test_merge_deeply_nested(self):
        """Merge deeply nested paths (4 levels)"""
        pass1 = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": None
                    }
                }
            }
        }

        pass2 = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "found!"
                    }
                }
            }
        }

        merged = merge_extraction_results(pass1, pass2, ["level1.level2.level3.level4"])

        assert merged["level1"]["level2"]["level3"]["level4"] == "found!"


class TestFocusedPromptGeneration:
    """Test focused prompt generation for nested fields"""

    def test_create_focused_prompt_groups_by_section(self):
        """Prompt should group fields by top-level section"""
        missing = [
            "claim.review.review_date",
            "claim.audit.audit_officer",
            "claimant.personal_info.email"
        ]

        prompt = create_focused_prompt(None, missing, pass_number=2)

        assert "CLAIM SECTION:" in prompt
        assert "CLAIMANT SECTION:" in prompt
        assert "claim.review.review_date" in prompt
        assert "claim.audit.audit_officer" in prompt
        assert "claimant.personal_info.email" in prompt

    def test_create_focused_prompt_includes_instructions(self):
        """Prompt should include helpful instructions"""
        missing = ["claim.review.review_date"]

        prompt = create_focused_prompt(None, missing, pass_number=2)

        assert "FOCUSED RE-EXTRACTION" in prompt
        assert "Pass 2/2" in prompt
        assert "not found in Pass 1" in prompt  # Updated to match new wording
        assert "search the document carefully" in prompt  # Updated to match new wording

    def test_create_focused_prompt_includes_original(self):
        """Prompt should include original prompt if provided"""
        missing = ["claim.review.review_date"]
        original = "Extract insurance claim data"

        prompt = create_focused_prompt(original, missing, pass_number=2)

        assert "Original instructions: Extract insurance claim data" in prompt

    def test_create_focused_prompt_flat_fields(self):
        """Prompt should work with flat (non-nested) fields"""
        missing = ["field1", "field2"]

        prompt = create_focused_prompt(None, missing, pass_number=2)

        assert "FIELD1 SECTION:" in prompt
        assert "FIELD2 SECTION:" in prompt

