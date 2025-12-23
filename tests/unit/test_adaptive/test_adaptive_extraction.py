"""
Tests for adaptive extraction (two-pass strategy).

Tests cover:
- Empty field identification
- Focused schema creation
- Focused prompt generation
- Result merging
- Two-pass extraction flow
- Edge cases
"""

from nextract.adaptive_extraction import (
    identify_missing_fields,
    create_focused_schema,
    create_focused_prompt,
    merge_extraction_results
)


class TestIdentifyMissingFields:
    """Test missing field identification"""
    
    def test_all_fields_populated(self):
        result = {
            "name": "John Doe",
            "age": "30",
            "city": "New York"
        }
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "city": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(result, schema)
        assert missing == []
    
    def test_some_fields_missing(self):
        result = {
            "name": "John Doe",
            "age": None,
            "city": ""
        }
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "city": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(result, schema)
        assert set(missing) == {"age", "city"}
    
    def test_placeholder_values_detected(self):
        result = {
            "name": "John Doe",
            "age": "Pending",
            "city": "N/A",
            "email": "Unknown"
        }
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "city": {"type": "string"},
                "email": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(result, schema)
        assert set(missing) == {"age", "city", "email"}
    
    def test_threshold_not_met(self):
        # Only 1 out of 10 fields empty (10%) < 30% threshold
        result = {f"field{i}": "value" for i in range(10)}
        result["field0"] = None
        
        schema = {
            "type": "object",
            "properties": {f"field{i}": {"type": "string"} for i in range(10)}
        }
        
        missing = identify_missing_fields(result, schema, empty_threshold=0.3)
        # Should return empty list because below threshold
        assert missing == []
    
    def test_threshold_met(self):
        # 4 out of 10 fields empty (40%) > 30% threshold
        result = {f"field{i}": "value" for i in range(10)}
        result["field0"] = None
        result["field1"] = ""
        result["field2"] = "N/A"
        result["field3"] = "Pending"
        
        schema = {
            "type": "object",
            "properties": {f"field{i}": {"type": "string"} for i in range(10)}
        }
        
        missing = identify_missing_fields(result, schema, empty_threshold=0.3)
        assert len(missing) == 4
        assert set(missing) == {"field0", "field1", "field2", "field3"}
    
    def test_field_not_in_result(self):
        result = {
            "name": "John Doe"
        }
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "city": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(result, schema)
        assert set(missing) == {"age", "city"}


class TestCreateFocusedSchema:
    """Test focused schema creation"""
    
    def test_subset_schema(self):
        original = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
                "city": {"type": "string", "description": "City of residence"},
                "email": {"type": "string", "description": "Email address"}
            },
            "required": ["name", "age"]
        }
        
        target_fields = ["age", "email"]
        focused = create_focused_schema(original, target_fields)
        
        assert focused["type"] == "object"
        assert set(focused["properties"].keys()) == {"age", "email"}
        assert focused["properties"]["age"] == original["properties"]["age"]
        assert focused["properties"]["email"] == original["properties"]["email"]
    
    def test_preserves_descriptions(self):
        original = {
            "type": "object",
            "properties": {
                "field1": {"type": "string", "description": "Description 1"},
                "field2": {"type": "string", "description": "Description 2"}
            }
        }
        
        focused = create_focused_schema(original, ["field1"])
        assert focused["properties"]["field1"]["description"] == "Description 1"
    
    def test_empty_target_fields(self):
        original = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            }
        }
        
        focused = create_focused_schema(original, [])
        assert focused["properties"] == {}
    
    def test_nonexistent_field(self):
        original = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            }
        }
        
        # Should skip nonexistent fields
        focused = create_focused_schema(original, ["field1", "nonexistent"])
        assert set(focused["properties"].keys()) == {"field1"}


class TestCreateFocusedPrompt:
    """Test focused prompt generation"""
    
    def test_basic_focused_prompt(self):
        target_fields = ["age", "email", "city"]
        prompt = create_focused_prompt(None, target_fields)

        assert "Pass 2" in prompt  # Updated to match new format
        assert "FOCUSED" in prompt and "EXTRACTION" in prompt
        assert "3 fields" in prompt
        assert "age" in prompt
        assert "email" in prompt
        assert "city" in prompt
    
    def test_with_original_prompt(self):
        original = "Extract insurance claim data."
        target_fields = ["claim_id", "amount"]
        prompt = create_focused_prompt(original, target_fields)
        
        assert original in prompt
        assert "claim_id" in prompt
        assert "amount" in prompt
    
    def test_pass_number_in_prompt(self):
        prompt = create_focused_prompt(None, ["field1"], pass_number=3)
        assert "Pass 3" in prompt  # Updated to match new format
    
    def test_single_field(self):
        prompt = create_focused_prompt(None, ["single_field"])
        assert "1 field" in prompt or "single_field" in prompt
    
    def test_many_fields(self):
        fields = [f"field{i}" for i in range(20)]
        prompt = create_focused_prompt(None, fields)
        assert "20 fields" in prompt


class TestMergeExtractionResults:
    """Test result merging from two passes"""
    
    def test_pass2_overrides_empty_fields(self):
        pass1 = {
            "name": "John Doe",
            "age": None,
            "city": ""
        }
        pass2 = {
            "age": "30",
            "city": "New York"
        }
        target_fields = ["age", "city"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        assert merged["name"] == "John Doe"
        assert merged["age"] == "30"
        assert merged["city"] == "New York"
    
    def test_pass1_preserved_for_non_target_fields(self):
        pass1 = {
            "name": "John Doe",
            "age": "25",
            "city": "Boston"
        }
        pass2 = {
            "age": "30",
            "name": "Jane Smith"  # Should be ignored
        }
        target_fields = ["age"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Pass 1 values preserved for non-target fields
        assert merged["name"] == "John Doe"
        assert merged["city"] == "Boston"
        
        # Pass 2 value used for target field
        assert merged["age"] == "30"
    
    def test_pass2_empty_values_ignored(self):
        pass1 = {
            "name": "John Doe",
            "age": "25"
        }
        pass2 = {
            "age": None  # Empty in pass 2
        }
        target_fields = ["age"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Should keep pass 1 value if pass 2 is empty
        assert merged["age"] == "25"
    
    def test_pass2_placeholder_values_ignored(self):
        pass1 = {
            "age": "25"
        }
        pass2 = {
            "age": "N/A"  # Placeholder in pass 2
        }
        target_fields = ["age"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Should keep pass 1 value if pass 2 is placeholder
        assert merged["age"] == "25"
    
    def test_new_fields_from_pass2(self):
        pass1 = {
            "name": "John Doe"
        }
        pass2 = {
            "age": "30",
            "city": "New York"
        }
        target_fields = ["age", "city"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        assert merged["name"] == "John Doe"
        assert merged["age"] == "30"
        assert merged["city"] == "New York"
    
    def test_nested_objects(self):
        pass1 = {
            "person": {
                "name": "John",
                "age": None
            }
        }
        pass2 = {
            "person": {
                "age": "30"
            }
        }
        target_fields = ["person"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Should merge nested objects
        assert merged["person"]["age"] == "30"
    
    def test_list_values(self):
        pass1 = {
            "tags": []
        }
        pass2 = {
            "tags": ["tag1", "tag2"]
        }
        target_fields = ["tags"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        assert merged["tags"] == ["tag1", "tag2"]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_result(self):
        result = {}
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(result, schema)
        assert "field1" in missing
    
    def test_empty_schema(self):
        result = {"field1": "value"}
        schema = {
            "type": "object",
            "properties": {}
        }
        
        missing = identify_missing_fields(result, schema)
        assert missing == []
    
    def test_schema_without_properties(self):
        result = {"field1": "value"}
        schema = {"type": "object"}
        
        # Should handle gracefully
        missing = identify_missing_fields(result, schema)
        assert missing == []
    
    def test_merge_with_empty_pass2(self):
        pass1 = {"name": "John", "age": "30"}
        pass2 = {}
        target_fields = ["age"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Should preserve pass1 values
        assert merged == pass1
    
    def test_merge_with_empty_pass1(self):
        pass1 = {}
        pass2 = {"name": "John", "age": "30"}
        target_fields = ["name", "age"]
        
        merged = merge_extraction_results(pass1, pass2, target_fields)
        
        # Should use pass2 values
        assert merged["name"] == "John"
        assert merged["age"] == "30"


class TestIntegration:
    """Integration tests for adaptive extraction workflow"""
    
    def test_typical_workflow(self):
        # Simulate Pass 1 result with some missing fields
        pass1_result = {
            "claim_id": "CLM-001",
            "claimant_name": "John Doe",
            "claimant_email": None,
            "incident_date": "2024-01-15",
            "incident_description": "",
            "approved_amount": "Pending"
        }
        
        schema = {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "claimant_name": {"type": "string"},
                "claimant_email": {"type": "string"},
                "incident_date": {"type": "string"},
                "incident_description": {"type": "string"},
                "approved_amount": {"type": "string"}
            }
        }
        
        # Step 1: Identify missing fields
        missing = identify_missing_fields(pass1_result, schema, empty_threshold=0.3)
        
        # Should identify 3 missing fields (50% > 30% threshold)
        assert len(missing) == 3
        assert set(missing) == {"claimant_email", "incident_description", "approved_amount"}
        
        # Step 2: Create focused schema
        focused_schema = create_focused_schema(schema, missing)
        assert len(focused_schema["properties"]) == 3
        
        # Step 3: Create focused prompt
        prompt = create_focused_prompt(
            "Extract insurance claim data.",
            missing
        )
        assert "3 fields" in prompt
        
        # Step 4: Simulate Pass 2 result
        pass2_result = {
            "claimant_email": "john@example.com",
            "incident_description": "Car accident on Main St",
            "approved_amount": "$5000"
        }
        
        # Step 5: Merge results
        final = merge_extraction_results(pass1_result, pass2_result, missing)
        
        # Verify final result
        assert final["claim_id"] == "CLM-001"
        assert final["claimant_name"] == "John Doe"
        assert final["claimant_email"] == "john@example.com"
        assert final["incident_date"] == "2024-01-15"
        assert final["incident_description"] == "Car accident on Main St"
        assert final["approved_amount"] == "$5000"
    
    def test_no_missing_fields_workflow(self):
        # All fields populated in Pass 1
        pass1_result = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"}
            }
        }
        
        missing = identify_missing_fields(pass1_result, schema)
        
        # No missing fields, so no Pass 2 needed
        assert missing == []

