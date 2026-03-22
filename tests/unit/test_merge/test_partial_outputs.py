"""
Unit tests for merge.py functions.

Tests:
- merge_partial_outputs
- _is_empty_value
- merge_with_conflict_resolution
- get_merge_report
"""


from nextract.merge import (
    _is_empty_value,
    get_merge_report,
    merge_partial_outputs,
    merge_with_conflict_resolution,
)


class TestIsEmptyValue:
    """Tests for _is_empty_value helper."""

    def test_none_is_empty(self):
        """None is considered empty."""
        assert _is_empty_value(None) is True

    def test_empty_string_is_empty(self):
        """Empty string is considered empty."""
        assert _is_empty_value("") is True

    def test_empty_list_is_empty(self):
        """Empty list is considered empty."""
        assert _is_empty_value([]) is True

    def test_empty_dict_is_empty(self):
        """Empty dict is considered empty."""
        assert _is_empty_value({}) is True

    def test_non_empty_string_not_empty(self):
        """Non-empty string is not empty."""
        assert _is_empty_value("hello") is False

    def test_non_empty_list_not_empty(self):
        """Non-empty list is not empty."""
        assert _is_empty_value([1, 2, 3]) is False

    def test_zero_is_not_empty(self):
        """Zero is not considered empty."""
        assert _is_empty_value(0) is False

    def test_false_is_not_empty(self):
        """False is not considered empty."""
        assert _is_empty_value(False) is False

    def test_nested_empty_dict_not_empty(self):
        """Dict with keys is not empty."""
        assert _is_empty_value({"key": None}) is False


class TestMergePartialOutputs:
    """Tests for merge_partial_outputs."""

    def test_single_output(self):
        """Single output should be returned unchanged."""
        output = {"name": "John", "age": 30}
        result = merge_partial_outputs([output])
        assert result == output

    def test_merge_non_overlapping(self):
        """Non-overlapping outputs should merge all keys."""
        outputs = [
            {"name": "John"},
            {"age": 30},
            {"city": "New York"},
        ]
        result = merge_partial_outputs(outputs)
        assert result == {"name": "John", "age": 30, "city": "New York"}

    def test_first_non_empty_wins(self):
        """First non-empty value should be kept."""
        outputs = [
            {"name": "John", "age": None},
            {"name": "Jane", "age": 30},
        ]
        result = merge_partial_outputs(outputs)
        assert result["name"] == "John"
        assert result["age"] == 30

    def test_empty_values_filled_later(self):
        """Empty values should be filled by later rounds."""
        outputs = [
            {"name": "John", "email": ""},
            {"email": "john@example.com"},
        ]
        result = merge_partial_outputs(outputs)
        assert result["name"] == "John"
        assert result["email"] == "john@example.com"

    def test_empty_list(self):
        """Empty list should return empty dict."""
        result = merge_partial_outputs([])
        assert result == {}

    def test_non_dict_outputs_skipped(self):
        """Non-dict outputs should be skipped."""
        outputs = [
            {"name": "John"},
            "not a dict",
            {"age": 30},
        ]
        result = merge_partial_outputs(outputs)
        assert result == {"name": "John", "age": 30}

    def test_preserves_complex_values(self):
        """Complex values (lists, dicts) should be preserved."""
        outputs = [
            {"items": [1, 2, 3], "details": {"a": 1}},
        ]
        result = merge_partial_outputs(outputs)
        assert result["items"] == [1, 2, 3]
        assert result["details"] == {"a": 1}


class TestMergeWithConflictResolution:
    """Tests for merge_with_conflict_resolution."""

    def test_first_strategy(self):
        """First strategy keeps first non-empty value."""
        outputs = [
            {"name": "John"},
            {"name": "Jane"},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="first")
        assert result["name"] == "John"

    def test_last_strategy(self):
        """Last strategy keeps last non-empty value."""
        outputs = [
            {"name": "John"},
            {"name": "Jane"},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="last")
        assert result["name"] == "Jane"

    def test_concat_strategy_strings(self):
        """Concat strategy concatenates strings."""
        outputs = [
            {"description": "First part"},
            {"description": "Second part"},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="concat")
        assert result["description"] == "First part Second part"

    def test_concat_strategy_lists(self):
        """Concat strategy concatenates lists."""
        outputs = [
            {"items": [1, 2]},
            {"items": [3, 4]},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="concat")
        assert result["items"] == [1, 2, 3, 4]

    def test_prefer_longer_strategy_string(self):
        """Prefer longer strategy keeps longer string."""
        outputs = [
            {"name": "John"},
            {"name": "Jonathan"},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="prefer_longer")
        assert result["name"] == "Jonathan"

    def test_prefer_longer_strategy_list(self):
        """Prefer longer strategy keeps longer list."""
        outputs = [
            {"items": [1, 2]},
            {"items": [1, 2, 3, 4]},
        ]
        result = merge_with_conflict_resolution(outputs, strategy="prefer_longer")
        assert result["items"] == [1, 2, 3, 4]


class TestGetMergeReport:
    """Tests for get_merge_report."""

    def test_basic_report(self):
        """Basic report generation."""
        outputs = [
            {"name": "John"},
            {"age": 30},
        ]
        merged = merge_partial_outputs(outputs)
        report = get_merge_report(outputs, merged)
        
        assert report["total_rounds"] == 2
        assert report["total_keys"] == 2
        assert len(report["keys_per_round"]) == 2

    def test_field_provenance_tracked(self):
        """Field provenance should track source round."""
        outputs = [
            {"name": "John", "city": ""},
            {"age": 30, "city": "NYC"},
        ]
        merged = merge_partial_outputs(outputs)
        report = get_merge_report(outputs, merged)
        
        assert report["field_provenance"]["name"] == 1
        assert report["field_provenance"]["age"] == 2
        assert report["field_provenance"]["city"] == 2

    def test_keys_per_round(self):
        """Keys per round should be tracked."""
        outputs = [
            {"a": 1, "b": 2},
            {"c": 3},
        ]
        merged = merge_partial_outputs(outputs)
        report = get_merge_report(outputs, merged)
        
        assert report["keys_per_round"][0]["count"] == 2
        assert report["keys_per_round"][1]["count"] == 1
