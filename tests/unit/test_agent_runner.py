"""Tests for nextract.agent_runner — validation, schema pruning, pricing."""

from __future__ import annotations

import json

import pytest

from nextract.agent_runner import (
    _prune_optional_empty_values,
    _collect_required_empty_errors,
    ExtractionReport,
    ExtractionMetrics,
)


class TestPruneOptionalEmptyValues:
    def test_removes_empty_optional_string(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        data = {"name": "Alice", "nickname": ""}
        result = _prune_optional_empty_values(data, schema)
        assert result == {"name": "Alice"}

    def test_keeps_required_empty_string(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"name": ""}
        result = _prune_optional_empty_values(data, schema)
        assert result == {"name": ""}

    def test_removes_null_optional(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["id"],
        }
        data = {"id": 1, "label": None}
        result = _prune_optional_empty_values(data, schema)
        assert result == {"id": 1}

    def test_handles_arrays(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"val": {"type": "string"}},
                "required": [],
            },
        }
        data = [{"val": ""}, {"val": "x"}]
        result = _prune_optional_empty_values(data, schema)
        assert result == [{}, {"val": "x"}]

    def test_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                    },
                    "required": [],
                },
            },
            "required": ["outer"],
        }
        data = {"outer": {"inner": ""}}
        result = _prune_optional_empty_values(data, schema)
        assert result == {"outer": {}}


class TestCollectRequiredEmptyErrors:
    def test_finds_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}
        errors = _collect_required_empty_errors(data, schema)
        assert "name" in errors

    def test_finds_empty_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"name": ""}
        errors = _collect_required_empty_errors(data, schema)
        assert "name" in errors

    def test_no_errors_for_filled_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"name": "Alice"}
        errors = _collect_required_empty_errors(data, schema)
        assert errors == []

    def test_nested_paths(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            "required": ["address"],
        }
        data = {"address": {}}
        errors = _collect_required_empty_errors(data, schema)
        assert "address.city" in errors


class TestExtractionReport:
    def test_report_creation(self):
        report = ExtractionReport(
            model="gpt-4o",
            files=["test.pdf"],
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert report.model == "gpt-4o"
        assert report.files == ["test.pdf"]
        assert report.cost_estimate_usd is None
        assert report.warnings == []


class TestPricing:
    def test_parse_pricing_json_empty(self):
        from nextract.pricing import parse_pricing_json
        assert parse_pricing_json("") == {}

    def test_parse_pricing_json_invalid(self):
        from nextract.pricing import parse_pricing_json
        assert parse_pricing_json("not json") == {}

    def test_parse_pricing_json_valid(self):
        from nextract.pricing import parse_pricing_json
        raw = json.dumps({"gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015}})
        result = parse_pricing_json(raw)
        assert "gpt-4o" in result
        assert result["gpt-4o"].input_per_1k == 0.005

    def test_estimate_cost_sync(self):
        from nextract.pricing import parse_pricing_json, estimate_cost_usd_sync
        raw = json.dumps({"gpt-4o": {"input_per_1k": 5.0, "output_per_1k": 15.0}})
        pricing_map = parse_pricing_json(raw)
        cost = estimate_cost_usd_sync(1000, 1000, "gpt-4o", pricing_map)
        assert cost is not None
        assert abs(cost - 20.0) < 0.01

    def test_estimate_cost_unknown_model(self):
        from nextract.pricing import parse_pricing_json, estimate_cost_usd_sync
        raw = json.dumps({"gpt-4o": {"input_per_1k": 5.0, "output_per_1k": 15.0}})
        pricing_map = parse_pricing_json(raw)
        cost = estimate_cost_usd_sync(1000, 1000, "unknown", pricing_map)
        assert cost is None
