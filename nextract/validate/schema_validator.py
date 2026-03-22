from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from nextract.core import BaseValidator, ValidationResult


class SchemaValidator(BaseValidator):
    """Validate structured data against JSON Schema with completeness scoring."""

    def validate(self, data: Any, schema: dict[str, Any], **kwargs: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        validator_instance = Draft202012Validator(schema)
        all_errors = sorted(validator_instance.iter_errors(data), key=lambda e: list(e.absolute_path))
        for exc in all_errors:
            path_str = " -> ".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "(root)"
            errors.append(f"[{path_str}] {exc.message}")
        if all_errors:
            metadata["path"] = list(all_errors[0].absolute_path)
            metadata["schema_path"] = list(all_errors[0].absolute_schema_path)

        metadata["completeness"] = self._calculate_completeness(data, schema)

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings, metadata=metadata)

    def _calculate_completeness(self, data: Any, schema: dict[str, Any]) -> float:
        if schema.get("type") == "array" and isinstance(data, list):
            if not data:
                return 0.0
            item_schema = schema.get("items", {})
            return sum(self._calculate_completeness(item, item_schema) for item in data) / len(data)

        required = schema.get("required", [])
        if not required or not isinstance(data, dict):
            return 1.0
        filled = 0
        for field in required:
            value = data.get(field)
            if value not in (None, "", [], {}):
                filled += 1
        return filled / max(len(required), 1)
