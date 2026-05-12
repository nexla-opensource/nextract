from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from nextract.core import BaseValidator, ValidationResult
from nextract.core.exceptions import ValidationError as NextractValidationError

_MAX_SCHEMA_SIZE_BYTES = 256 * 1024  # 256 KB
_MAX_SCHEMA_DEPTH = 20
_MAX_SCHEMA_REF_DEPTH = 10


def validate_schema_complexity(schema: dict[str, Any]) -> None:
    """Validate schema complexity limits before use.

    Raises ValidationError if the schema exceeds size, depth, or $ref limits.
    """
    raw = json.dumps(schema)
    if len(raw) > _MAX_SCHEMA_SIZE_BYTES:
        raise NextractValidationError(
            f"Schema exceeds maximum size of {_MAX_SCHEMA_SIZE_BYTES} bytes "
            f"(got {len(raw)} bytes)"
        )

    depth = _measure_schema_depth(schema)
    if depth > _MAX_SCHEMA_DEPTH:
        raise NextractValidationError(
            f"Schema nesting depth {depth} exceeds maximum of {_MAX_SCHEMA_DEPTH}"
        )

    ref_depth = _count_max_ref_depth(schema, set())
    if ref_depth > _MAX_SCHEMA_REF_DEPTH:
        raise NextractValidationError(
            f"Schema $ref chain depth {ref_depth} exceeds maximum of {_MAX_SCHEMA_REF_DEPTH}"
        )


def _measure_schema_depth(schema: Any, current: int = 0) -> int:
    if not isinstance(schema, dict):
        return current
    max_depth = current
    for key, value in schema.items():
        if isinstance(value, dict):
            max_depth = max(max_depth, _measure_schema_depth(value, current + 1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    max_depth = max(max_depth, _measure_schema_depth(item, current + 1))
    return max_depth


def _count_max_ref_depth(schema: dict[str, Any], seen: set[str]) -> int:
    """Count the maximum depth of $ref chains.

    Properly tracks visited refs to detect circular references and
    prevent double-counting.
    """
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return 0  # circular ref detected, stop
        # Count this ref and add it to seen set
        seen.add(ref)
        return 1
    max_depth = 0
    for value in schema.values():
        if isinstance(value, dict):
            child_depth = _count_max_ref_depth(value, seen.copy())
            max_depth = max(max_depth, child_depth)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    child_depth = _count_max_ref_depth(item, seen.copy())
                    max_depth = max(max_depth, child_depth)
    return max_depth


class SchemaValidator(BaseValidator):
    """Validate structured data against JSON Schema with completeness scoring."""

    def validate(self, data: Any, schema: dict[str, Any], **kwargs: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        validate_schema_complexity(schema)

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
