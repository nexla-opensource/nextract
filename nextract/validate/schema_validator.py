from __future__ import annotations

from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError as JsonSchemaError

from nextract.core import BaseValidator, ValidationResult


class SchemaValidator(BaseValidator):
    """Validate structured data against JSON Schema with completeness scoring."""

    def validate(self, data: Any, schema: Dict[str, Any], **kwargs: Any) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Any] = {}

        try:
            Draft202012Validator(schema).validate(data)
        except JsonSchemaError as exc:
            errors.append(exc.message)
            metadata["path"] = list(exc.path)
            metadata["schema_path"] = list(exc.schema_path)

        metadata["completeness"] = self._calculate_completeness(data, schema)

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings, metadata=metadata)

    def _calculate_completeness(self, data: Any, schema: Dict[str, Any]) -> float:
        required = schema.get("required", [])
        if not required or not isinstance(data, dict):
            return 1.0
        filled = 0
        for field in required:
            value = data.get(field)
            if value not in (None, "", [], {}):
                filled += 1
        return filled / max(len(required), 1)
