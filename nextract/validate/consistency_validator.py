from __future__ import annotations

from typing import Any

from nextract.core import BaseValidator, ValidationResult


class ConsistencyValidator(BaseValidator):
    """Placeholder consistency checks for extracted data."""

    def validate(self, data: Any, schema: dict, **kwargs: Any) -> ValidationResult:
        warnings: list[str] = []
        return ValidationResult(valid=True, warnings=warnings)
