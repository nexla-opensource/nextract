from __future__ import annotations

from typing import Any, Callable, List, Tuple

from nextract.core import BaseValidator, ValidationResult

RuleFn = Callable[[Any], bool | Tuple[bool, str]]


class BusinessRuleValidator(BaseValidator):
    """Apply custom business rules to extracted data."""

    def __init__(self, rules: List[RuleFn] | None = None) -> None:
        self.rules = rules or []

    def validate(self, data: Any, schema: dict, **kwargs: Any) -> ValidationResult:
        errors: List[str] = []
        for rule in self.rules:
            result = rule(data)
            if isinstance(result, tuple):
                passed, message = result
            else:
                passed, message = result, "Business rule failed"
            if not passed:
                errors.append(message)

        return ValidationResult(valid=not errors, errors=errors)
