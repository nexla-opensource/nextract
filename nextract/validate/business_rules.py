from __future__ import annotations

from typing import Any, Callable

from nextract.core import BaseValidator, ValidationResult

RuleFn = Callable[[Any], bool | tuple[bool, str]]


class BusinessRuleValidator(BaseValidator):
    """Apply custom business rules to extracted data."""

    def __init__(self, rules: list[RuleFn] | None = None) -> None:
        self.rules = rules or []

    def validate(self, data: Any, schema: dict, **kwargs: Any) -> ValidationResult:
        errors: list[str] = []
        for rule in self.rules:
            result = rule(data)
            if isinstance(result, tuple):
                passed, message = result
            else:
                passed, message = result, "Business rule failed"
            if not passed:
                errors.append(message)

        return ValidationResult(valid=not errors, errors=errors)
