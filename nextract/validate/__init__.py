from .business_rules import BusinessRuleValidator
from .plan_validator import CapabilityDetector, PlanValidator
from .schema_validator import SchemaValidator

__all__ = [
    "BusinessRuleValidator",
    "CapabilityDetector",
    "PlanValidator",
    "SchemaValidator",
]


def __getattr__(name: str):
    """Lazy-load deprecated exports with warnings."""
    if name == "ConsistencyValidator":
        import warnings
        warnings.warn(
            "ConsistencyValidator is a stub that always returns valid and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .consistency_validator import ConsistencyValidator
        return ConsistencyValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
