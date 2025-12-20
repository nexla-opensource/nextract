from __future__ import annotations

from typing import Any, Type, Union
import copy
import structlog

from pydantic import BaseModel
from pydantic_ai import StructuredDict

log = structlog.get_logger(__name__)

JsonSchema = dict[str, Any]
PydModelType = Type[BaseModel]

def is_pydantic_model(obj: Any) -> bool:
    try:
        return issubclass(obj, BaseModel)  # type: ignore[arg-type]
    except Exception:
        return False


def validate_json_schema(schema: JsonSchema) -> None:
    """
    Validate that a JSON Schema is properly formatted.

    Ensures:
    - Schema has "type": "object" at the root
    - Schema has "properties" dict
    - Nested objects have proper structure

    Raises:
        ValueError: If schema is not properly formatted

    Example of valid schema:
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"}
                    }
                }
            }
        }
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")

    # Check root level
    if "type" not in schema:
        log.warning(
            "json_schema_missing_type",
            recommendation="Add 'type': 'object' to root schema"
        )
    elif schema.get("type") != "object":
        raise ValueError(
            f"Root schema must have 'type': 'object', got '{schema.get('type')}'"
        )

    if "properties" not in schema:
        log.warning(
            "json_schema_missing_properties",
            recommendation="Add 'properties' dict to schema"
        )
        return

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Schema 'properties' must be a dictionary")

    # Recursively validate nested objects
    def validate_properties(props: dict, path: str = "") -> None:
        for field_name, field_schema in props.items():
            field_path = f"{path}.{field_name}" if path else field_name

            if not isinstance(field_schema, dict):
                log.warning(
                    "json_schema_invalid_field",
                    field=field_path,
                    issue="Field schema must be a dict",
                    recommendation=f"Use {{'type': 'string'}} instead of '{field_schema}'"
                )
                continue

            # Check if it's a nested object
            if field_schema.get("type") == "object":
                if "properties" not in field_schema:
                    log.warning(
                        "json_schema_nested_object_missing_properties",
                        field=field_path,
                        recommendation="Add 'properties' dict to nested object"
                    )
                else:
                    # Recurse into nested properties
                    validate_properties(field_schema["properties"], field_path)

    validate_properties(properties)

def augment_schema_with_extra(schema: JsonSchema, include_extra: bool) -> JsonSchema:
    """Optionally add an 'extra' object bag to carry any additional fields detected.

    We do *not* relax the original 'required' set for small-file mode.
    """
    if not include_extra:
        return schema

    new_schema: JsonSchema = {
        "type": "object",
        "properties": {},
        "required": schema.get("required", []),
        "additionalProperties": False,
        "title": schema.get("title", "Extraction"),
        "description": schema.get("description", ""),
    }
    # Merge properties
    base_props = schema.get("properties", {})
    if isinstance(base_props, dict):
        new_schema["properties"] = dict(base_props)
    # Preserve standard JSON Schema top-level fields used for resolution/semantics
    if "$schema" in schema:
        new_schema["$schema"] = schema["$schema"]
    if "$id" in schema:
        new_schema["$id"] = schema["$id"]
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        new_schema["$defs"] = dict(schema["$defs"])  # keep refs like #/$defs/foo working
    # Add extra bag
    new_schema["properties"]["extra"] = {
        "type": "object",
        "additionalProperties": True,
        "description": "Any additional fields relevant to the user that were detected but not in the schema.",
    }
    return new_schema

def build_output_type(
    schema_or_model: Union[JsonSchema, PydModelType],
    include_extra: bool,
) -> Any:
    """Return the output_type to pass into Agent.

    - If JSON Schema dict: return StructuredDict(schema) type
    - If Pydantic model type: return the model type as is

    Validates JSON schemas to ensure proper structure.
    """
    if is_pydantic_model(schema_or_model):
        return schema_or_model  # a BaseModel subclass

    # else: JSON schema path - validate it first
    validate_json_schema(schema_or_model)  # type: ignore[arg-type]

    schema = augment_schema_with_extra(schema_or_model, include_extra)  # type: ignore[arg-type]
    # Inline local $ref references to avoid issues with Pydantic's schema generator
    schema_inlined = _inline_local_refs(schema)
    return StructuredDict(schema_inlined, name=schema_inlined.get("title", "Output"))

def cast_to_pydantic(model_type: PydModelType, data: dict[str, Any]) -> BaseModel:
    # Fast conversion using the model itself
    return model_type.model_validate(data)

def cast_to_dict_from_pydantic(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump()


def _inline_local_refs(schema: JsonSchema) -> JsonSchema:
    """Return a deep-copied schema with local $refs (e.g. "#/$defs/foo") inlined.

    This helps interop with Pydantic's JSON schema generator used by pydantic-ai,
    which expects fully-resolved definitions for arbitrary $refs in literal schemas.
    """
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict):
        defs = {}

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            # If this object is a pure $ref, inline the referenced def
            ref = obj.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/") and len(obj) == 1:
                key = ref.split("/")[-1]
                target = defs.get(key)
                if isinstance(target, dict):
                    return resolve(copy.deepcopy(target))
            # Otherwise, recurse into children
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(copy.deepcopy(schema))

def to_json_schema(schema_or_model: Union[JsonSchema, PydModelType]) -> JsonSchema:
    """Return a JSON Schema dict for either a literal schema or a Pydantic model type."""
    if is_pydantic_model(schema_or_model):
        # BaseModel subclass → JSON Schema (inline local refs for downstream consumers)
        schema = schema_or_model.model_json_schema()  # type: ignore[union-attr]
        return _inline_local_refs(schema)
    return schema_or_model
