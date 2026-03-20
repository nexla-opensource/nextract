from __future__ import annotations

from typing import Any
import copy
import structlog

from pydantic import BaseModel

from .generator import SchemaGenerator, SchemaSuggestion

log = structlog.get_logger(__name__)

JsonSchema = dict[str, Any]
PydModelType = type[BaseModel]
NON_OBJECT_OUTPUT_KEY = "result"


def is_pydantic_model(obj: Any) -> bool:
    try:
        return issubclass(obj, BaseModel)  # type: ignore[arg-type]
    except Exception:
        return False


def validate_json_schema(schema: JsonSchema) -> None:
    """
    Validate that a JSON Schema is properly formatted.

    Ensures:
    - Root schema is an object, array, or composition/ref schema
    - Object schemas have valid "properties" dicts
    - Array schemas have valid "items" definitions when present
    - Nested objects have proper structure
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")

    schema_type = schema.get("type")
    if schema_type and schema_type not in ("object", "array"):
        raise ValueError(
            f"Root schema must have 'type': 'object' or 'array', got '{schema_type}'"
        )
    if not schema_type and not any(k in schema for k in ("anyOf", "oneOf", "allOf", "$ref")):
        log.warning(
            "schema_missing_type",
            recommendation="Add 'type': 'object' to root schema",
        )

    def validate_properties(props: dict, path: str = "") -> None:
        for field_name, field_schema in props.items():
            field_path = f"{path}.{field_name}" if path else field_name

            if not isinstance(field_schema, dict):
                log.warning(
                    "json_schema_invalid_field",
                    field=field_path,
                    issue="Field schema must be a dict",
                    recommendation=f"Use {{'type': 'string'}} instead of '{field_schema}'",
                )
                continue

            if field_schema.get("type") == "object":
                if "properties" not in field_schema:
                    log.warning(
                        "json_schema_nested_object_missing_properties",
                        field=field_path,
                        recommendation="Add 'properties' dict to nested object",
                    )
                else:
                    validate_properties(field_schema["properties"], field_path)
            elif field_schema.get("type") == "array":
                items = field_schema.get("items")
                if items is None:
                    log.warning(
                        "json_schema_array_missing_items",
                        field=field_path,
                        recommendation="Add 'items' schema to array field",
                    )
                elif not isinstance(items, dict):
                    raise ValueError(f"Schema '{field_path}.items' must be a dictionary")
                elif items.get("type") == "object":
                    item_props = items.get("properties")
                    if isinstance(item_props, dict):
                        validate_properties(item_props, f"{field_path}[]")

    if schema_type == "array":
        items = schema.get("items")
        if items is None:
            log.warning(
                "json_schema_array_missing_items",
                recommendation="Add 'items' schema to array schema",
            )
            return
        if not isinstance(items, dict):
            raise ValueError("Schema 'items' must be a dictionary")
        if items.get("type") == "object":
            item_props = items.get("properties")
            if isinstance(item_props, dict):
                validate_properties(item_props, "[]")
        return

    if "properties" not in schema:
        if schema_type == "object":
            log.warning(
                "json_schema_missing_properties",
                recommendation="Add 'properties' dict to schema",
            )
        return

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Schema 'properties' must be a dictionary")

    validate_properties(properties)


def augment_schema_with_extra(schema: JsonSchema, include_extra: bool) -> JsonSchema:
    """Optionally add an 'extra' object bag to carry any additional fields detected."""
    if not include_extra:
        return schema

    base_props = schema.get("properties", {})
    if schema.get("type") not in (None, "object") or not isinstance(base_props, dict):
        log.warning(
            "schema_extra_unsupported_root",
            schema_type=schema.get("type"),
            msg="Skipping include_extra for non-object root schema",
        )
        return schema

    new_schema: JsonSchema = {
        "type": "object",
        "properties": {},
        "required": list(schema.get("required", [])),
        "additionalProperties": False,
        "title": schema.get("title", "Extraction"),
        "description": schema.get("description", ""),
    }
    new_schema["properties"] = dict(base_props)
    if "$schema" in schema:
        new_schema["$schema"] = schema["$schema"]
    if "$id" in schema:
        new_schema["$id"] = schema["$id"]
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        new_schema["$defs"] = dict(schema["$defs"])

    if "extra" not in new_schema["properties"]:
        new_schema["properties"]["extra"] = {
            "type": "object",
            "additionalProperties": True,
            "description": "Any additional fields relevant to the user that were detected but not in the schema.",
        }
    else:
        log.warning("schema_extra_field_collision", msg="Schema already has 'extra' field; skipping augmentation")
    return new_schema


def build_output_type(
    schema_or_model: JsonSchema | PydModelType,
    include_extra: bool,
) -> Any:
    """Return the output_type to pass into Agent."""
    if is_pydantic_model(schema_or_model):
        return schema_or_model

    schema, _unwrap_key = prepare_output_schema(schema_or_model, include_extra)

    from pydantic_ai import StructuredDict

    return StructuredDict(schema, name=schema.get("title", "Output"))


def prepare_output_schema(
    schema: JsonSchema,
    include_extra: bool,
) -> tuple[JsonSchema, str | None]:
    """Prepare a JSON schema for providers that require an object root."""
    validate_json_schema(schema)

    schema_inlined = _inline_local_refs(schema)
    validate_json_schema(schema_inlined)
    schema_with_extra = augment_schema_with_extra(schema_inlined, include_extra)

    if _needs_object_wrapper(schema_with_extra):
        title = schema_with_extra.get("title", "Output")
        wrapped_schema: JsonSchema = {
            "type": "object",
            "title": title,
            "description": schema_with_extra.get("description", ""),
            "properties": {
                NON_OBJECT_OUTPUT_KEY: schema_with_extra,
            },
            "required": [NON_OBJECT_OUTPUT_KEY],
            "additionalProperties": False,
        }
        return wrapped_schema, NON_OBJECT_OUTPUT_KEY

    return schema_with_extra, None


def cast_to_pydantic(model_type: PydModelType, data: dict[str, Any]) -> BaseModel:
    return model_type.model_validate(data)


def cast_to_dict_from_pydantic(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump()


def _inline_local_refs(schema: JsonSchema) -> JsonSchema:
    """Return a deep-copied schema with local $refs inlined."""
    defs = schema.get("$defs", schema.get("definitions", {}))
    if not isinstance(defs, dict):
        defs = {}

    def resolve(obj: Any, seen: frozenset[str] = frozenset()) -> Any:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if isinstance(ref, str) and (ref.startswith("#/$defs/") or ref.startswith("#/definitions/")):
                key = ref.split("/")[-1]
                if key in seen:
                    return obj  # Leave circular $ref intact
                target = defs.get(key)
                if isinstance(target, dict):
                    return resolve(copy.deepcopy(target), seen | {key})
            return {k: resolve(v, seen) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(item, seen) for item in obj]
        return obj

    resolved = resolve(copy.deepcopy(schema))
    resolved.pop("$defs", None)
    resolved.pop("definitions", None)
    return resolved


def _needs_object_wrapper(schema: JsonSchema) -> bool:
    return schema.get("type") != "object" and "$ref" not in schema


def to_json_schema(schema_or_model: JsonSchema | PydModelType) -> JsonSchema:
    """Return a JSON Schema dict for either a literal schema or a Pydantic model type."""
    if is_pydantic_model(schema_or_model):
        schema = schema_or_model.model_json_schema()  # type: ignore[union-attr]
        return _inline_local_refs(schema)
    return schema_or_model


__all__ = [
    "JsonSchema",
    "PydModelType",
    "is_pydantic_model",
    "validate_json_schema",
    "augment_schema_with_extra",
    "build_output_type",
    "prepare_output_schema",
    "cast_to_pydantic",
    "cast_to_dict_from_pydantic",
    "to_json_schema",
    "SchemaGenerator",
    "SchemaSuggestion",
]
