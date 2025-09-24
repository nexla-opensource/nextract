from __future__ import annotations

from typing import Any, Optional, Type, Union
import copy

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import StructuredDict

JsonSchema = dict[str, Any]
PydModelType = Type[BaseModel]

def is_pydantic_model(obj: Any) -> bool:
    try:
        return issubclass(obj, BaseModel)  # type: ignore[arg-type]
    except Exception:
        return False

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
    """
    if is_pydantic_model(schema_or_model):
        return schema_or_model  # a BaseModel subclass
    # else: JSON schema path
    schema = augment_schema_with_extra(schema_or_model, include_extra)
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
