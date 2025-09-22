from __future__ import annotations

from typing import Any, Optional, Type, Union

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
    return StructuredDict(schema, name=schema.get("title", "Output"))

def cast_to_pydantic(model_type: PydModelType, data: dict[str, Any]) -> BaseModel:
    # Fast conversion using the model itself
    return model_type.model_validate(data)

def cast_to_dict_from_pydantic(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump()
