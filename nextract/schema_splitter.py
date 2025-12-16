from __future__ import annotations

from typing import Any

JsonSchema = dict[str, Any]

def split_schema(schema: JsonSchema, group_size: int = 3) -> list[JsonSchema]:
    """Split a JSON Schema into multiple sub-schemas for multi-round extraction.
    
    Each sub-schema contains a subset of the original properties (up to group_size keys).
    This enables processing large schemas in smaller, more focused extraction rounds.
    
    Args:
        schema: Original JSON Schema with multiple properties
        group_size: Maximum number of properties per sub-schema (default: 3)
        
    Returns:
        List of sub-schemas, each containing a subset of properties
        
    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "number"},
        ...         "email": {"type": "string"},
        ...         "phone": {"type": "string"},
        ...         "address": {"type": "string"}
        ...     },
        ...     "required": ["name", "email"]
        ... }
        >>> subschemas = split_schema(schema, group_size=2)
        >>> len(subschemas)
        3
        >>> list(subschemas[0]["properties"].keys())
        ['name', 'age']
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")
    
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return [schema]  # No properties to split
    
    if len(properties) == 0:
        return [schema]  # Empty schema
    
    # Get all property keys
    keys = list(properties.keys())
    
    # Split keys into groups
    groups = [keys[i:i+group_size] for i in range(0, len(keys), group_size)]
    
    # Create sub-schemas
    subschemas: list[JsonSchema] = []
    required_fields = schema.get("required", [])
    
    for group_idx, group_keys in enumerate(groups):
        # Build sub-schema with only this group's properties
        sub_properties = {k: properties[k] for k in group_keys}
        
        # Only include required fields that are in this group
        sub_required = [r for r in required_fields if r in group_keys]
        
        sub_schema: JsonSchema = {
            "type": "object",
            "properties": sub_properties,
        }
        
        # Only add required if there are required fields in this group
        if sub_required:
            sub_schema["required"] = sub_required
        
        # Preserve title and description with round indicator
        if "title" in schema:
            sub_schema["title"] = f"{schema['title']}_Round{group_idx + 1}"
        else:
            sub_schema["title"] = f"ExtractionRound{group_idx + 1}"
        
        if "description" in schema:
            sub_schema["description"] = f"{schema['description']} (Round {group_idx + 1})"
        
        # Preserve $defs if present (for nested references)
        if "$defs" in schema and isinstance(schema["$defs"], dict):
            sub_schema["$defs"] = schema["$defs"]
        
        # Preserve $schema if present
        if "$schema" in schema:
            sub_schema["$schema"] = schema["$schema"]
        
        subschemas.append(sub_schema)
    
    return subschemas

def get_schema_metadata(schema: JsonSchema) -> dict[str, Any]:
    """Extract metadata about a schema for logging and debugging.
    
    Args:
        schema: JSON Schema to analyze
        
    Returns:
        Dictionary with metadata including property count, required fields, etc.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    return {
        "title": schema.get("title", "Untitled"),
        "property_count": len(properties) if isinstance(properties, dict) else 0,
        "required_count": len(required) if isinstance(required, list) else 0,
        "properties": list(properties.keys()) if isinstance(properties, dict) else [],
        "required_fields": required if isinstance(required, list) else [],
    }

