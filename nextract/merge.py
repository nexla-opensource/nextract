from __future__ import annotations

from typing import Any
from jsonschema import Draft202012Validator, ValidationError

import structlog

log = structlog.get_logger(__name__)

JsonSchema = dict[str, Any]

def merge_partial_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple partial extraction outputs into a single unified result.
    
    Merging strategy:
    - For each key, keep the first non-empty value encountered
    - Empty values are: None, empty string "", empty list [], empty dict {}
    - Later rounds can fill in missing values but won't overwrite existing ones
    
    Args:
        outputs: List of partial extraction results from multiple rounds
        
    Returns:
        Merged dictionary containing all non-empty values
        
    Example:
        >>> round1 = {"name": "John", "age": None}
        >>> round2 = {"age": 30, "email": "john@example.com"}
        >>> round3 = {"phone": "555-1234"}
        >>> merge_partial_outputs([round1, round2, round3])
        {'name': 'John', 'age': 30, 'email': 'john@example.com', 'phone': '555-1234'}
    """
    merged: dict[str, Any] = {}
    
    for round_idx, output in enumerate(outputs, start=1):
        if not isinstance(output, dict):
            log.warning("merge_skip_non_dict", round=round_idx, type=type(output).__name__)
            continue
        
        for key, value in output.items():
            # Check if key already exists with a non-empty value
            if key in merged and not _is_empty_value(merged[key]):
                # Already have a value for this key, skip
                log.debug("merge_skip_existing", key=key, round=round_idx)
                continue
            
            # Set the value (even if empty, to track that we've seen this key)
            merged[key] = value
            
            if not _is_empty_value(value):
                log.debug("merge_set_value", key=key, round=round_idx, value_type=type(value).__name__)
    
    return merged

def _is_empty_value(value: Any) -> bool:
    """Check if a value is considered empty for merging purposes.
    
    Empty values:
    - None
    - Empty string ""
    - Empty list []
    - Empty dict {}
    
    Non-empty values:
    - Any other value, including 0, False, etc.
    """
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False

def validate_against_full_schema(output: dict[str, Any], schema: JsonSchema) -> dict[str, Any]:
    """Validate merged output against the original full schema.
    
    This ensures that the merged result from multiple rounds satisfies
    the complete schema requirements (required fields, types, etc.).
    
    Args:
        output: Merged extraction result
        schema: Original full JSON Schema
        
    Returns:
        The output dict if validation succeeds
        
    Raises:
        ValidationError: If output doesn't match schema
    """
    try:
        Draft202012Validator(schema).validate(output)
        log.info("merge_validation_success", 
                 property_count=len(output),
                 required_count=len(schema.get("required", [])))
        return output
    except ValidationError as e:
        log.error("merge_validation_failed", 
                  error=e.message,
                  path=list(e.path) if e.path else [],
                  schema_path=list(e.schema_path) if e.schema_path else [])
        raise

def merge_with_conflict_resolution(
    outputs: list[dict[str, Any]], 
    strategy: str = "first"
) -> dict[str, Any]:
    """Advanced merge with configurable conflict resolution strategies.
    
    Strategies:
    - "first": Keep first non-empty value (default)
    - "last": Keep last non-empty value
    - "concat": Concatenate string values, merge lists
    - "prefer_longer": For strings/lists, keep the longer value
    
    Args:
        outputs: List of partial extraction results
        strategy: Conflict resolution strategy
        
    Returns:
        Merged dictionary
    """
    if strategy == "first":
        return merge_partial_outputs(outputs)
    
    merged: dict[str, Any] = {}
    
    for round_idx, output in enumerate(outputs, start=1):
        if not isinstance(output, dict):
            continue
        
        for key, value in output.items():
            if _is_empty_value(value):
                continue
            
            if key not in merged:
                merged[key] = value
                continue
            
            # Handle conflicts based on strategy
            existing = merged[key]
            
            if strategy == "last":
                merged[key] = value
                log.debug("merge_conflict_last", key=key, round=round_idx)
            
            elif strategy == "concat":
                if isinstance(existing, str) and isinstance(value, str):
                    merged[key] = existing + " " + value
                    log.debug("merge_conflict_concat_str", key=key, round=round_idx)
                elif isinstance(existing, list) and isinstance(value, list):
                    merged[key] = existing + value
                    log.debug("merge_conflict_concat_list", key=key, round=round_idx)
                else:
                    merged[key] = value  # Fallback to last
            
            elif strategy == "prefer_longer":
                if isinstance(existing, (str, list)) and isinstance(value, (str, list)):
                    if len(value) > len(existing):
                        merged[key] = value
                        log.debug("merge_conflict_prefer_longer", key=key, round=round_idx)
                else:
                    merged[key] = value  # Fallback to last
    
    return merged

def get_merge_report(outputs: list[dict[str, Any]], merged: dict[str, Any]) -> dict[str, Any]:
    """Generate a report about the merge process.
    
    Useful for debugging and understanding which round contributed which fields.
    
    Args:
        outputs: Original partial outputs
        merged: Final merged result
        
    Returns:
        Report dictionary with merge statistics
    """
    report = {
        "total_rounds": len(outputs),
        "total_keys": len(merged),
        "keys_per_round": [],
        "field_provenance": {},
    }
    
    # Track which round each field came from
    for key in merged:
        for round_idx, output in enumerate(outputs, start=1):
            if isinstance(output, dict) and key in output and not _is_empty_value(output[key]):
                report["field_provenance"][key] = round_idx
                break
    
    # Count keys per round
    for round_idx, output in enumerate(outputs, start=1):
        if isinstance(output, dict):
            non_empty_keys = [k for k, v in output.items() if not _is_empty_value(v)]
            report["keys_per_round"].append({
                "round": round_idx,
                "keys": non_empty_keys,
                "count": len(non_empty_keys)
            })
    
    return report

