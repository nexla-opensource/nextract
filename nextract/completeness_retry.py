"""
Completeness-based retry for array schema extractions.

For array schemas, checks if extraction is complete and retries with
context from the first attempt if needed.
"""

from __future__ import annotations

from typing import Any
import structlog

from .schema import JsonSchema

log = structlog.get_logger(__name__)


def is_array_schema(schema: JsonSchema) -> bool:
    """Check if schema is an array schema"""
    return isinstance(schema, dict) and schema.get("type") == "array"


def create_completeness_schema(original_schema: JsonSchema) -> JsonSchema:
    """
    Create a schema that includes completeness metadata.
    
    For array schemas, wraps the array with metadata about completeness:
    {
        "items": [...],  # The actual array
        "extraction_metadata": {
            "items_found": 5,
            "extraction_complete": true/false,
            "completeness_confidence": 0.8,
            "reason": "Found all items in table" or "Table appears to continue"
        }
    }
    """
    import copy
    
    if not is_array_schema(original_schema):
        return original_schema
    
    # Create wrapped schema with metadata
    wrapped_schema = {
        "type": "object",
        "properties": {
            "items": copy.deepcopy(original_schema),
            "extraction_metadata": {
                "type": "object",
                "properties": {
                    "items_found": {
                        "type": "integer",
                        "description": "Number of items extracted from the document"
                    },
                    "extraction_complete": {
                        "type": "boolean",
                        "description": "Whether you believe the extraction is complete (true) or if there are more items you couldn't extract (false)"
                    },
                    "completeness_confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score (0.0 to 1.0) that you extracted ALL items. Use 1.0 if you're certain you got everything, 0.5 if you're unsure, 0.0 if you know you missed items."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why extraction is complete or incomplete. E.g., 'Found all 29 lenders in table' or 'Table continues beyond visible text' or 'Only found 2 of expected 29 lenders'"
                    }
                },
                "required": ["items_found", "extraction_complete", "completeness_confidence", "reason"]
            }
        },
        "required": ["items", "extraction_metadata"],
        "title": original_schema.get("title", "Array with Completeness Metadata"),
        "description": original_schema.get("description", "Array extraction with completeness tracking")
    }
    
    return wrapped_schema


def extract_items_and_metadata(result: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """
    Extract items and metadata from completeness-wrapped result.
    
    Returns: (items, metadata)
    """
    if not isinstance(result, dict):
        return [], {}
    
    items = result.get("items", [])
    metadata = result.get("extraction_metadata", {})
    
    if not isinstance(items, list):
        items = []
    
    return items, metadata


def should_retry(
    metadata: dict[str, Any],
    completeness_threshold: float = 0.7
) -> bool:
    """
    Determine if retry is needed based on completeness metadata.
    
    Args:
        metadata: Extraction metadata from first attempt
        completeness_threshold: Minimum confidence to accept (default: 0.7)
    
    Returns:
        True if retry is recommended
    """
    # Check if extraction is marked as incomplete
    if not metadata.get("extraction_complete", True):
        return True
    
    # Check if confidence is below threshold
    confidence = metadata.get("completeness_confidence", 1.0)
    if confidence < completeness_threshold:
        return True
    
    return False


def create_retry_prompt(
    original_prompt: str | None,
    first_attempt_items: list[Any],
    metadata: dict[str, Any]
) -> str:
    """
    Create an enhanced prompt for retry attempt.
    
    Includes context from first attempt to help LLM extract missing items.
    """
    items_found = len(first_attempt_items)
    reason = metadata.get("reason", "Incomplete extraction")
    
    retry_context = f"""
IMPORTANT: This is a RETRY extraction. The first attempt found {items_found} items but was incomplete.

First attempt result: {reason}

Please look MORE CAREFULLY and extract ALL items. Common issues:
- Tables that span multiple pages or sections
- Items in different formats or layouts
- Items that were partially visible or cut off
- Items in headers, footers, or margins

Extract EVERY SINGLE item you can find, even if formatting varies.
"""
    
    if original_prompt:
        return retry_context + "\n\n" + original_prompt
    else:
        return retry_context

