"""
Two-Pass Adaptive Extraction for improving field completeness.

Strategy:
1. Pass 1: Extract all fields at once (fast, cheap)
2. Analyze: Identify empty/missing fields
3. Pass 2: Re-extract only missing fields with focused prompt (targeted)
4. Merge: Combine results with Pass 2 overriding Pass 1

This is much cheaper than full field chunking while still improving quality.

Cost comparison (50-field schema):
- Single pass: $0.03
- Two-pass adaptive: $0.04 (1.3× cost)
- Full field chunking: $0.13-0.28 (4-9× cost)
"""

from __future__ import annotations

from typing import Any

import structlog

from .config import RuntimeConfig
from .schema import JsonSchema

log = structlog.get_logger(__name__)


# ============================================================================
# Helper Functions for Nested Schema Operations
# ============================================================================

def count_leaf_fields(schema: JsonSchema, prefix: str = "") -> int:
    """
    Recursively count total leaf fields in a nested schema.

    Args:
        schema: JSON Schema object
        prefix: Current path prefix (for recursion)

    Returns:
        Total number of leaf fields

    Example:
        Schema with structure:
        - claim (object)
          - claim_id (string) ← leaf
          - review (object)
            - reviewer_name (string) ← leaf
            - approval (object)
              - approved_amount (string) ← leaf

        Returns: 3 leaf fields
    """
    if schema.get("type") != "object":
        return 1  # This is a leaf field

    properties = schema.get("properties", {})
    if not properties:
        return 1  # Empty object counts as 1 field

    total = 0
    for field_name, field_schema in properties.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name

        if field_schema.get("type") == "object":
            # Recurse into nested object
            total += count_leaf_fields(field_schema, field_path)
        else:
            # Leaf field
            total += 1

    return total


def get_nested_value(data: dict[str, Any], dot_path: str) -> Any:
    """
    Get value from nested dict using dot notation.

    Args:
        data: Nested dictionary
        dot_path: Dot-notation path (e.g., "claim.review.reviewer_name")

    Returns:
        Value at the path, or None if not found

    Example:
        data = {"claim": {"review": {"reviewer_name": "John"}}}
        get_nested_value(data, "claim.review.reviewer_name") → "John"
    """
    keys = dot_path.split(".")
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None

    return current


def set_nested_value(data: dict[str, Any], dot_path: str, value: Any) -> None:
    """
    Set value in nested dict using dot notation.

    Args:
        data: Nested dictionary (modified in place)
        dot_path: Dot-notation path (e.g., "claim.review.reviewer_name")
        value: Value to set

    Example:
        data = {"claim": {"review": {}}}
        set_nested_value(data, "claim.review.reviewer_name", "John")
        → data = {"claim": {"review": {"reviewer_name": "John"}}}
    """
    keys = dot_path.split(".")
    current = data

    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            # Overwrite non-dict values with dict
            current[key] = {}
        current = current[key]

    # Set value
    current[keys[-1]] = value


def is_empty_value(value: Any) -> bool:
    """
    Check if a value should be considered empty.

    Args:
        value: Value to check

    Returns:
        True if value is empty/missing
    """
    # None, empty string, empty list/dict
    if value is None or value == "" or value == [] or value == {}:
        return True

    # Placeholder values that LLMs sometimes use
    if isinstance(value, str) and value.lower() in [
        "n/a", "na", "none", "null", "unknown", "not found", "pending", "not available"
    ]:
        return True

    return False


# ============================================================================
# Adaptive Extraction Functions
# ============================================================================

def identify_missing_fields(
    result: dict[str, Any],
    schema: JsonSchema,
    empty_threshold: float = 0.3,
    prefix: str = ""
) -> list[str]:
    """
    Recursively identify missing leaf fields in nested structures.

    Args:
        result: Extraction result (nested dict)
        schema: JSON Schema (nested)
        empty_threshold: Threshold for triggering re-extraction (0.0-1.0)
        prefix: Current path prefix (for recursion)

    Returns:
        List of dot-notation paths for missing fields
        Example: ["claim.review.review_date", "claimant.personal_info.email"]

    Algorithm:
        1. If schema is not an object, check if value is empty
        2. If schema is an object, recurse into each property
        3. Collect all missing leaf field paths
        4. Return paths if empty_rate > threshold (only at top level)
    """
    if schema.get("type") != "object":
        log.debug("adaptive_extraction_skipped", reason="not_object_schema")
        return []

    missing_fields = []
    properties = schema.get("properties", {})

    if not properties:
        return []

    # Recurse into each property
    for field_name, field_schema in properties.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        field_value = get_nested_value(result, field_path) if prefix else result.get(field_name)

        if field_schema.get("type") == "object":
            # Recurse into nested object
            nested_missing = identify_missing_fields(
                result,
                field_schema,
                empty_threshold=1.0,  # Don't check threshold in recursion
                prefix=field_path
            )
            missing_fields.extend(nested_missing)
        else:
            # Leaf field - check if empty
            if is_empty_value(field_value):
                missing_fields.append(field_path)

    # Check threshold only at top level
    if not prefix:
        total_fields = count_leaf_fields(schema)
        empty_rate = len(missing_fields) / total_fields if total_fields > 0 else 0

        log.info(
            "missing_fields_identified",
            total_fields=total_fields,
            empty_fields=len(missing_fields),
            empty_percentage=f"{empty_rate * 100:.1f}%",
            should_retry=empty_rate > empty_threshold
        )

        if empty_rate <= empty_threshold:
            return []  # Don't retry

    return missing_fields


def _get_field_schema(schema: JsonSchema, segments: list[str]) -> JsonSchema:
    """
    Navigate schema to get field definition.

    Args:
        schema: JSON Schema to navigate
        segments: Path segments (e.g., ["claim", "review", "review_date"])

    Returns:
        Field schema at the path

    Raises:
        ValueError: If path is invalid
    """
    current = schema

    for segment in segments:
        if current.get("type") != "object":
            raise ValueError(f"Cannot navigate into non-object: {segment}")

        properties = current.get("properties", {})
        if segment not in properties:
            raise ValueError(f"Field not found: {segment}")

        current = properties[segment]

    return current


def _add_nested_field(
    schema: JsonSchema,
    segments: list[str],
    field_schema: JsonSchema
) -> None:
    """
    Add a nested field to the schema.

    Args:
        schema: Schema to modify (modified in place)
        segments: Path segments (e.g., ["claim", "review", "review_date"])
        field_schema: Schema for the leaf field
    """
    current = schema

    # Navigate/create nested structure
    for segment in segments[:-1]:
        if "properties" not in current:
            current["properties"] = {}

        if segment not in current["properties"]:
            current["properties"][segment] = {
                "type": "object",
                "properties": {}
            }

        current = current["properties"][segment]

    # Add leaf field
    if "properties" not in current:
        current["properties"] = {}

    current["properties"][segments[-1]] = field_schema


def create_focused_schema(
    original_schema: JsonSchema,
    dot_paths: list[str]
) -> JsonSchema:
    """
    Create a focused schema containing only the specified leaf fields.

    Args:
        original_schema: Full JSON Schema
        dot_paths: List of dot-notation paths to include

    Returns:
        Focused JSON Schema with only specified fields

    Example:
        Input paths: ["claim.review.review_date", "claim.audit.audit_date"]
        Output schema:
        {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "object",
                    "properties": {
                        "review": {
                            "type": "object",
                            "properties": {
                                "review_date": {<original field schema>}
                            }
                        },
                        "audit": {
                            "type": "object",
                            "properties": {
                                "audit_date": {<original field schema>}
                            }
                        }
                    }
                }
            }
        }
    """
    if original_schema.get("type") != "object":
        return original_schema

    focused_schema: JsonSchema = {
        "type": "object",
        "properties": {}
    }

    for dot_path in dot_paths:
        segments = dot_path.split(".")

        try:
            # Get field schema from original
            field_schema = _get_field_schema(original_schema, segments)

            # Build nested structure in focused schema
            _add_nested_field(focused_schema, segments, field_schema)
        except ValueError as e:
            log.warning(
                "focused_schema_field_skipped",
                field=dot_path,
                error=str(e)
            )
            continue

    total_leaf_fields = count_leaf_fields(focused_schema)

    log.debug(
        "focused_schema_created",
        original_fields=count_leaf_fields(original_schema),
        focused_fields=total_leaf_fields,
        target_paths=len(dot_paths)
    )

    return focused_schema


def create_focused_prompt(
    original_prompt: str | None,
    missing_fields: list[str],
    pass_number: int = 2,
    pass1_data: dict[str, Any] | None = None,
    document_info: dict[str, Any] | None = None,
    schema_info: dict[str, Any] | None = None
) -> str:
    """
    Create a focused prompt for re-extracting missing nested fields.

    Enhanced with Pass 1 context to help LLM understand the document better.

    Args:
        original_prompt: Original user prompt
        missing_fields: Dot-notation paths of missing fields
        pass_number: Current pass number
        pass1_data: Successfully extracted data from Pass 1 (for context)
        document_info: Document complexity analysis
        schema_info: Schema complexity analysis

    Returns:
        Focused prompt emphasizing missing fields with context

    Example:
        With context, the prompt includes:
        - Document characteristics (pages, type)
        - Successfully extracted fields from Pass 1
        - Extraction strategy hints based on what was found
    """
    # Build focused prompt with context
    prompt_parts = [
        f"FOCUSED RE-EXTRACTION (Pass {pass_number}/2)",
        ""
    ]

    # Add document context if available
    if document_info and schema_info:
        doc_types = ", ".join(document_info.get("document_types", ["unknown"]))
        prompt_parts.extend([
            "DOCUMENT CONTEXT:",
            f"- Document Type: {doc_types}",
            f"- Total Pages: {document_info.get('total_pages', 'unknown')}",
            f"- Schema Complexity: {schema_info.get('num_fields', 'unknown')} fields across {schema_info.get('nesting_depth', 'unknown')} nesting levels",
            ""
        ])

    # Add successfully extracted fields from Pass 1 (for context)
    if pass1_data:
        # Collect non-empty fields from Pass 1
        successful_fields = []

        def collect_successful_fields(data: dict[str, Any], prefix: str = "") -> None:
            for key, value in data.items():
                field_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Recurse into nested objects
                    collect_successful_fields(value, field_path)
                elif not is_empty_value(value):
                    # Non-empty leaf field
                    # Truncate long values for readability
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    successful_fields.append((field_path, value_str))

        collect_successful_fields(pass1_data)

        if successful_fields:
            # Show first 15 successful fields for context
            num_to_show = min(15, len(successful_fields))
            total_successful = len(successful_fields)

            prompt_parts.extend([
                f"SUCCESSFULLY EXTRACTED FIELDS (Pass 1 found {total_successful} fields):",
                "These fields were already found - use them as context clues:",
                ""
            ])

            for field_path, value in successful_fields[:num_to_show]:
                prompt_parts.append(f"✓ {field_path}: {value}")

            if total_successful > num_to_show:
                prompt_parts.append(f"... and {total_successful - num_to_show} more fields")

            prompt_parts.append("")

    # Group missing fields by top-level section
    sections: dict[str, list[str]] = {}
    for path in missing_fields:
        section = path.split(".")[0]
        if section not in sections:
            sections[section] = []
        sections[section].append(path)

    prompt_parts.extend([
        "MISSING FIELDS TO EXTRACT:",
        f"The following {len(missing_fields)} fields were not found in Pass 1.",
        "Please search the document carefully for these specific fields:",
        ""
    ])

    for section, paths in sorted(sections.items()):
        prompt_parts.append(f"{section.upper()} SECTION:")
        for path in sorted(paths):
            prompt_parts.append(f"  - {path}")
        prompt_parts.append("")

    # Add extraction strategy hints based on Pass 1 results
    if pass1_data and sections:
        prompt_parts.extend([
            "EXTRACTION STRATEGY:",
            "Based on the successfully extracted fields above, look for missing fields in:",
        ])

        for section in sorted(sections.keys()):
            prompt_parts.append(f"- {section.upper()}: Near other {section} fields that were found")

        prompt_parts.append("")

    prompt_parts.extend([
        "IMPORTANT EXTRACTION RULES:",
        "1. Extract ONLY actual filled-in data that is explicitly present in the document",
        "2. Use the successfully extracted fields above as context clues for document structure",
        "3. Field labels (e.g., 'CUSTOMER NAME:', 'ACCOUNT #:') are NOT data - skip them",
        "4. Empty fields with underscores/lines (e.g., '____', '___________') should return null",
        "5. If a field has actual handwritten or typed data, extract it exactly as shown",
        "6. Do NOT infer, guess, or create example/placeholder data (e.g., 'John Doe', '123 Main St')",
        "7. When in doubt, return null rather than guessing",
        ""
    ])

    if original_prompt:
        prompt_parts.append(f"Original instructions: {original_prompt}")

    return "\n".join(prompt_parts)


def merge_extraction_results(
    pass1_data: dict[str, Any],
    pass2_data: dict[str, Any],
    target_paths: list[str]
) -> dict[str, Any]:
    """
    Deep merge Pass 1 and Pass 2 results.

    Strategy:
        - Start with Pass 1 data (complete structure)
        - For each target path, replace with Pass 2 value if non-empty
        - Preserve Pass 1 values for non-target fields

    Args:
        pass1_data: Results from Pass 1 (all fields)
        pass2_data: Results from Pass 2 (focused fields)
        target_paths: Dot-notation paths that were re-extracted

    Returns:
        Merged result with best values from both passes

    Example:
        pass1_data = {
            "claim": {
                "claim_id": "CLM-123",
                "review": {
                    "reviewer_name": "John",
                    "review_date": None  ← empty
                }
            }
        }

        pass2_data = {
            "claim": {
                "review": {
                    "review_date": "2025-01-15"  ← found in Pass 2
                }
            }
        }

        target_paths = ["claim.review.review_date"]

        Result = {
            "claim": {
                "claim_id": "CLM-123",  ← from Pass 1
                "review": {
                    "reviewer_name": "John",  ← from Pass 1
                    "review_date": "2025-01-15"  ← from Pass 2
                }
            }
        }
    """
    import copy
    merged = copy.deepcopy(pass1_data)

    improvements = 0
    for dot_path in target_paths:
        pass2_value = get_nested_value(pass2_data, dot_path)
        pass1_value = get_nested_value(pass1_data, dot_path)

        # Only update if Pass 2 found a non-empty value
        if not is_empty_value(pass2_value) and pass2_value != pass1_value:
            set_nested_value(merged, dot_path, pass2_value)
            improvements += 1
            log.debug("field_improved", field=dot_path)

    log.info(
        "extraction_results_merged",
        pass2_target_fields=len(target_paths),
        improvements=improvements,
        improvement_rate=f"{improvements/len(target_paths)*100:.1f}%" if target_paths else "0%"
    )

    return merged


# ============================================================================
# Intelligent Threshold Calculation
# ============================================================================

def analyze_schema_complexity(schema: JsonSchema) -> dict[str, Any]:
    """
    Analyze schema complexity to determine adaptive threshold.

    Analyzes:
    - Number of leaf fields (terminal fields in nested structures)
    - Nesting depth (how many levels deep)
    - Whether it's an array schema
    - Overall complexity score (0.0-1.0)

    Args:
        schema: JSON Schema object

    Returns:
        {
            "num_fields": int,           # Total leaf fields
            "nesting_depth": int,        # Maximum nesting depth
            "is_array_schema": bool,     # True if top-level is array
            "complexity_score": float    # 0.0-1.0 (simple to complex)
        }

    Example:
        Simple schema (5 fields, flat):
        → {"num_fields": 5, "nesting_depth": 1, "complexity_score": 0.15}

        Complex schema (50 fields, 4 levels deep):
        → {"num_fields": 50, "nesting_depth": 4, "complexity_score": 0.65}
    """
    # Check if array schema
    is_array_schema = schema.get("type") == "array"

    # For array schemas, analyze the items schema
    analysis_schema = schema.get("items", {}) if is_array_schema else schema

    # Count leaf fields
    num_fields = count_leaf_fields(analysis_schema)

    # Calculate nesting depth
    def get_max_depth(s: JsonSchema, current_depth: int = 1) -> int:
        if s.get("type") != "object":
            return current_depth

        properties = s.get("properties", {})
        if not properties:
            return current_depth

        max_child_depth = current_depth
        for field_schema in properties.values():
            if field_schema.get("type") == "object":
                child_depth = get_max_depth(field_schema, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    nesting_depth = get_max_depth(analysis_schema)

    # Calculate complexity score (0.0-1.0)
    # Based on number of fields and nesting depth
    field_score = min(num_fields / 100.0, 1.0)  # 100+ fields = max
    depth_score = min(nesting_depth / 10.0, 1.0)  # 10+ levels = max
    complexity_score = (field_score + depth_score) / 2.0

    result = {
        "num_fields": num_fields,
        "nesting_depth": nesting_depth,
        "is_array_schema": is_array_schema,
        "complexity_score": complexity_score
    }

    log.debug(
        "schema_complexity_analyzed",
        **result
    )

    return result


def analyze_document_complexity(files: list[str]) -> dict[str, Any]:
    """
    Analyze document complexity.

    Analyzes:
    - Number of files
    - Total pages (for PDFs)
    - Estimated tokens
    - Document types

    Args:
        files: List of file paths

    Returns:
        {
            "num_files": int,
            "total_pages": int,
            "estimated_tokens": int,
            "document_types": list[str]
        }

    Example:
        Single 2-page PDF:
        → {"num_files": 1, "total_pages": 2, "estimated_tokens": 5000, "document_types": ["pdf"]}

        Large 100-page PDF:
        → {"num_files": 1, "total_pages": 100, "estimated_tokens": 250000, "document_types": ["pdf"]}
    """
    import os
    from pathlib import Path

    num_files = len(files)
    total_pages = 0
    estimated_tokens = 0
    document_types = set()

    for file_path in files:
        # Get file extension
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            document_types.add("pdf")

            # Try to get page count from PDF
            try:
                from .pdf_analyzer import PDFAnalyzer
                analyzer = PDFAnalyzer()
                analysis = analyzer.analyze(file_path)
                total_pages += analysis.total_pages

                # Estimate tokens: ~2500 tokens per page for text-rich PDFs
                estimated_tokens += analysis.total_pages * 2500
            except Exception as e:
                log.debug("pdf_analysis_failed", file=file_path, error=str(e))
                # Fallback: estimate based on file size
                file_size = os.path.getsize(file_path)
                estimated_pages = max(1, file_size // 50000)  # ~50KB per page
                total_pages += estimated_pages
                estimated_tokens += estimated_pages * 2500

        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            document_types.add("image")
            total_pages += 1
            estimated_tokens += 1000  # Images typically use fewer tokens

        else:
            document_types.add("other")
            # For text files, estimate based on file size
            try:
                file_size = os.path.getsize(file_path)
                estimated_tokens += file_size // 4  # ~4 chars per token
            except Exception:
                estimated_tokens += 1000  # Default estimate

    result = {
        "num_files": num_files,
        "total_pages": total_pages,
        "estimated_tokens": estimated_tokens,
        "document_types": sorted(list(document_types))
    }

    log.debug(
        "document_complexity_analyzed",
        **result
    )

    return result


def estimate_array_instances(
    schema_complexity: dict[str, Any],
    document_complexity: dict[str, Any]
) -> int:
    """
    Estimate number of instances for array schemas.

    Uses heuristics based on document size:
    - Small documents (< 5 pages): ~5 instances per page
    - Medium documents (5-50 pages): ~10 instances per page
    - Large documents (> 50 pages): ~15 instances per page

    Args:
        schema_complexity: Schema complexity analysis
        document_complexity: Document complexity analysis

    Returns:
        Estimated number of instances

    Example:
        120-page PDF with array schema:
        → 120 * 15 = 1800 estimated instances
    """
    if not schema_complexity["is_array_schema"]:
        return 0

    total_pages = document_complexity["total_pages"]

    # Heuristic: items per page based on document size
    if total_pages < 5:
        items_per_page = 5
    elif total_pages < 50:
        items_per_page = 10
    else:
        items_per_page = 15

    estimated_instances = max(1, total_pages * items_per_page)

    log.debug(
        "array_instances_estimated",
        total_pages=total_pages,
        items_per_page=items_per_page,
        estimated_instances=estimated_instances
    )

    return estimated_instances


def calculate_adaptive_threshold(
    schema_complexity: dict[str, Any],
    document_complexity: dict[str, Any],
    base_threshold: float = 0.3
) -> float:
    """
    Calculate intelligent adaptive threshold based on complexity.

    Formula:
        base_threshold (0.3)
        + schema complexity adjustment (-0.1 to +0.1)
        + nesting depth adjustment (0 to +0.05)
        + document size adjustment (-0.05 to +0.05)
        + array instances adjustment (0 to +0.1)
        = final threshold (clamped to 0.1-0.6)

    Logic:
    - Simple schemas (< 10 fields) → Lower threshold (stricter)
    - Complex schemas (> 50 fields) → Higher threshold (more lenient)
    - Deep nesting (> 3 levels) → Higher threshold
    - Small documents (< 5 pages) → Lower threshold
    - Large documents (> 50 pages) → Higher threshold
    - Large arrays (> 100 instances) → Higher threshold

    Args:
        schema_complexity: Schema complexity analysis
        document_complexity: Document complexity analysis
        base_threshold: Base threshold (default: 0.3)

    Returns:
        Adaptive threshold between 0.1 and 0.6

    Examples:
        Simple 5-field schema, 2-page PDF:
        → 0.3 - 0.1 - 0.05 = 0.15 (strict)

        Complex 50-field nested schema, 100-page PDF:
        → 0.3 + 0.1 + 0.05 + 0.05 = 0.50 (lenient)

        Array schema with 500 instances:
        → 0.3 + 0.1 = 0.40 (lenient)
    """
    adjustment = 0.0

    # Schema complexity adjustment
    num_fields = schema_complexity["num_fields"]
    if num_fields < 10:
        adjustment -= 0.1  # Stricter for simple schemas
        reason_schema = "simple (<10 fields)"
    elif num_fields > 50:
        adjustment += 0.1  # More lenient for complex schemas
        reason_schema = "complex (>50 fields)"
    else:
        reason_schema = "medium (10-50 fields)"

    # Nesting depth adjustment
    nesting_depth = schema_complexity["nesting_depth"]
    if nesting_depth > 3:
        adjustment += 0.05  # More lenient for deep nesting
        reason_nesting = "deep (>3 levels)"
    else:
        reason_nesting = "shallow (≤3 levels)"

    # Document size adjustment
    total_pages = document_complexity["total_pages"]
    if total_pages > 50:
        adjustment += 0.05  # More lenient for large documents
        reason_pages = "large (>50 pages)"
    elif total_pages < 5:
        adjustment -= 0.05  # Stricter for small documents
        reason_pages = "small (<5 pages)"
    else:
        reason_pages = "medium (5-50 pages)"

    # Array instances adjustment
    if schema_complexity["is_array_schema"]:
        estimated_instances = estimate_array_instances(schema_complexity, document_complexity)
        if estimated_instances > 100:
            adjustment += 0.1  # More lenient for large arrays
            reason_array = f"large array (~{estimated_instances} instances)"
        else:
            reason_array = f"small array (~{estimated_instances} instances)"
    else:
        reason_array = "not array"

    # Calculate final threshold
    final_threshold = base_threshold + adjustment

    # Clamp to reasonable range [0.1, 0.6]
    final_threshold = max(0.1, min(0.6, final_threshold))

    log.info(
        "adaptive_threshold_calculated",
        base_threshold=f"{base_threshold:.2f}",
        adjustment=f"{adjustment:+.2f}",
        final_threshold=f"{final_threshold:.2f}",
        schema=reason_schema,
        nesting=reason_nesting,
        pages=reason_pages,
        array=reason_array,
        num_fields=num_fields,
        nesting_depth=nesting_depth,
        total_pages=total_pages
    )

    return final_threshold


async def extract_with_adaptive_retry(
    files: list[str],
    schema: JsonSchema,
    config: RuntimeConfig,
    user_prompt: str | None,
    examples: list[Any] | None,
    include_extra: bool,
    empty_threshold: float = 0.3,
    max_passes: int = 2
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extract data with adaptive retry for missing fields.
    
    Process:
    1. Pass 1: Extract all fields at once
    2. Identify missing/empty fields
    3. Pass 2: Re-extract only missing fields with focused prompt
    4. Merge results
    
    Args:
        files: List of file paths
        schema: JSON schema
        config: Runtime configuration
        user_prompt: Optional user prompt
        examples: Optional examples
        include_extra: Include extra fields
        empty_threshold: Trigger retry if > this % of fields are empty (default: 30%)
        max_passes: Maximum number of extraction passes (default: 2)
    
    Returns:
        (merged_data, report)
    """
    from .agent_runner import run_extraction_async

    # Analyze complexity and calculate intelligent threshold
    schema_complexity = analyze_schema_complexity(schema)
    document_complexity = analyze_document_complexity(files)

    # Calculate adaptive threshold (overrides the parameter if not explicitly set)
    # If user explicitly passed a threshold, use it; otherwise calculate
    if empty_threshold == 0.3:  # Default value, calculate intelligent threshold
        calculated_threshold = calculate_adaptive_threshold(
            schema_complexity,
            document_complexity
        )
    else:
        # User explicitly set threshold, respect it
        calculated_threshold = empty_threshold
        log.info(
            "using_explicit_threshold",
            threshold=f"{empty_threshold:.2f}",
            reason="user_specified"
        )

    log.info(
        "adaptive_extraction_started",
        num_fields=schema_complexity["num_fields"],
        nesting_depth=schema_complexity["nesting_depth"],
        total_pages=document_complexity["total_pages"],
        threshold=f"{calculated_threshold:.0%}",
        max_passes=max_passes
    )

    # Pass 1: Extract all fields
    log.info("extraction_pass_1_started", strategy="extract_all_fields")

    pass1_data, pass1_report = await run_extraction_async(
        config=config,
        files=files,
        schema_or_model=schema,
        user_prompt=user_prompt,
        examples=examples,
        include_extra=include_extra,
        return_pydantic=False
    )
    
    log.info(
        "extraction_pass_1_complete",
        fields_extracted=len(pass1_data),
        populated_fields=sum(1 for v in pass1_data.values() if v)
    )

    # Identify missing fields using calculated threshold
    missing_fields = identify_missing_fields(pass1_data, schema, calculated_threshold)
    
    if not missing_fields:
        log.info(
            "adaptive_extraction_complete",
            reason="no_missing_fields",
            total_passes=1
        )

        # Convert ExtractionReport to dict
        report_dict = {
            "model": pass1_report.model,
            "files": pass1_report.files,
            "usage": pass1_report.usage,
            "cost_estimate_usd": pass1_report.cost_estimate_usd,
            "warnings": pass1_report.warnings or [],
            "adaptive_extraction": {
                "enabled": True,
                "total_passes": 1,
                "missing_fields_pass1": [],
                "improvements_pass2": [],
                "pass1_usage": pass1_report.usage,
                "pass2_usage": {}
            }
        }

        return pass1_data, report_dict
    
    # Pass 2: Re-extract missing fields
    log.info(
        "extraction_pass_2_started",
        strategy="focused_re_extraction",
        target_fields=len(missing_fields),
        fields=missing_fields[:10]  # Log first 10
    )
    
    focused_schema = create_focused_schema(schema, missing_fields)
    focused_prompt = create_focused_prompt(
        user_prompt,
        missing_fields,
        pass_number=2,
        pass1_data=pass1_data,
        document_info=document_complexity,
        schema_info=schema_complexity
    )
    
    pass2_data, pass2_report = await run_extraction_async(
        config=config,
        files=files,
        schema_or_model=focused_schema,
        user_prompt=focused_prompt,
        examples=examples,
        include_extra=include_extra,
        return_pydantic=False
    )
    
    log.info(
        "extraction_pass_2_complete",
        target_fields=len(missing_fields),
        fields_found=sum(1 for v in pass2_data.values() if v)
    )
    
    # Merge results
    merged_data = merge_extraction_results(pass1_data, pass2_data, missing_fields)

    # Aggregate usage from both passes
    total_usage = {
        "requests": pass1_report.usage.get("requests", 0) + pass2_report.usage.get("requests", 0),
        "input_tokens": pass1_report.usage.get("input_tokens", 0) + pass2_report.usage.get("input_tokens", 0),
        "output_tokens": pass1_report.usage.get("output_tokens", 0) + pass2_report.usage.get("output_tokens", 0),
        "total_tokens": pass1_report.usage.get("total_tokens", 0) + pass2_report.usage.get("total_tokens", 0)
    }

    # Calculate total cost
    total_cost = (pass1_report.cost_estimate_usd or 0) + (pass2_report.cost_estimate_usd or 0)

    # Identify improvements (fields that were empty in Pass 1 but populated in merged result)
    improvements: list[str] = []
    for field_path in missing_fields:
        merged_value = get_nested_value(merged_data, field_path)
        pass1_value = get_nested_value(pass1_data, field_path)
        if merged_value != pass1_value:
            improvements.append(field_path)

    # Create combined report
    report = {
        "model": config.model,
        "files": files,
        "usage": total_usage,
        "cost_estimate_usd": total_cost,
        "warnings": [],
        "adaptive_extraction": {
            "enabled": True,
            "total_passes": 2,
            "missing_fields_pass1": missing_fields,
            "improvements_pass2": improvements,
            "pass1_usage": pass1_report.usage,
            "pass2_usage": pass2_report.usage
        }
    }
    
    log.info(
        "adaptive_extraction_complete",
        total_passes=2,
        missing_fields_pass1=len(missing_fields),
        improvements_pass2=report["adaptive_extraction"]["improvements_pass2"],
        total_cost=report["cost_estimate_usd"]
    )
    
    return merged_data, report
