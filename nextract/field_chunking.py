"""
Field-based chunking for large schemas.

When a schema has too many fields (e.g., 50+), extracting all fields at once
can lead to:
- Cognitive overload for the LLM
- Slower responses
- Higher error rates
- Missed fields

This module implements "field chunking" - breaking a large schema into smaller
field groups, extracting each group separately, then stitching results together.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import structlog

from .config import RuntimeConfig
from .schema import JsonSchema

log = structlog.get_logger(__name__)


def should_chunk_fields(
    schema: JsonSchema,
    document_size: int,
    max_fields_threshold: int = 30
) -> bool:
    """
    Determine if schema should be chunked into field groups.
    
    Args:
        schema: JSON schema to analyze
        document_size: Size of document in characters
        max_fields_threshold: Maximum fields before chunking (default: 30)
    
    Returns:
        True if schema should be chunked, False otherwise
    """
    # Only chunk object schemas
    if schema.get("type") != "object":
        return False
    
    properties = schema.get("properties", {})
    num_fields = len(properties)
    
    # Small schemas: no chunking
    if num_fields <= 20:
        log.debug(
            "field_chunking_not_needed",
            num_fields=num_fields,
            reason="schema_too_small"
        )
        return False
    
    # Very large schemas: always chunk
    if num_fields >= max_fields_threshold:
        log.info(
            "field_chunking_recommended",
            num_fields=num_fields,
            reason="schema_very_large"
        )
        return True
    
    # Medium schemas (20-30 fields): check document size
    if document_size > 50000:  # ~50KB
        log.info(
            "field_chunking_recommended",
            num_fields=num_fields,
            document_size=document_size,
            reason="large_document_with_medium_schema"
        )
        return True
    
    log.debug(
        "field_chunking_not_needed",
        num_fields=num_fields,
        document_size=document_size,
        reason="manageable_size"
    )
    return False


def group_fields_semantically(
    schema: JsonSchema,
    max_fields_per_group: int = 15
) -> list[dict[str, Any]]:
    """
    Group schema fields into semantic clusters.
    
    Strategy:
    1. Identify field groups by naming patterns (e.g., "claimant_*", "policy_*")
    2. Create sub-schemas for each group
    3. Ensure no group exceeds max_fields_per_group
    
    Args:
        schema: JSON schema with "properties"
        max_fields_per_group: Maximum fields per group (default: 15)
    
    Returns:
        List of sub-schemas, each with a subset of fields
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Define semantic groups based on common naming patterns
    semantic_patterns = {
        "claimant": ["claimant_", "customer_", "insured_"],
        "claim": ["claim_id", "claim_number", "claim_status", "claim_submission"],
        "incident": ["incident_", "loss_", "damage_"],
        "policy": ["policy_", "coverage_", "deductible"],
        "financial": ["amount", "payment", "cost", "price", "approved_amount"],
        "adjuster": ["adjuster_", "inspector_", "reviewer_"],
        "vendor": ["vendor_", "supplier_", "contractor_"],
        "dates": ["_date", "_time", "last_updated", "created_at"],
        "contact": ["email", "phone", "contact", "address"],
        "status": ["status", "state", "flag", "check"],
        "metadata": ["notes", "comments", "attachments", "system_", "data_quality"]
    }
    
    # Assign fields to groups
    field_groups: dict[str, list[str]] = {group: [] for group in semantic_patterns}
    field_groups["other"] = []  # Catch-all for unmatched fields
    
    for field_name in properties:
        assigned = False
        for group_name, patterns in semantic_patterns.items():
            if any(pattern in field_name.lower() for pattern in patterns):
                field_groups[group_name].append(field_name)
                assigned = True
                break
        
        if not assigned:
            field_groups["other"].append(field_name)
    
    # Remove empty groups
    field_groups = {k: v for k, v in field_groups.items() if v}
    
    # Split large groups if they exceed max_fields_per_group
    final_groups = []
    for group_name, fields in field_groups.items():
        if len(fields) <= max_fields_per_group:
            final_groups.append({
                "name": group_name,
                "fields": fields
            })
        else:
            # Split into multiple sub-groups
            num_subgroups = (len(fields) + max_fields_per_group - 1) // max_fields_per_group
            for i in range(num_subgroups):
                start_idx = i * max_fields_per_group
                end_idx = min((i + 1) * max_fields_per_group, len(fields))
                final_groups.append({
                    "name": f"{group_name}_{i+1}",
                    "fields": fields[start_idx:end_idx]
                })
    
    # Create sub-schemas
    sub_schemas = []
    for group in final_groups:
        group_properties = {
            field: properties[field]
            for field in group["fields"]
        }
        
        group_required = [
            field for field in group["fields"]
            if field in required
        ]
        
        sub_schema = {
            "type": "object",
            "properties": group_properties,
            "required": group_required,
            "_group_name": group["name"],  # Metadata
            "_field_count": len(group["fields"])
        }
        
        sub_schemas.append(sub_schema)
    
    log.info(
        "fields_grouped_semantically",
        total_fields=len(properties),
        num_groups=len(sub_schemas),
        group_sizes=[s["_field_count"] for s in sub_schemas],
        group_names=[s["_group_name"] for s in sub_schemas]
    )
    
    return sub_schemas


def merge_field_results(
    results: list[dict[str, Any]],
    original_schema: JsonSchema
) -> dict[str, Any]:
    """
    Merge results from multiple field-chunked extractions.
    
    Strategy:
    - Combine all fields from all results
    - Handle conflicts (same field in multiple results) by taking first non-empty value
    - Validate against original schema
    
    Args:
        results: List of extraction results (one per field group)
        original_schema: Original schema with all fields
    
    Returns:
        Merged result with all fields
    """
    merged = {}
    field_sources = {}  # Track which result each field came from
    
    for i, result in enumerate(results):
        if not result:
            continue
        
        for field_name, value in result.items():
            # Skip metadata fields
            if field_name.startswith("_"):
                continue
            
            # First-non-empty strategy
            if field_name not in merged:
                merged[field_name] = value
                field_sources[field_name] = i
            elif not merged[field_name] and value:
                # Replace empty value with non-empty
                merged[field_name] = value
                field_sources[field_name] = i
    
    log.info(
        "field_results_merged",
        num_results=len(results),
        total_fields=len(merged),
        populated_fields=sum(1 for v in merged.values() if v)
    )
    
    return merged


async def extract_with_field_chunking(
    files: list[str],
    schema: JsonSchema,
    config: RuntimeConfig,
    user_prompt: str | None,
    examples: list[Any] | None,
    include_extra: bool,
    max_fields_per_group: int = 15
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extract data using field chunking strategy.
    
    Process:
    1. Group schema fields into semantic clusters
    2. Extract each field group separately (in parallel)
    3. Merge results
    4. Return combined result
    
    Args:
        files: List of file paths
        schema: Full schema with all fields
        config: Runtime configuration
        user_prompt: Optional user prompt
        examples: Optional examples
        include_extra: Include extra fields
        max_fields_per_group: Maximum fields per group
    
    Returns:
        (merged_data, report)
    """
    from .agent_runner import run_extraction_async
    
    log.info(
        "field_chunking_extraction_started",
        num_fields=len(schema.get("properties", {})),
        max_fields_per_group=max_fields_per_group
    )
    
    # 1. Group fields
    sub_schemas = group_fields_semantically(schema, max_fields_per_group)
    
    # 2. Extract each group in parallel
    tasks = []
    for sub_schema in sub_schemas:
        group_name = sub_schema.pop("_group_name")
        field_count = sub_schema.pop("_field_count")
        
        # Create custom prompt for this field group
        group_prompt = user_prompt or ""
        group_prompt += f"\n\nFocus on extracting these {field_count} fields: {', '.join(sub_schema['properties'].keys())}"
        
        log.debug(
            "extracting_field_group",
            group_name=group_name,
            num_fields=field_count
        )
        
        task = run_extraction_async(
            config=config,
            files=files,
            schema_or_model=sub_schema,
            user_prompt=group_prompt,
            examples=examples,
            include_extra=include_extra,
            return_pydantic=False
        )
        tasks.append(task)
    
    # Wait for all extractions
    group_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 3. Merge results
    successful_results = []
    failed_groups = []
    
    for i, result in enumerate(group_results):
        if isinstance(result, Exception):
            log.error(
                "field_group_extraction_failed",
                group_index=i,
                error=str(result)
            )
            failed_groups.append(i)
        else:
            # run_extraction_async returns (data, report) tuple
            data, report = result
            successful_results.append(data)
    
    merged_data = merge_field_results(successful_results, schema)
    
    # 4. Create report
    report = {
        "model": config.model,
        "files": files,
        "field_chunking": {
            "enabled": True,
            "num_groups": len(sub_schemas),
            "successful_groups": len(successful_results),
            "failed_groups": len(failed_groups),
            "total_fields": len(schema.get("properties", {})),
            "extracted_fields": len(merged_data)
        },
        "warnings": []
    }
    
    if failed_groups:
        report["warnings"].append(
            f"{len(failed_groups)} field group(s) failed extraction"
        )
    
    log.info(
        "field_chunking_extraction_complete",
        successful_groups=len(successful_results),
        failed_groups=len(failed_groups),
        extracted_fields=len(merged_data)
    )
    
    return merged_data, report

