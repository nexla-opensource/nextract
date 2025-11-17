"""
Multi-pass extraction for improved recall and accuracy.

Runs extraction multiple times and merges results using configurable
strategies to improve recall and handle variability in LLM outputs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import structlog

from .schema import JsonSchema
from .config import RuntimeConfig

log = structlog.get_logger(__name__)


@dataclass
class PassResult:
    """Result from a single extraction pass"""
    pass_number: int
    data: Dict[str, Any]
    usage: Dict[str, Any]
    cost: Optional[float] = None
    errors: List[str] = None


@dataclass
class MultiPassResult:
    """Result from multi-pass extraction"""
    merged_data: Dict[str, Any]
    pass_results: List[PassResult]
    total_passes: int
    successful_passes: int
    failed_passes: int
    merge_strategy: str
    total_usage: Dict[str, Any]
    total_cost: Optional[float] = None


class MultiPassExtractor:
    """
    Run extraction multiple times and merge results.
    
    Multi-pass extraction improves recall by running the same extraction
    multiple times and merging the results. This helps handle:
    - LLM variability (different outputs on same input)
    - Missed fields in single pass
    - Improved confidence through consensus
    
    Merge strategies:
    - "union": Take all non-empty values from all passes
    - "intersection": Only keep values that appear in all passes
    - "majority": Keep values that appear in majority of passes
    - "highest_confidence": Keep values with highest confidence (requires provenance)
    - "first_non_empty": Use first non-empty value found
    
    Example:
        extractor = MultiPassExtractor(num_passes=3)
        
        result = await extractor.extract_multipass(
            extraction_fn=my_extraction_function,
            schema=schema,
            merge_strategy="union"
        )
        
        print(f"Merged data: {result.merged_data}")
        print(f"Successful passes: {result.successful_passes}/{result.total_passes}")
    """
    
    def __init__(
        self,
        num_passes: int = 3,
        fail_threshold: Optional[int] = None
    ):
        """
        Initialize multi-pass extractor.
        
        Args:
            num_passes: Number of extraction passes to run (default: 3)
            fail_threshold: Maximum number of failed passes before giving up
                          (default: num_passes - 1, i.e., at least 1 must succeed)
        
        Raises:
            ValueError: If num_passes < 1
        """
        if num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {num_passes}")
        
        self.num_passes = num_passes
        self.fail_threshold = fail_threshold or (num_passes - 1)
        
        log.info(
            "multipass_extractor_initialized",
            num_passes=num_passes,
            fail_threshold=self.fail_threshold
        )
    
    async def extract_multipass(
        self,
        extraction_fn: Callable[..., Any],
        schema: JsonSchema,
        merge_strategy: str = "union",
        **extraction_kwargs
    ) -> MultiPassResult:
        """
        Run multi-pass extraction.
        
        Args:
            extraction_fn: Async function that performs extraction
                          Should return tuple of (data, report)
            schema: JSON schema for extraction
            merge_strategy: Strategy for merging results
            **extraction_kwargs: Additional arguments to pass to extraction_fn
        
        Returns:
            MultiPassResult with merged data and pass details
        
        Raises:
            ValueError: If merge_strategy is not recognized
            MultiPassExtractionError: If too many passes fail
        """
        if merge_strategy not in ["union", "intersection", "majority", "highest_confidence", "first_non_empty"]:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        log.info(
            "multipass_extraction_started",
            num_passes=self.num_passes,
            merge_strategy=merge_strategy
        )
        
        pass_results = []
        failed_passes = 0
        
        # Run extraction passes
        for pass_num in range(1, self.num_passes + 1):
            log.info("extraction_pass_started", pass_number=pass_num)
            
            try:
                # Run extraction
                data, report = await extraction_fn(schema=schema, **extraction_kwargs)
                
                # Extract usage and cost from report
                usage = report.get("usage", {})
                cost = report.get("cost_estimate_usd")
                errors = report.get("warnings", [])
                
                pass_result = PassResult(
                    pass_number=pass_num,
                    data=data,
                    usage=usage,
                    cost=cost,
                    errors=errors
                )
                
                pass_results.append(pass_result)
                
                log.info(
                    "extraction_pass_completed",
                    pass_number=pass_num,
                    fields_extracted=len(data),
                    cost=cost
                )
            
            except Exception as e:
                failed_passes += 1
                
                log.error(
                    "extraction_pass_failed",
                    pass_number=pass_num,
                    error=str(e),
                    failed_passes=failed_passes,
                    fail_threshold=self.fail_threshold
                )
                
                # Check if we've exceeded failure threshold
                if failed_passes > self.fail_threshold:
                    raise MultiPassExtractionError(
                        f"Too many extraction passes failed: {failed_passes}/{self.num_passes}",
                        pass_results=pass_results,
                        failed_passes=failed_passes
                    )
                
                # Continue to next pass
                continue
        
        # Check if we have any successful passes
        if not pass_results:
            raise MultiPassExtractionError(
                f"All {self.num_passes} extraction passes failed",
                pass_results=[],
                failed_passes=failed_passes
            )
        
        # Merge results
        log.info(
            "merging_pass_results",
            successful_passes=len(pass_results),
            merge_strategy=merge_strategy
        )
        
        merged_data = self._merge_results(
            pass_results=[pr.data for pr in pass_results],
            schema=schema,
            strategy=merge_strategy
        )
        
        # Aggregate usage
        total_usage = self._aggregate_usage([pr.usage for pr in pass_results])
        
        # Sum costs
        total_cost = None
        if any(pr.cost is not None for pr in pass_results):
            total_cost = sum(pr.cost for pr in pass_results if pr.cost is not None)
        
        result = MultiPassResult(
            merged_data=merged_data,
            pass_results=pass_results,
            total_passes=self.num_passes,
            successful_passes=len(pass_results),
            failed_passes=failed_passes,
            merge_strategy=merge_strategy,
            total_usage=total_usage,
            total_cost=total_cost
        )
        
        log.info(
            "multipass_extraction_completed",
            successful_passes=len(pass_results),
            failed_passes=failed_passes,
            merged_fields=len(merged_data),
            total_cost=total_cost
        )
        
        return result
    
    def _merge_results(
        self,
        pass_results: List[Dict[str, Any]],
        schema: JsonSchema,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Merge results from multiple passes.
        
        Args:
            pass_results: List of data dicts from each pass
            schema: JSON schema
            strategy: Merge strategy
        
        Returns:
            Merged data dict
        """
        if not pass_results:
            return {}
        
        if len(pass_results) == 1:
            return pass_results[0]
        
        # Get all field names from schema
        field_names = set()
        if "properties" in schema:
            field_names = set(schema["properties"].keys())
        
        # Also get field names from results
        for result in pass_results:
            field_names.update(result.keys())
        
        merged = {}
        
        for field_name in field_names:
            # Collect values from all passes
            values = []
            for result in pass_results:
                value = result.get(field_name)
                if value is not None and value != "" and value != []:
                    values.append(value)
            
            # Apply merge strategy
            if strategy == "union" or strategy == "first_non_empty":
                # Take first non-empty value
                if values:
                    merged[field_name] = values[0]
            
            elif strategy == "intersection":
                # Only keep if present in all passes
                if len(values) == len(pass_results):
                    # All passes have this field
                    # Use most common value
                    merged[field_name] = self._most_common_value(values)
            
            elif strategy == "majority":
                # Keep if present in majority of passes
                if len(values) > len(pass_results) / 2:
                    merged[field_name] = self._most_common_value(values)
            
            elif strategy == "highest_confidence":
                # For now, same as union (requires provenance integration)
                if values:
                    merged[field_name] = values[0]
        
        return merged
    
    def _most_common_value(self, values: List[Any]) -> Any:
        """Get most common value from list"""
        if not values:
            return None
        
        # Count occurrences
        value_counts = {}
        for value in values:
            # Convert to string for comparison
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1
        
        # Find most common
        most_common_str = max(value_counts, key=value_counts.get)
        
        # Return original value (not string)
        for value in values:
            if str(value) == most_common_str:
                return value
        
        return values[0]
    
    def _aggregate_usage(self, usage_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate usage from multiple passes"""
        total = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        for usage in usage_list:
            total["requests"] += usage.get("requests", 0)
            total["input_tokens"] += usage.get("input_tokens", 0)
            total["output_tokens"] += usage.get("output_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)
        
        return total


class MultiPassExtractionError(Exception):
    """Raised when multi-pass extraction fails"""
    
    def __init__(
        self,
        message: str,
        pass_results: List[PassResult],
        failed_passes: int
    ):
        super().__init__(message)
        self.pass_results = pass_results
        self.failed_passes = failed_passes
    
    def __str__(self):
        return (
            f"{super().__str__()}\n"
            f"Successful passes: {len(self.pass_results)}\n"
            f"Failed passes: {self.failed_passes}"
        )

