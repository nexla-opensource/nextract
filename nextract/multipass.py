"""
Multi-pass extraction for improved recall and accuracy.

Runs extraction multiple times and merges results using configurable
strategies to improve recall and handle variability in LLM outputs.

Note on pipeline / CLI multipass
--------------------------------
``MultiPassExtractor`` in this module provides full merge strategies
(union, intersection, majority, first_non_empty). ``highest_confidence``
is accepted as a name but raises ``NotImplementedError`` (needs provenance).

Pipeline / CLI ``--num-passes`` (via ``ExtractionPlan.num_passes`` and
``pipeline.orchestrator.ExtractionPipeline._multi_pass_extract``) uses a
simpler re-run + re-merge path (array dedupe or ``merge_partial_outputs``),
not the full strategy set implemented here. Use this class directly when
you need strategy-based multipass merging.
"""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass, field
import structlog

from .schema import JsonSchema

log = structlog.get_logger(__name__)


@dataclass
class PassResult:
    """Result from a single extraction pass"""
    pass_number: int
    data: dict[str, Any]
    usage: dict[str, Any]
    cost: float | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class MultiPassResult:
    """Result from multi-pass extraction"""
    merged_data: dict[str, Any]
    pass_results: list[PassResult]
    total_passes: int
    successful_passes: int
    failed_passes: int
    merge_strategy: str
    total_usage: dict[str, Any]
    total_cost: float | None = None


class MultiPassExtractor:
    """
    Run extraction multiple times and merge results.

    Multi-pass extraction improves recall by running the same extraction
    multiple times and merging the results. This helps handle:
    - LLM variability (different outputs on same input)
    - Missed fields in single pass
    - Improved confidence through consensus

    Merge strategies:
    - "union": Merge list fields by extending; for scalar fields keep the
      first non-empty value (does not wrap scalars in a list)
    - "intersection": Only keep values that appear in all passes
    - "majority": Keep values that appear in majority of passes
    - "highest_confidence": **Not implemented** — raises
      ``NotImplementedError``. True per-field confidence selection
      requires provenance integration.
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
        fail_threshold: int | None = None
    ):
        """
        Initialize multi-pass extractor.

        Args:
            num_passes: Number of extraction passes to run (default: 3)
            fail_threshold: Maximum number of failed passes allowed before
                aborting. ``0`` means zero-tolerance (abort on first failure).
                Default when omitted (``None``): ``num_passes - 1``
                (at least one pass must succeed).

        Raises:
            ValueError: If num_passes < 1 or fail_threshold < 0
        """
        if num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {num_passes}")

        if fail_threshold is None:
            resolved_fail_threshold = num_passes - 1
        else:
            if fail_threshold < 0:
                raise ValueError(
                    f"fail_threshold must be >= 0, got {fail_threshold}"
                )
            resolved_fail_threshold = fail_threshold

        self.num_passes = num_passes
        self.fail_threshold = resolved_fail_threshold

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
            NotImplementedError: If merge_strategy is ``highest_confidence``
            MultiPassExtractionError: If too many passes fail
        """
        if merge_strategy not in [
            "union",
            "intersection",
            "majority",
            "highest_confidence",
            "first_non_empty",
        ]:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        if merge_strategy == "highest_confidence":
            # Fail before any extraction work; strategy needs provenance wiring.
            raise NotImplementedError(
                "merge_strategy='highest_confidence' is not implemented. "
                "It requires per-field confidence/provenance integration. "
                "Use 'union', 'intersection', 'majority', or 'first_non_empty'."
            )

        log.info(
            "multipass_extraction_started",
            num_passes=self.num_passes,
            merge_strategy=merge_strategy
        )

        pass_results: list[PassResult] = []
        failed_passes = 0
        pass_errors: list[str] = []

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
                error_msg = str(e)
                pass_errors.append(error_msg)

                log.error(
                    "extraction_pass_failed",
                    pass_number=pass_num,
                    error=error_msg,
                    failed_passes=failed_passes,
                    fail_threshold=self.fail_threshold
                )

                # Check if we've exceeded failure threshold
                if failed_passes > self.fail_threshold:
                    raise MultiPassExtractionError(
                        f"Too many failed passes: {failed_passes} failed "
                        f"(threshold: {self.fail_threshold})",
                        pass_results=pass_results,
                        failed_passes=failed_passes,
                        errors=pass_errors,
                    )

                # Continue to next pass
                continue

        # Check if we have any successful passes
        if not pass_results:
            raise MultiPassExtractionError(
                f"All {self.num_passes} extraction passes failed",
                pass_results=[],
                failed_passes=failed_passes,
                errors=pass_errors,
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
        pass_results: list[dict[str, Any]],
        schema: JsonSchema,
        strategy: str
    ) -> dict[str, Any]:
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
        field_names: set[str] = set()
        if "properties" in schema:
            field_names = set(schema["properties"].keys())

        # Also get field names from results
        for result in pass_results:
            field_names.update(result.keys())

        merged: dict[str, Any] = {}

        for field_name in field_names:
            # Collect values from all passes
            values: list[Any] = []
            for result in pass_results:
                value = result.get(field_name)
                if value is not None and value != "" and value != []:
                    values.append(value)

            # Apply merge strategy
            if strategy == "union":
                if values:
                    merged[field_name] = self._union_field_values(values)

            elif strategy == "first_non_empty":
                if values:
                    merged[field_name] = values[0]

            elif strategy == "intersection":
                # Only keep if present in all passes
                if len(values) == len(pass_results):
                    # All passes have this field — use most common value
                    merged[field_name] = self._most_common_value(values)

            elif strategy == "majority":
                # Keep if present in majority of passes
                if len(values) > len(pass_results) / 2:
                    merged[field_name] = self._most_common_value(values)

            elif strategy == "highest_confidence":
                raise NotImplementedError(
                    "merge_strategy='highest_confidence' is not implemented. "
                    "It requires per-field confidence/provenance integration. "
                    "Use 'union', 'intersection', 'majority', or 'first_non_empty'."
                )

        return merged

    def _union_field_values(self, values: list[Any]) -> Any:
        """
        Union-merge values for a single field.

        - If all non-empty values are lists: extend/merge lists.
        - If mixed lists and scalars: extend lists and append scalars.
        - If all values are non-list scalars: keep the first non-empty value
          (do not wrap scalars in a list).
        """
        if not values:
            return None

        if all(isinstance(v, list) for v in values):
            merged_list: list[Any] = []
            for v in values:
                merged_list.extend(v)
            return merged_list

        if any(isinstance(v, list) for v in values):
            merged_list = []
            for v in values:
                if isinstance(v, list):
                    merged_list.extend(v)
                elif v is not None and v != "" and v != []:
                    merged_list.append(v)
            return merged_list

        # All non-list scalars: first non-empty (already filtered)
        return values[0]

    def _most_common_value(self, values: list[Any]) -> Any:
        """Get most common value from list"""
        if not values:
            return None

        # Count occurrences
        value_counts: dict[str, int] = {}
        for value in values:
            # Convert to string for comparison
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1

        # Find most common
        most_common_str = max(value_counts, key=value_counts.get)  # type: ignore[arg-type]

        # Return original value (not string)
        for value in values:
            if str(value) == most_common_str:
                return value

        return values[0]

    def _aggregate_usage(self, usage_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate usage from multiple passes.

        Normalizes both OpenAI-style (``prompt_tokens`` / ``completion_tokens``)
        and generic (``input_tokens`` / ``output_tokens``) keys so either form
        of per-pass usage is counted and both key styles appear in the total.
        """
        total: dict[str, Any] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for usage in usage_list:
            if not usage:
                continue

            total["requests"] += int(usage.get("requests") or 0)

            input_tokens = usage.get("input_tokens")
            if input_tokens is None:
                input_tokens = usage.get("prompt_tokens")
            input_tokens = int(input_tokens or 0)

            output_tokens = usage.get("output_tokens")
            if output_tokens is None:
                output_tokens = usage.get("completion_tokens")
            output_tokens = int(output_tokens or 0)

            total["input_tokens"] += input_tokens
            total["output_tokens"] += output_tokens
            total["prompt_tokens"] += input_tokens
            total["completion_tokens"] += output_tokens

            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            total["total_tokens"] += int(total_tokens or 0)

        return total


class MultiPassExtractionError(Exception):
    """Raised when multi-pass extraction fails"""

    def __init__(
        self,
        message: str,
        pass_results: list[PassResult],
        failed_passes: int,
        errors: list[str] | None = None,
    ):
        super().__init__(message)
        self.pass_results = pass_results
        self.failed_passes = failed_passes
        self.failed_count = failed_passes  # alias used by some callers/tests
        self.errors = errors or []

    def __str__(self):
        base = super().__str__()
        return (
            f"{base}\n"
            f"Successful passes: {len(self.pass_results)}\n"
            f"Failed passes: {self.failed_passes}"
        )
