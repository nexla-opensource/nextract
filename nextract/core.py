from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Union, Type

from pydantic import BaseModel
import structlog

from .config import load_runtime_config, RuntimeConfig
from .logging import setup_logging
from .agent_runner import run_extraction_async, run_improvement_async
from .schema import JsonSchema, cast_to_pydantic, is_pydantic_model, to_json_schema
from .chunking import TokenEstimator, TokenEstimate, DocumentChunker, ChunkExtractor
from .adaptive_extraction import extract_with_adaptive_retry

log = structlog.get_logger(__name__)

def extract(
    files: Sequence[str],
    *,
    schema_or_model: Union[JsonSchema, Type[BaseModel]],
    user_prompt: Optional[str] = None,
    examples: Optional[Sequence[dict | tuple[Optional[str], dict]]] = None,
    include_extra: bool = False,
    return_pydantic: bool = False,
    model: Optional[str] = None,
    config: Optional[RuntimeConfig] = None,
    setup_logs: bool = True,
    enable_chunking: bool = True,
    max_workers: Optional[int] = None,
    enable_multipass: Optional[bool] = None,
    num_passes: Optional[int] = None,
    multipass_merge_strategy: Optional[str] = None,
    enable_provenance: Optional[bool] = None,
    enable_adaptive_extraction: bool = True,
    enable_completeness_retry: bool = False,
    completeness_threshold: float = 0.7,
)-> dict[str, Any]:
    """
    Extract structured data from documents using AI.

    Returns a dict with keys: data, report (model, files, usage, cost_estimate_usd, warnings).

    Args:
        files: List of file paths to extract from
        schema_or_model: JSON schema dict or Pydantic model class
        user_prompt: Optional custom prompt
        examples: Optional examples for few-shot learning
        include_extra: Include extra fields not in schema
        return_pydantic: Return Pydantic model instance (if schema_or_model is Pydantic)
        model: Override model (e.g., "openai:gpt-4o")
        config: Custom RuntimeConfig
        setup_logs: Setup structured logging
        enable_chunking: Automatically chunk large documents (default: True)
        max_workers: Number of parallel workers for chunk processing (default: from config or 10)
        enable_multipass: Run extraction multiple times and merge (default: from config or False)
        num_passes: Number of extraction passes for multipass (default: from config or 3)
        multipass_merge_strategy: Merge strategy for multipass ("union", "intersection", "majority", default: "union")
        enable_provenance: Track where each field came from (default: from config or False)
        enable_adaptive_extraction: Use intelligent two-pass extraction to improve field completeness (default: True)
        enable_completeness_retry: Enable completeness-based retry for array schemas (default: False)
        completeness_threshold: Minimum completeness confidence to accept without retry (default: 0.7)

    Returns:
        Dict with "data" (extracted data) and "report" (metadata, usage, cost, warnings)

    Example:
        from nextract import extract
        from pydantic import BaseModel

        class Invoice(BaseModel):
            invoice_number: str
            total_amount: float

        result = extract(
            files=["invoice.pdf"],
            schema_or_model=Invoice,
            max_workers=10,
            enable_provenance=True
        )

        print(result["data"]["invoice_number"])
        print(result["report"]["usage"]["field_provenance"])
    """
    if setup_logs:
        setup_logging()

    cfg = config or load_runtime_config()

    # Apply parameter overrides
    if any([model, max_workers, enable_multipass, num_passes, multipass_merge_strategy, enable_provenance]):
        cfg = RuntimeConfig(
            model=model or cfg.model,
            max_concurrency=cfg.max_concurrency,
            max_run_retries=cfg.max_run_retries,
            per_call_timeout_secs=cfg.per_call_timeout_secs,
            pricing_json=cfg.pricing_json,
            max_validation_rounds=cfg.max_validation_rounds,
            max_workers=max_workers if max_workers is not None else cfg.max_workers,
            enable_multipass=enable_multipass if enable_multipass is not None else cfg.enable_multipass,
            num_passes=num_passes if num_passes is not None else cfg.num_passes,
            multipass_merge_strategy=multipass_merge_strategy or cfg.multipass_merge_strategy,
            enable_provenance=enable_provenance if enable_provenance is not None else cfg.enable_provenance,
        )

    # Convert to JSON schema for token estimation
    schema = to_json_schema(schema_or_model)

    # NEW: Estimate tokens and check if chunking needed
    if enable_chunking:
        estimator = TokenEstimator(cfg.model)
        estimate = estimator.estimate_tokens(
            files=list(files),
            schema=schema,
            user_prompt=user_prompt,
            examples=list(examples) if examples else None
        )

        log.info(
            "token_estimation",
            file_tokens=estimate.file_tokens,
            schema_tokens=estimate.schema_tokens,
            prompt_tokens=estimate.prompt_tokens,
            total_tokens=estimate.total_tokens,
            model_limit=estimate.model_limit,
            utilization=f"{estimate.utilization:.1%}",
            needs_chunking=estimate.needs_chunking,
            recommended_chunks=estimate.recommended_chunks
        )

        # Force chunking for array schemas to avoid schema validation issues
        # Array schemas need to be wrapped in objects, which only happens in chunked extraction
        is_array_schema = schema.get("type") == "array"
        if is_array_schema and not estimate.needs_chunking:
            log.info(
                "forcing_chunking_for_array_schema",
                reason="Array schemas require chunking to wrap in object for extraction"
            )
            estimate = TokenEstimate(
                file_tokens=estimate.file_tokens,
                schema_tokens=estimate.schema_tokens,
                prompt_tokens=estimate.prompt_tokens,
                total_tokens=estimate.total_tokens,
                model_limit=estimate.model_limit,
                utilization=estimate.utilization,
                needs_chunking=True,  # Force chunking
                recommended_chunks=max(2, estimate.recommended_chunks)  # At least 2 chunks
            )

        # NEW: Check if field chunking is beneficial (before document chunking)
        # Only use field chunking if adaptive extraction is NOT enabled
        if not enable_adaptive_extraction:
            from .field_chunking import should_chunk_fields, extract_with_field_chunking

            # Calculate approximate document size
            doc_size = sum(estimate.file_tokens for _ in files) * 4  # Rough char estimate

            if should_chunk_fields(schema, doc_size) and not estimate.needs_chunking:
                # Use field chunking instead of document chunking
                log.info(
                    "using_field_chunking",
                    num_fields=len(schema.get("properties", {})),
                    reason="large_schema_detected"
                )

                data, report_dict = asyncio.run(
                    extract_with_field_chunking(
                        files=list(files),
                        schema=schema,
                        config=cfg,
                        user_prompt=user_prompt,
                        examples=list(examples) if examples else None,
                        include_extra=include_extra
                    )
                )

                out: dict[str, Any] = {
                    "data": data,
                    "report": report_dict
                }
                return out

        # If chunking needed, use chunk-based extraction
        if estimate.needs_chunking:
            log.warning(
                "document_too_large_chunking_enabled",
                total_tokens=estimate.total_tokens,
                model_limit=estimate.model_limit,
                effective_limit=int(estimate.model_limit * 0.7),
                recommended_chunks=estimate.recommended_chunks,
                message=f"Document exceeds context window ({estimate.total_tokens} tokens > {int(estimate.model_limit * 0.7)} limit). Automatically chunking into {estimate.recommended_chunks} pieces."
            )

            # Chunk documents
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(
                file_paths=list(files),
                num_chunks=estimate.recommended_chunks,
                strategy="auto"
            )

            log.info(
                "documents_chunked",
                num_chunks=len(chunks),
                chunk_types=[c.chunk_type for c in chunks]
            )

            # Extract from chunks with parallel processing and provenance
            chunk_extractor = ChunkExtractor(
                max_workers=cfg.max_workers,
                enable_provenance=cfg.enable_provenance,
                enable_completeness_retry=enable_completeness_retry,
                completeness_threshold=completeness_threshold
            )
            data, report_dict = asyncio.run(
                chunk_extractor.extract_from_chunks(
                    chunks=chunks,
                    schema=schema,
                    config=cfg,
                    user_prompt=user_prompt,
                    examples=list(examples) if examples else None,
                    include_extra=include_extra
                )
            )

            # Return chunked extraction result
            out: dict[str, Any] = {
                "data": data,
                "report": report_dict
            }
            return out

    # Original extraction logic (no chunking needed)

    # NEW: Check if adaptive extraction is enabled
    if enable_adaptive_extraction and schema.get("type") == "object":
        log.info(
            "adaptive_extraction_enabled",
            num_fields=len(schema.get("properties", {}))
        )

        data, report_dict = asyncio.run(
            extract_with_adaptive_retry(
                files=list(files),
                schema=schema,
                config=cfg,
                user_prompt=user_prompt,
                examples=list(examples) if examples else None,
                include_extra=include_extra
            )
        )

        out: dict[str, Any] = {
            "data": data,
            "report": report_dict
        }
        if return_pydantic and is_pydantic_model(schema_or_model):
            data = cast_to_pydantic(schema_or_model, data)  # type: ignore[arg-type]
            out["data"] = data
        return out

    # Check if multi-pass extraction is enabled
    if cfg.enable_multipass:
        from .multipass import MultiPassExtractor

        log.info(
            "multipass_extraction_enabled",
            num_passes=cfg.num_passes,
            merge_strategy=cfg.multipass_merge_strategy
        )

        # Create multi-pass extractor
        multipass_extractor = MultiPassExtractor(
            num_passes=cfg.num_passes,
            fail_threshold=cfg.num_passes - 1  # At least 1 must succeed
        )

        # Define extraction function for multi-pass
        async def extraction_fn(**kwargs):
            result = await run_extraction_async(
                config=cfg,
                files=list(files),
                schema_or_model=schema_or_model,
                user_prompt=user_prompt,
                examples=examples,
                include_extra=include_extra,
                return_pydantic=False,  # Always use dict for merging
            )

            # Convert to (data, report) tuple
            return result[0], {
                "usage": result[1].usage,
                "cost_estimate_usd": result[1].cost_estimate_usd,
                "warnings": []
            }

        # Run multi-pass extraction
        multipass_result = asyncio.run(
            multipass_extractor.extract_multipass(
                extraction_fn=extraction_fn,
                schema=schema,
                merge_strategy=cfg.multipass_merge_strategy
            )
        )

        # Extract data and create report
        data = multipass_result.merged_data
        report = type('Report', (), {
            'model': cfg.model,
            'files': list(files),
            'usage': multipass_result.total_usage,
            'cost_estimate_usd': multipass_result.total_cost,
            'warnings': [
                f"Multi-pass extraction: {multipass_result.successful_passes}/{multipass_result.total_passes} passes succeeded",
                f"Merge strategy: {multipass_result.merge_strategy}"
            ]
        })()
    else:
        # Single-pass extraction
        data, report = asyncio.run(
            run_extraction_async(
                config=cfg,
                files=list(files),
                schema_or_model=schema_or_model,
                user_prompt=user_prompt,
                examples=examples,
                include_extra=include_extra,
                return_pydantic=return_pydantic,
            )
        )

    # Ensure dict return by default
    if is_pydantic_model(schema_or_model) and not return_pydantic:
        # (should already be dict from runner; be defensive)
        try:
            from .schema import cast_to_dict_from_pydantic
            if hasattr(data, "model_dump"):
                data = cast_to_dict_from_pydantic(data)
        except Exception:
            pass

    out: dict[str, Any] = {
        "data": data,
        "report": {
            "model": report.model,
            "files": report.files,
            "usage": report.usage,
            "cost_estimate_usd": report.cost_estimate_usd,
            "warnings": report.warnings,
        },
    }
    return out

async def _extract_one_for_batch(
    files: Sequence[str],
    *,
    schema_or_model: Union[JsonSchema, Type[BaseModel]],
    user_prompt: Optional[str],
    examples: Optional[Sequence[dict | tuple[Optional[str], dict]]],
    include_extra: bool,
    return_pydantic: bool,
    config: RuntimeConfig,
) -> tuple[str, dict[str, Any]]:
    """Return (first_file_key, result_dict)."""
    data, report = await run_extraction_async(
        config=config,
        files=list(files),
        schema_or_model=schema_or_model,
        user_prompt=user_prompt,
        examples=examples,
        include_extra=include_extra,
        return_pydantic=return_pydantic,
    )
    # Standardize output payload
    first_key = files[0] if files else "batch_item"
    result = {
        "data": data if return_pydantic else data,
        "report": {
            "model": report.model,
            "files": report.files,
            "usage": report.usage,
            "cost_estimate_usd": report.cost_estimate_usd,
            "warnings": report.warnings,
        },
    }
    return first_key, result

def batch_extract(
    batch: Sequence[Sequence[str] | str],
    *,
    schema_or_model: Union[JsonSchema, Type[BaseModel]],
    user_prompt: Optional[str] = None,
    examples: Optional[Sequence[dict | tuple[Optional[str], dict]]] = None,
    include_extra: bool = False,
    provide_improvements: bool = False,
    return_pydantic: bool = False,
    max_concurrency: Optional[int] = None,
    model: Optional[str] = None,
    config: Optional[RuntimeConfig] = None,
    setup_logs: bool = True,
) -> dict[str, Any]:
    """Process multiple filesets in parallel.
    `batch` may be:
        - a list of file paths (each entry is a single file), or
        - a list of list-of-files (a group processed together in one Agent run).
    Returns a dict keyed by the first file path in each item.
    """
    if setup_logs:
        setup_logging()
    cfg = config or load_runtime_config()
    # Apply explicit model override if provided
    if model is not None:
        cfg = RuntimeConfig(
            model=model,
            max_concurrency=cfg.max_concurrency,
            max_run_retries=cfg.max_run_retries,
            per_call_timeout_secs=cfg.per_call_timeout_secs,
            pricing_json=cfg.pricing_json,
            max_validation_rounds=cfg.max_validation_rounds,
        )
    # Apply explicit concurrency override if provided
    if max_concurrency is not None:
        cfg = RuntimeConfig(
            model=cfg.model,
            max_concurrency=int(max_concurrency),
            max_run_retries=cfg.max_run_retries,
            per_call_timeout_secs=cfg.per_call_timeout_secs,
            pricing_json=cfg.pricing_json,
            max_validation_rounds=cfg.max_validation_rounds,
        )
    # If improvements are requested, ensure extra collection is enabled
    if provide_improvements:
        include_extra = True

    async def runner() -> dict[str, Any]:
        sem = asyncio.Semaphore(cfg.max_concurrency)
        results: dict[str, Any] = {}

        async def run_one(group: Sequence[str] | str):
            async with sem:
                files = [group] if isinstance(group, str) else list(group)
                key, res = await _extract_one_for_batch(
                    files=files,
                    schema_or_model=schema_or_model,
                    user_prompt=user_prompt,
                    examples=examples,
                    include_extra=include_extra,
                    return_pydantic=return_pydantic,
                    config=cfg,
                )
                results[key] = res

        tasks = [asyncio.create_task(run_one(item)) for item in batch]
        await asyncio.gather(*tasks)
        return results

    results = asyncio.run(runner())

    if not provide_improvements:
        return results

    # Build improvement payload
    schema_json = to_json_schema(schema_or_model)
    batch_data: list[dict[str, Any]] = []
    for _, res in results.items():
        d = res.get("data")
        if hasattr(d, "model_dump"):
            try:
                d = d.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(d, dict):
            batch_data.append(d)
        else:
            batch_data.append({"value": d})

    improvements = asyncio.run(
        run_improvement_async(
            config=cfg,
            current_schema=schema_json,
            user_prompt=user_prompt,
            batch_results=batch_data,
        )
    )

    # Attach improvement outputs alongside results
    results_out: dict[str, Any] = {
        "results": results,
        "improved_schema": improvements.get("improved_schema"),
        "improved_user_prompt": improvements.get("improved_user_prompt"),
    }
    return results_out
