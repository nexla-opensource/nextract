from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Union, Type

from pydantic import BaseModel

from .config import load_runtime_config, RuntimeConfig
from .logging import setup_logging
from .agent_runner import run_extraction_async, ExtractionReport
from .schema import JsonSchema, PydModelType, is_pydantic_model

def extract(
    files: Sequence[str],
    *,
    schema_or_model: Union[JsonSchema, Type[BaseModel]],
    user_prompt: Optional[str] = None,
    examples: Optional[Sequence[dict | tuple[Optional[str], dict]]] = None,
    include_extra: bool = False,
    return_pydantic: bool = False,
    config: Optional[RuntimeConfig] = None,
    setup_logs: bool = True,
) -> dict[str, Any]:
    """Synchronous convenience wrapper.
    Returns a dict with keys: data, report (usage+cost), model, files.

    - If a Pydantic model class is passed and return_pydantic=True, `data` will be that model instance;
      otherwise `data` is a dict (Pydantic models are .model_dump()).
    """
    if setup_logs:
        setup_logging()

    cfg = config or load_runtime_config()

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
    return_pydantic: bool = False,
    max_concurrency: Optional[int] = None,
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
    if max_concurrency is not None:
        cfg = RuntimeConfig(
            model=cfg.model,
            max_concurrency=int(max_concurrency),
            max_run_retries=cfg.max_run_retries,
            per_call_timeout_secs=cfg.per_call_timeout_secs,
            pricing_json=cfg.pricing_json,
        )

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

    return asyncio.run(runner())
