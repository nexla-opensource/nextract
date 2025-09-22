from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, Type

from jsonschema import Draft202012Validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, ModelRetry
from pydantic_ai.result import AgentRunResult
from pydantic_ai.usage import RunUsage

from .schema import build_output_type, is_pydantic_model, cast_to_pydantic, cast_to_dict_from_pydantic, JsonSchema, PydModelType
from .prompts import build_examples_block, combine_system_prompt
from .files import prepare_parts, flatten_for_agent
from .pricing import estimate_cost_usd, parse_pricing_json
from .config import RuntimeConfig
import structlog

log = structlog.get_logger(__name__)

@dataclass
class ExtractionMetrics:
    usage: RunUsage
    cost_estimate_usd: Optional[float]

@dataclass
class ExtractionReport:
    model: str
    files: list[str]
    usage: dict[str, Any]
    cost_estimate_usd: Optional[float] = None
    warnings: list[str] = None  # type: ignore[assignment]

def _attach_jsonschema_validator(agent: Agent, schema: JsonSchema, max_validation_rounds: int = 2) -> None:
    """Add an output validator that validates the dict against the user's JSON Schema and asks the model to retry on failure.
    Pydantic AI docs: 'Output validators' + ModelRetry.  """
    validator = Draft202012Validator(schema)

    # We embed simple state on the function object to cap retries
    agent._validation_rounds = 0  # type: ignore[attr-defined]

    @agent.output_validator  # type: ignore[misc]
    async def validate_output(output: dict[str, Any]) -> dict[str, Any]:
        # Limit the number of schema-enforced retry loops
        rounds = getattr(agent, "_validation_rounds", 0)  # type: ignore[attr-defined]
        try:
            validator.validate(output)
            return output
        except ValidationError as e:
            if rounds >= max_validation_rounds:
                # Give up; return as-is (caller will validate again)
                return output
            msg = f"Schema validation failed: {e.message}. " \
                  f"Please try again and produce JSON that *strictly* matches the schema."
            setattr(agent, "_validation_rounds", rounds + 1)  # type: ignore[attr-defined]
            raise ModelRetry(msg)

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((ModelHTTPError, TimeoutError, UnexpectedModelBehavior)),
)
async def _run_agent_once(
    agent: Agent,
    model_name: str,
    parts: Sequence[str | Any],
    *,
    timeout_s: float,
) -> AgentRunResult[Any]:
    # Pydantic AI supports run() (async) and run_sync(); we prefer async for batch/concurrency.
    return await asyncio.wait_for(agent.run(parts), timeout=timeout_s)

async def run_extraction_async(
    *,
    config: RuntimeConfig,
    files: Sequence[str],
    schema_or_model: Union[JsonSchema, PydModelType],
    user_prompt: Optional[str],
    examples: Optional[Sequence[dict[str, Any] | tuple[Optional[str], dict[str, Any]]]],
    include_extra: bool,
    return_pydantic: bool = False,
) -> tuple[Any, ExtractionReport]:
    """Run a single-file or multi-file extraction asynchronously.

    Returns (data, report). `data` is dict by default, unless `return_pydantic=True` and a Pydantic model type was passed.
    """
    examples_block = build_examples_block(examples)
    sys_prompt = combine_system_prompt(user_prompt, include_extra, examples_block)
    output_type = build_output_type(schema_or_model, include_extra)

    # Initialize Agent with model and desired output
    agent: Agent = Agent(config.model, output_type=output_type, system_prompt=sys_prompt)

    # For JSON Schema dicts, attach a jsonschema validator to drive retries until it matches.
    if not is_pydantic_model(schema_or_model):
        _attach_jsonschema_validator(agent, schema_or_model)  # small-file mode: enforce required fields

    # Build message content parts from files
    prepared = prepare_parts(files)
    content_parts = flatten_for_agent(prepared)

    # Execute with retry wrapper
    result = await _run_agent_once(
        agent,
        model_name=config.model,
        parts=content_parts,
        timeout_s=config.per_call_timeout_secs,
    )

    usage = result.usage()
    # Post-validate one last time for dict outputs (raise if still invalid)
    warnings: list[str] = []
    data_out: Any = result.output

    if is_pydantic_model(schema_or_model):
        # data_out is a Pydantic model instance already
        if return_pydantic:
            pass  # keep as is
        else:
            data_out = cast_to_dict_from_pydantic(data_out)
    else:
        # Output is dict[str, Any]; validate strictly against the provided schema
        try:
            Draft202012Validator(schema_or_model).validate(data_out)
        except ValidationError as e:
            warnings.append(f"final_validation_error: {e.message}")

    # Cost estimation
    pricing_map = parse_pricing_json(config.pricing_json)
    cost = estimate_cost_usd(usage, config.model, pricing_map)

    # Build report
    report = ExtractionReport(
        model=config.model,
        files=[str(f) for f in files],
        usage={
            "requests": usage.requests,
            "tool_calls": usage.tool_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "details": usage.details,
        },
        cost_estimate_usd=cost,
        warnings=warnings or [],
    )

    log.info(
        "nextract_run_complete",
        model=config.model,
        files=report.files,
        usage=report.usage,
        cost_estimate_usd=report.cost_estimate_usd,
        warnings=report.warnings,
    )

    return data_out, report
