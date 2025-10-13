from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from jsonschema import Draft202012Validator, ValidationError
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, ModelRetry
try:
    from pydantic_ai.run import AgentRunResult
except ImportError:  # compatibility with older pydantic-ai versions that re-exported from result
    from pydantic_ai.result import AgentRunResult
from pydantic_ai.usage import RunUsage

from .schema import build_output_type, is_pydantic_model, cast_to_dict_from_pydantic, JsonSchema, PydModelType
from .prompts import build_examples_block, combine_system_prompt, build_improvement_system_prompt
from .files import prepare_parts, flatten_for_agent
from .pricing import estimate_cost_usd, parse_pricing_json
from .config import RuntimeConfig
import structlog

log = structlog.get_logger(__name__)

def _prune_optional_empty_values(value: Any, schema: dict[str, Any]) -> Any:
    """Return a copy of value with optional properties that are empty string or null removed.

    This helps validate structure while tolerating optional empty values. It handles
    common 'object' and 'array' shapes; complex combinators are not specially processed.
    """
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(value, dict) and schema_type == "object":
        props: dict[str, Any] = (schema.get("properties") or {}) if isinstance(schema.get("properties"), dict) else {}
        required = set(schema.get("required", []) or [])
        out: dict[str, Any] = {}
        for key, val in value.items():
            prop_schema = props.get(key, {})
            if key not in required and (val is None or (isinstance(val, str) and val == "")):
                # Skip optional empty values during validation
                continue
            out[key] = _prune_optional_empty_values(val, prop_schema)
        return out
    if isinstance(value, list) and schema_type == "array":
        item_schema: dict[str, Any] = (schema.get("items") or {}) if isinstance(schema.get("items"), dict) else {}
        return [_prune_optional_empty_values(v, item_schema) for v in value]
    return value

def _collect_required_empty_errors(value: Any, schema: dict[str, Any], path: list[str] | None = None) -> list[str]:
    """Collect dot-paths for required fields that are missing or empty (null/"")."""
    if path is None:
        path = []
    errors: list[str] = []
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if schema_type == "object" and isinstance(value, dict):
        props: dict[str, Any] = (schema.get("properties") or {}) if isinstance(schema.get("properties"), dict) else {}
        required = list(schema.get("required", []) or [])
        for key in required:
            if key not in value:
                errors.append(".".join(path + [key]))
            else:
                v = value[key]
                if v is None or (isinstance(v, str) and v == ""):
                    errors.append(".".join(path + [key]))
        for key, v in value.items():
            child_schema = props.get(key, {})
            errors.extend(_collect_required_empty_errors(v, child_schema, path + [key]))
    elif schema_type == "array" and isinstance(value, list):
        item_schema: dict[str, Any] = (schema.get("items") or {}) if isinstance(schema.get("items"), dict) else {}
        for idx, item in enumerate(value):
            errors.extend(_collect_required_empty_errors(item, item_schema, path + [str(idx)]))
    return errors

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

        # Enforce: required fields must exist and be non-empty (not null/"")
        required_empty = _collect_required_empty_errors(output, schema)  # type: ignore[arg-type]
        if required_empty:
            if rounds >= max_validation_rounds:
                return output
            missing_list = ", ".join(required_empty)
            setattr(agent, "_validation_rounds", rounds + 1)  # type: ignore[attr-defined]
            raise ModelRetry(
                f"Required fields missing or empty: {missing_list}. Please fill all required fields with valid values."
            )

        # Validate structure/types with tolerance for optional empty values
        pruned = _prune_optional_empty_values(output, schema)  # type: ignore[arg-type]
        try:
            validator.validate(pruned)
            return output
        except ValidationError as e:
            if rounds >= max_validation_rounds:
                return output
            msg = (
                f"Schema validation failed: {e.message}. Please try again and produce JSON that strictly matches the schema."
            )
            setattr(agent, "_validation_rounds", rounds + 1)  # type: ignore[attr-defined]
            raise ModelRetry(msg)

async def _run_agent_once(
    agent: Agent,
    model_name: str,
    parts: Sequence[str | Any],
    *,
    timeout_s: float,
) -> AgentRunResult[Any]:
    # Pydantic AI supports run() (async) and run_sync(); we prefer async for batch/concurrency.
    return await asyncio.wait_for(agent.run(parts), timeout=timeout_s)

async def _run_agent_with_retries(
    agent: Agent,
    model_name: str,
    parts: Sequence[str | Any],
    *,
    timeout_s: float,
    max_attempts: int,
) -> AgentRunResult[Any]:
    """Run the agent with retry/backoff using runtime-configured attempts.

    Retries on provider/model errors and timeouts from asyncio.wait_for.
    """
    retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (ModelHTTPError, asyncio.TimeoutError, TimeoutError, UnexpectedModelBehavior)
        ),
    )
    async for attempt in retrying:
        with attempt:
            return await _run_agent_once(
                agent,
                model_name=model_name,
                parts=parts,
                timeout_s=timeout_s,
            )

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
        _attach_jsonschema_validator(
            agent,
            schema_or_model,  # small-file mode: enforce required fields
            max_validation_rounds=config.max_validation_rounds,
        )

    # Build message content parts from files
    prepared = prepare_parts(files)
    content_parts = flatten_for_agent(prepared)

    # Execute with retry wrapper
    result = await _run_agent_with_retries(
        agent,
        model_name=config.model,
        parts=content_parts,
        timeout_s=config.per_call_timeout_secs,
        max_attempts=config.max_run_retries,
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
        # Output is dict[str, Any]; validate strictly against the provided schema,
        # but tolerate optional fields being empty (null or ""). Still flag required empty.
        try:
            required_empty = _collect_required_empty_errors(data_out, schema_or_model)  # type: ignore[arg-type]
            if required_empty:
                warnings.append(
                    "final_validation_error: required fields missing or empty: " + ", ".join(required_empty)
                )
            else:
                pruned = _prune_optional_empty_values(data_out, schema_or_model)  # type: ignore[arg-type]
                Draft202012Validator(schema_or_model).validate(pruned)
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

    if report.warnings:
        log.warning(
            "nextract_run_completed_with_warnings",
            model=config.model,
            files=report.files,
            warnings=report.warnings,
        )
    else:
        log.debug(
            "nextract_run_complete",
            model=config.model,
            files=report.files,
            usage=report.usage,
            cost_estimate_usd=report.cost_estimate_usd,
        )

    return data_out, report


async def run_improvement_async(
    *,
    config: RuntimeConfig,
    current_schema: JsonSchema,
    user_prompt: Optional[str],
    batch_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the model to suggest an improved schema and user prompt using batch outputs."""
    improvement_schema: JsonSchema = {
        "title": "ImprovementResult",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "improved_schema": {
                "type": "object",
                "additionalProperties": True,
                "description": "A full JSON Schema (Draft 2020-12) for the desired output."
            },
            "improved_user_prompt": {
                "type": "string",
                "description": "Rewritten user prompt to guide extraction more accurately."
            }
        },
        "required": ["improved_schema", "improved_user_prompt"]
    }

    sys_prompt = build_improvement_system_prompt(current_schema, user_prompt)

    from pydantic_ai import Agent, StructuredDict  # local import to mirror other paths

    agent: Agent = Agent(
        config.model,
        output_type=StructuredDict(improvement_schema, name="ImprovementResult"),
        system_prompt=sys_prompt,
    )

    parts = [
        "BATCH RESULTS DATA (JSON array of per-item 'data'):",
        json.dumps(batch_results, ensure_ascii=False),
    ]

    result = await _run_agent_once(
        agent,
        model_name=config.model,
        parts=parts,
        timeout_s=config.per_call_timeout_secs,
    )

    out = result.output
    if not isinstance(out, dict):
        raise UnexpectedModelBehavior("Improvement output was not a dict")

    log.debug("nextract_improvement_complete", model=config.model)
    return out
