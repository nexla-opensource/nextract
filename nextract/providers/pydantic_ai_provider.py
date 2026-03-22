from __future__ import annotations

import base64
from typing import Any

import structlog
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from pydantic_ai import Agent, BinaryContent, ModelSettings, StructuredDict, UsageLimits, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.core.model_capabilities import get_model_capability
from nextract.schema import prepare_output_schema

log = structlog.get_logger(__name__)


class PydanticAIProvider(BaseProvider):
    """Provider implementation backed by pydantic-ai Agent."""

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.config.validate()

    def supports_vision(self) -> bool:
        if not self.config:
            return False
        return get_model_capability(
            model=self.config.model,
            capability="vision",
            default=False,
            provider=self.config.name,
        )

    def supports_structured_output(self) -> bool:
        if not self.config:
            return True
        return get_model_capability(
            model=self.config.model,
            capability="structured_output",
            default=True,
            provider=self.config.name,
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "vision": self.supports_vision(),
            "structured_output": self.supports_structured_output(),
            "streaming": False,
            "max_tokens": self.config.max_tokens if self.config else None,
        }

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not self.config:
            raise ValueError("Provider not initialized")

        system_prompt, parts = self._build_parts(request)
        include_extra = bool(request.options.get("include_extra"))
        unwrap_key: str | None = None
        if request.schema:
            prepared_schema, unwrap_key = prepare_output_schema(
                request.schema,
                include_extra=include_extra,
            )
            output_type = StructuredDict(prepared_schema, name=prepared_schema.get("title", "Output"))
        else:
            output_type = str

        # Build ModelSettings from config for timeout, temperature, max_tokens
        model_settings = self._build_model_settings()

        agent = Agent(
            self._model_id(),
            output_type=output_type,
            system_prompt=system_prompt,
            retries=2,  # Built-in output validation retries
            model_settings=model_settings,
        )

        result = self._run_with_retries_sync(
            agent,
            parts,
            max_attempts=self.config.max_retries,
        )

        usage = result.usage()
        output = result.output

        if isinstance(output, dict):
            structured_output = output
            text_output = ""
        elif hasattr(output, "model_dump"):
            structured_output = output.model_dump()
            text_output = ""
        elif isinstance(output, str):
            structured_output = None
            text_output = output
        else:
            structured_output = None
            text_output = str(output) if output is not None else ""

        if unwrap_key and isinstance(structured_output, dict) and unwrap_key in structured_output:
            structured_output = structured_output[unwrap_key]

        return ProviderResponse(
            text=text_output,
            structured_output=structured_output,
            usage={
                "requests": usage.requests,
                "tool_calls": usage.tool_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "details": usage.details,
            },
            raw=output,
        )

    def _model_id(self) -> str:
        if not self.config:
            raise ValueError("Provider not initialized")
        if ":" in self.config.model:
            return self.config.model
        return f"{self.config.name}:{self.config.model}"

    def _build_parts(self, request: ProviderRequest) -> tuple[str, list[str | BinaryContent]]:
        system_prompt = ""
        parts: list[str | BinaryContent] = []

        for message in request.messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system" and isinstance(content, str):
                system_prompt = content
                continue

            for part in self._normalize_content(content):
                parts.append(part)

        if request.images:
            for image_b64 in request.images:
                try:
                    data = base64.b64decode(image_b64)
                    parts.append(BinaryContent(data=data, media_type="image/png"))
                except Exception as exc:  # noqa: BLE001
                    log.warning("image_decode_failed", error=str(exc))

        binary_parts = request.options.get("binary_parts")
        if isinstance(binary_parts, list):
            for part in binary_parts:
                if isinstance(part, BinaryContent):
                    parts.append(part)

        if not parts:
            parts.append("")

        return system_prompt, parts

    def _normalize_content(self, content: Any) -> list[str | BinaryContent]:
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            out: list[str | BinaryContent] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    out.append(item["text"])
                if item.get("type") == "image" and isinstance(item.get("image"), str):
                    try:
                        data = base64.b64decode(item["image"])
                        out.append(BinaryContent(data=data, media_type="image/png"))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("image_decode_failed", error=str(exc))
            return out
        return []

    def _run_with_retries_sync(
        self,
        agent: Agent,
        parts: list[str | BinaryContent],
        max_attempts: int,
    ):
        """Run agent with sync retries using pydantic-ai's run_sync method."""
        # Build UsageLimits from config to prevent runaway token usage
        usage_limits = self._build_usage_limits(max_attempts)

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(multiplier=1, max=10),
            retry=retry_if_exception_type(
                (ModelHTTPError, TimeoutError, ConnectionError, OSError, UnexpectedModelBehavior)
            ),
        )

        for attempt in retrying:
            with attempt:
                try:
                    with capture_run_messages() as messages:
                        return agent.run_sync(parts, usage_limits=usage_limits)
                except Exception as exc:
                    log.error(
                        "provider_run_failed",
                        error=str(exc),
                        messages_count=len(messages),
                        attempt=attempt.retry_state.attempt_number,
                    )
                    raise

    def _build_model_settings(self) -> ModelSettings | None:
        """Build ModelSettings from provider config."""
        if not self.config:
            return None

        return ModelSettings(
            timeout=self.config.timeout if self.config.timeout else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens if self.config.max_tokens else None,
        )

    def _build_usage_limits(self, max_attempts: int) -> UsageLimits:
        """Build UsageLimits from config to prevent runaway usage."""
        # Allow retries plus some buffer for the request limit
        request_limit = max_attempts * 2

        # Get max total tokens from extra_params if specified
        max_total_tokens = None
        if self.config and self.config.extra_params:
            max_total_tokens = self.config.extra_params.get("max_total_tokens")

        return UsageLimits(
            request_limit=request_limit,
            total_tokens_limit=max_total_tokens,
        )
