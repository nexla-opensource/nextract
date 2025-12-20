from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, Optional

import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.schema import build_output_type

log = structlog.get_logger(__name__)


class PydanticAIProvider(BaseProvider):
    """Provider implementation backed by pydantic-ai Agent."""

    def __init__(self) -> None:
        self.config: Optional[ProviderConfig] = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.config.validate()

    def supports_vision(self) -> bool:
        if not self.config:
            return False
        model = self.config.model.lower()
        if "vision" in model or "gpt-4o" in model or "gpt-4" in model:
            return True
        if self.config.name in {"openai", "anthropic", "google", "azure", "local"}:
            return True
        return False

    def supports_structured_output(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
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
        output_type = build_output_type(request.schema, include_extra=include_extra) if request.schema else str

        agent = Agent(
            self._model_id(),
            output_type=output_type,
            system_prompt=system_prompt,
        )

        result = asyncio.run(
            self._run_with_retries(
                agent,
                parts,
                timeout_s=self.config.timeout,
                max_attempts=self.config.max_retries,
            )
        )

        usage = result.usage()
        output = result.output

        structured_output = output if isinstance(output, dict) else None
        text_output = output if isinstance(output, str) else ""

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

    async def _run_with_retries(
        self,
        agent: Agent,
        parts: list[str | BinaryContent],
        timeout_s: float,
        max_attempts: int,
    ):
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
                return await asyncio.wait_for(agent.run(parts), timeout=timeout_s)
