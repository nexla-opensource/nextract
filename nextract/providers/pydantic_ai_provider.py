from __future__ import annotations

import base64
from typing import Any

import structlog
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from pydantic_ai import Agent, BinaryContent, ModelSettings, StructuredDict, UsageLimits, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from nextract.core import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from nextract.core.exceptions import ProviderAuthError, ProviderRequestError
from nextract.core.model_capabilities import get_model_capability
from nextract.schema import prepare_output_schema

from nextract.providers.ocr_utils import infer_image_media_type

log = structlog.get_logger(__name__)

# Maps nextract provider names to their correct Pydantic AI model ID prefixes.
# See: https://pydantic.dev/docs/ai/models/overview
PROVIDER_PREFIX_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-gla",
    "google-vertex": "google-vertex",
    "azure": "azure",
    "cohere": "cohere",
    "aws": "bedrock",
    "bedrock": "bedrock",
    "local": "ollama",
}

# Maps nextract provider names to required environment variables.
# Providers requiring ALL listed env vars use all(); others use any().
PROVIDER_REQUIRED_ENV: dict[str, dict[str, list[str]]] = {
    "openai": {"any": ["OPENAI_API_KEY"]},
    "anthropic": {"any": ["ANTHROPIC_API_KEY"]},
    "google": {"any": ["GOOGLE_API_KEY"]},
    "google-vertex": {"all": ["GOOGLE_API_KEY"]},
    "azure": {"all": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]},
    "cohere": {"any": ["CO_API_KEY"]},
    "aws": {"all": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]},
    "bedrock": {"all": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]},
    "local": {"any": []},
}

# OCR providers that bypass the LLM model capability table
OCR_PROVIDERS = {"tesseract", "easyocr", "paddleocr", "textract"}


class PydanticAIProvider(BaseProvider):
    """Provider implementation backed by pydantic-ai Agent."""

    def __init__(self) -> None:
        self.name: str = "pydantic_ai"
        self.config: ProviderConfig | None = None
        self._model_instance: Any | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name
        self.config.validate()

    def supports_vision(self) -> bool:
        if not self.config:
            return False
        if self.config.name in OCR_PROVIDERS:
            return True
        return get_model_capability(
            model=self.config.model,
            capability="vision",
            default=False,
            provider=self.config.name,
        )

    def supports_structured_output(self) -> bool:
        if not self.config:
            return True
        if self.config.name in OCR_PROVIDERS:
            return False
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

    def get_required_env_vars(self) -> list[str]:
        """Return required environment variable names for this provider."""
        env_spec = PROVIDER_REQUIRED_ENV.get(self.name, {})
        all_vars = env_spec.get("all", []) + env_spec.get("any", [])
        return list(set(all_vars))

    def build_model(self) -> Any:
        """Build a Pydantic AI Model instance with credentials wired from config.

        Subclasses should override this to construct provider-specific Model
        objects. The default implementation constructs a model via string ID,
        which relies on environment variables for authentication.
        """
        from pydantic_ai.models import infer_model

        model_id = self._resolve_model_id()
        model_settings = self._build_model_settings()

        try:
            model = infer_model(model_id)
            if model_settings:
                model.settings = model_settings
            return model
        except Exception as exc:
            raise ProviderAuthError(
                f"Failed to construct model '{model_id}' for provider '{self.name}'. "
                f"Ensure required credentials are set: {self.get_required_env_vars()}. "
                f"Error: {exc}",
                provider=self.name,
                model=self.config.model if self.config else "",
            ) from exc

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

        model_settings = self._build_model_settings()

        # Build or reuse model instance with credentials
        if self._model_instance is None:
            self._model_instance = self.build_model()

        agent = Agent(
            self._model_instance,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=2,
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

    def _resolve_model_id(self) -> str:
        """Resolve the full Pydantic AI model ID from provider config.

        Uses PROVIDER_PREFIX_MAP to map nextract provider names to their
        correct Pydantic AI prefixes (e.g., 'google' -> 'google-gla',
        'aws' -> 'bedrock', 'local' -> 'ollama').

        Supports extra_params.backend to override the prefix
        (e.g., 'google-vertex' for Vertex AI).
        """
        if not self.config:
            raise ValueError("Provider not initialized")

        # Check if model already starts with a known provider prefix
        # (e.g., "openai:gpt-4o"). But be careful: some model names
        # contain colons naturally (e.g., "anthropic.claude-3:0"),
        # so we check against known prefixes specifically.
        if ":" in self.config.model:
            prefix = self.config.model.split(":", 1)[0]
            if prefix in PROVIDER_PREFIX_MAP.values() or prefix in PROVIDER_PREFIX_MAP:
                return self.config.model

        # Allow extra_params.backend to override the prefix
        prefix = PROVIDER_PREFIX_MAP.get(self.config.name, self.config.name)
        if self.config.extra_params:
            backend = self.config.extra_params.get("backend")
            if backend:
                prefix = PROVIDER_PREFIX_MAP.get(backend, backend)

        return f"{prefix}:{self.config.model}"

    def _model_id(self) -> str:
        """Legacy method for backward compatibility."""
        return self._resolve_model_id()

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
                    parts.append(BinaryContent(data=data, media_type=infer_image_media_type(data)))
                except (ValueError, Exception) as exc:  # noqa: BLE001
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
                        out.append(BinaryContent(data=data, media_type=infer_image_media_type(data)))
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
                except ModelHTTPError as exc:
                    status = getattr(exc, "status_code", None) or getattr(exc, "status_code", None)
                    if status in (401, 403):
                        raise ProviderAuthError(
                            f"Authentication failed for provider '{self.name}': {exc}",
                            provider=self.name,
                            model=self.config.model if self.config else "",
                            status_code=status,
                        ) from exc
                    raise ProviderRequestError(
                        f"Provider request failed for '{self.name}': {exc}",
                        provider=self.name,
                        model=self.config.model if self.config else "",
                        status_code=status,
                        retryable=status is None or status >= 500,
                    ) from exc
                except (TimeoutError, ConnectionError, OSError) as exc:
                    raise ProviderRequestError(
                        f"Provider request failed for '{self.name}': {exc}",
                        provider=self.name,
                        model=self.config.model if self.config else "",
                        retryable=True,
                    ) from exc
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
        request_limit = max_attempts * 2

        max_total_tokens = None
        if self.config and self.config.extra_params:
            max_total_tokens = self.config.extra_params.get("max_total_tokens")

        return UsageLimits(
            request_limit=request_limit,
            total_tokens_limit=max_total_tokens,
        )
