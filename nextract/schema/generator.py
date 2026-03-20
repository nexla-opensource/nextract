from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from nextract.core import ProviderConfig, ProviderRequest
from nextract.ingest import load_documents
from nextract.parse import extract_text
from nextract.registry import ProviderRegistry

log = structlog.get_logger(__name__)


@dataclass
class SchemaSuggestion:
    """Schema improvement suggestion with optional impact estimate."""

    description: str
    impact: float | None = None


class SchemaGenerator:
    """Generate JSON Schemas from sample documents and prompts."""

    def __init__(self, provider: ProviderConfig) -> None:
        import nextract.providers  # noqa: F401

        provider_class = ProviderRegistry.get_instance().get(provider.name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider.name}")
        self.provider = provider_class()
        self.provider.initialize(provider)
        self.provider_config = provider

    def suggest_schema(
        self,
        sample_documents: list[str],
        prompt: str,
        examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        documents = load_documents(sample_documents)
        samples: list[str] = []
        for document in documents:
            text = extract_text(document)
            if text:
                samples.append(text)

        sample_payload = "\n\n".join(samples) if samples else ""
        schema_prompt = self._build_prompt(prompt, sample_payload, examples)

        response_schema = {
            "type": "object",
            "properties": {
                "schema": {"type": "object"},
                "notes": {"type": "string"},
            },
            "required": ["schema"],
        }

        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "You are a schema design assistant."},
                {"role": "user", "content": schema_prompt},
            ],
            schema=response_schema,
        )

        response = self.provider.generate(request)
        payload = response.structured_output or self._try_parse_json(response.text)

        if isinstance(payload, dict) and "schema" in payload:
            return payload["schema"]
        if isinstance(payload, dict):
            return payload
        raise ValueError("Schema suggestion did not return a JSON object")

    def save_schema(self, schema: dict[str, Any], path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.write_text(json.dumps(schema, ensure_ascii=False, indent=2))

    def _build_prompt(
        self,
        user_prompt: str,
        sample_payload: str,
        examples: list[dict[str, Any]] | None,
    ) -> str:
        prompt_parts = [
            "You are an expert in JSON Schema design.",
            "Generate a Draft 2020-12 JSON Schema that matches the requested extraction.",
            f"USER PROMPT:\n{user_prompt}",
        ]

        if examples:
            prompt_parts.append("EXAMPLES (JSON):")
            prompt_parts.append(json.dumps(examples, ensure_ascii=False))

        if sample_payload:
            prompt_parts.append("SAMPLE DOCUMENTS:")
            if len(sample_payload) > 15000:
                log.warning("schema_generator_sample_truncated", original_len=len(sample_payload), truncated_len=15000)
            prompt_parts.append(sample_payload[:15000])

        return "\n\n".join(prompt_parts)

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            log.warning("schema_generator_json_parse_failed")
            return None
