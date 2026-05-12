from __future__ import annotations

import asyncio
import concurrent.futures
import json
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

import structlog

from nextract.core import (
    ChunkExtraction,
    ExtractionPlan,
    ExtractionResult,
    ExtractorConfig,
    ProviderConfig,
)
from nextract.core.exceptions import PipelineError, NextractError
from nextract.registry.bootstrap import ensure_plugins_loaded
from nextract.ingest.loaders import load_documents
from nextract.merge import merge_partial_outputs
from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry
from nextract.schema import SchemaSuggestion
from nextract.validate import PlanValidator

ensure_plugins_loaded()

log = structlog.get_logger(__name__)

# Known keys in extra_params that contain secrets and should be masked in repr/logs
_SECRET_EXTRA_PARAMS = {
    "aws_secret_key", "aws_session_token", "api_key", "secret_key",
    "password", "token", "credential",
}


@dataclass
class BatchExtractionResult:
    results: dict[str, ExtractionResult]
    suggestions: list[SchemaSuggestion] = field(default_factory=list)


class ExtractionPipeline:
    """Pipeline orchestrator for single-document extraction."""

    def __init__(self, plan: ExtractionPlan) -> None:
        self.plan = plan
        PlanValidator.raise_for_invalid(plan)

        self.extractor = self._build_extractor(plan.extractor)
        self.provider = self._build_provider(plan.extractor.provider)
        self.chunker = self._build_chunker(plan.chunker.name)

    def extract(
        self,
        document: str | list[str],
        schema: dict[str, Any],
        prompt: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
    ) -> ExtractionResult:
        documents = [document] if isinstance(document, str) else list(document)
        artifacts = load_documents(documents)

        from nextract.ingest import DocumentValidator

        doc_validator = DocumentValidator()
        for artifact in artifacts:
            validation = doc_validator.validate(artifact)
            if not validation.valid:
                raise PipelineError("; ".join(validation.errors))

        chunks: list[Any] = []
        for artifact in artifacts:
            chunks.extend(self.chunker.chunk(artifact, self.plan.chunker))

        if not chunks:
            log.warning("no_chunks_generated", files=documents)
            return ExtractionResult(data={}, metadata={"chunks": 0})

        prompt_text = prompt or "Extract the requested fields."
        if self.plan.num_passes > 1:
            return self._multi_pass_extract(
                chunks, schema, prompt_text, examples, include_extra,
            )

        merged_data, usage, extractor_name, provider_name = self._run_pass(
            chunks=chunks,
            schema=schema,
            prompt=prompt_text,
            examples=examples,
            include_extra=include_extra,
        )

        return self._build_result(
            merged_data, usage, extractor_name, provider_name, len(chunks), schema,
        )

    async def extract_async(
        self,
        document: str | list[str],
        schema: dict[str, Any],
        prompt: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
    ) -> ExtractionResult:
        """True async extraction.

        Runs the extraction in a thread to avoid blocking the event loop,
        while properly propagating CancelledError.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: self.extract(
                    document=document,
                    schema=schema,
                    prompt=prompt,
                    examples=examples,
                    include_extra=include_extra,
                ),
            )
        except asyncio.CancelledError:
            raise
        except NextractError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Async extraction failed: {exc}",
                stage="extract",
            ) from exc

    def _multi_pass_extract(
        self,
        chunks: list[Any],
        schema: dict[str, Any],
        prompt_text: str,
        examples: list[dict[str, Any]] | None,
        include_extra: bool,
    ) -> ExtractionResult:
        pass_outputs: list[Any] = []
        pass_usage: list[dict[str, Any]] = []

        for pass_idx in range(1, self.plan.num_passes + 1):
            merged_data, usage, extractor_name, provider_name = self._run_pass(
                chunks=chunks,
                schema=schema,
                prompt=prompt_text,
                examples=examples,
                include_extra=include_extra,
            )
            pass_outputs.append(merged_data)
            pass_usage.append(usage)
            log.info("extraction_pass_complete", pass_number=pass_idx)

        if schema.get("type") == "array":
            merged_data: list[Any] = []
            seen: set[str] = set()
            for payload in pass_outputs:
                if isinstance(payload, list):
                    for item in payload:
                        item_key = json.dumps(item, sort_keys=True, default=str)
                        if item_key not in seen:
                            seen.add(item_key)
                            merged_data.append(item)
        else:
            merged_data = merge_partial_outputs([p for p in pass_outputs if isinstance(p, dict)])
        usage = self._aggregate_usage_from_dicts(pass_usage)

        return self._build_result(
            merged_data, usage, extractor_name, provider_name, len(chunks), schema,
            passes=self.plan.num_passes,
        )

    def _build_result(
        self,
        merged_data: Any,
        usage: dict[str, Any],
        extractor_name: str,
        provider_name: str,
        num_chunks: int,
        schema: dict[str, Any],
        passes: int = 1,
    ) -> ExtractionResult:
        metadata: dict[str, Any] = {
            "chunks": num_chunks,
            "extractor": extractor_name,
            "provider": provider_name,
            "usage": usage,
            "passes": passes,
            "include_confidence": self.plan.include_confidence,
            "include_citations": self.plan.include_citations,
        }

        if self.plan.schema_validation and isinstance(schema, dict):
            from nextract.validate import SchemaValidator

            schema_validator = SchemaValidator()
            validation = schema_validator.validate(merged_data, schema)
            metadata["validation"] = validation
            if self.plan.include_confidence:
                metadata["confidence"] = validation.metadata.get("completeness")

        if self.plan.include_citations:
            metadata.setdefault("citations", [])

        return ExtractionResult(data=merged_data, metadata=metadata)

    def _merge_chunk_results(self, results: list[ChunkExtraction], schema: dict[str, Any]) -> Any:
        all_payloads: list[Any] = []
        for r in results:
            response = r.response
            if response is None:
                continue
            if isinstance(response, (dict, list)):
                all_payloads.append(response)
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, (dict, list)):
                        all_payloads.append(parsed)
                    else:
                        all_payloads.append({"raw_text": response})
                except (json.JSONDecodeError, ValueError):
                    all_payloads.append({"raw_text": response})
            else:
                all_payloads.append({"value": response})

        if schema.get("type") == "array":
            merged: list[Any] = []
            for payload in all_payloads:
                if isinstance(payload, list):
                    merged.extend(payload)
            return merged

        dict_payloads = [p for p in all_payloads if isinstance(p, dict)]
        if not dict_payloads:
            return {}
        return merge_partial_outputs(dict_payloads)

    def _run_pass(
        self,
        chunks: list[Any],
        schema: dict[str, Any],
        prompt: str,
        examples: list[dict[str, Any]] | None,
        include_extra: bool,
    ) -> tuple[Any, dict[str, Any], str, str]:
        extractor_result = self.extractor.run(
            chunks,
            provider=self.provider,
            prompt=prompt,
            schema=schema,
            examples=examples,
            include_extra=include_extra,
        )

        merged_data = self._merge_chunk_results(extractor_result.results, schema)
        usage = self._aggregate_usage_from_results(extractor_result.results)
        return merged_data, usage, extractor_result.name, extractor_result.provider_name

    def _aggregate_usage_from_results(self, results: list[ChunkExtraction]) -> dict[str, Any]:
        totals = {"requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        for r in results:
            if r.usage is not None:
                totals["requests"] += r.usage.requests
                totals["tool_calls"] += r.usage.tool_calls
                totals["input_tokens"] += r.usage.input_tokens
                totals["output_tokens"] += r.usage.output_tokens
        totals["cost_estimate_usd"] = self._estimate_cost(totals)
        return totals

    def _aggregate_usage_from_dicts(self, usage_items: list[dict[str, Any]]) -> dict[str, Any]:
        totals = {"requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        for item in usage_items:
            usage = item.get("usage") or item
            for key in totals:
                value = usage.get(key) if isinstance(usage, dict) else None
                if isinstance(value, int):
                    totals[key] += value
        totals["cost_estimate_usd"] = self._estimate_cost(totals)
        return totals

    def _estimate_cost(self, usage: dict[str, Any]) -> float | None:
        """Estimate cost from aggregated token usage using configured pricing."""
        import os
        from nextract.pricing import parse_pricing_json, estimate_cost_usd_sync

        pricing_json = os.getenv("NEXTRACT_PRICING", "")
        pricing_map = parse_pricing_json(pricing_json)
        if not pricing_map or not self.plan.extractor.provider.model:
            return None
        try:
            model_name = self.plan.extractor.provider.model
            return estimate_cost_usd_sync(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                model_name=model_name,
                pricing_map=pricing_map,
            )
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("cost_estimation_failed", error=str(exc))
            return None

    def _build_extractor(self, config: ExtractorConfig):
        extractor_class = ExtractorRegistry.get_instance().get(config.name)
        if extractor_class is None:
            raise PipelineError(f"Unknown extractor: {config.name}")
        extractor = extractor_class()
        extractor.initialize(config)
        return extractor

    def _build_provider(self, config: ProviderConfig):
        provider_class = ProviderRegistry.get_instance().get(config.name)
        if provider_class is None:
            raise PipelineError(f"Unknown provider: {config.name}")
        provider = provider_class()
        provider.initialize(config)
        return provider

    def _build_chunker(self, name: str):
        chunker_class = ChunkerRegistry.get_instance().get(name)
        if chunker_class is None:
            raise PipelineError(f"Unknown chunker: {name}")
        chunker = chunker_class()
        chunker.initialize(self.plan.chunker)
        return chunker


class BatchPipeline:
    """Batch extraction pipeline with shared provider for efficiency."""

    def __init__(
        self,
        plan: ExtractionPlan,
        max_workers: int = 4,
        enable_suggestions: bool = False,
        progress_callback: Any | None = None,
    ) -> None:
        self.plan = plan
        self.max_workers = max_workers
        self.enable_suggestions = enable_suggestions
        self.progress_callback = progress_callback

    def extract_batch(
        self,
        documents: Iterable[str],
        schema: dict[str, Any],
        prompt: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
    ) -> BatchExtractionResult:
        docs_list = list(documents)
        results: dict[str, ExtractionResult] = {}

        # Validate plan once up front to fail fast before spawning workers
        PlanValidator.raise_for_invalid(self.plan)

        # Build shared provider and extractor once (F22)
        provider_class = ProviderRegistry.get_instance().get(self.plan.extractor.provider.name)
        if provider_class is None:
            raise PipelineError(f"Unknown provider: {self.plan.extractor.provider.name}")
        shared_provider = provider_class()
        shared_provider.initialize(self.plan.extractor.provider)

        extractor_class = ExtractorRegistry.get_instance().get(self.plan.extractor.name)
        if extractor_class is None:
            raise PipelineError(f"Unknown extractor: {self.plan.extractor.name}")
        shared_extractor = extractor_class()
        shared_extractor.initialize(self.plan.extractor)

        chunker_class = ChunkerRegistry.get_instance().get(self.plan.chunker.name)
        if chunker_class is None:
            raise PipelineError(f"Unknown chunker: {self.plan.chunker.name}")
        shared_chunker = chunker_class()
        shared_chunker.initialize(self.plan.chunker)

        def _run(doc: str) -> tuple[str, ExtractionResult]:
            pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
            pipeline.plan = self.plan
            pipeline.extractor = shared_extractor
            pipeline.provider = shared_provider
            pipeline.chunker = shared_chunker
            return doc, pipeline.extract(
                document=doc,
                schema=schema,
                prompt=prompt,
                examples=examples,
                include_extra=include_extra,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_to_docs: dict[concurrent.futures.Future, str] = {}
            for doc in docs_list:
                future = executor.submit(_run, doc)
                futures_to_docs[future] = doc
            for idx, future in enumerate(concurrent.futures.as_completed(futures_to_docs), start=1):
                try:
                    doc, result = future.result()
                    results[doc] = result
                except NextractError as exc:
                    log.error(
                        "batch_document_failed",
                        error=str(exc),
                        context=exc.context,
                        error_type=type(exc).__name__,
                    )
                    doc_path = futures_to_docs.get(future, "unknown")
                    results[doc_path] = ExtractionResult(
                        data=None,
                        metadata={"error": str(exc), "error_context": exc.context},
                    )
                except Exception as exc:
                    log.error(
                        "batch_document_failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    doc_path = futures_to_docs.get(future, "unknown")
                    results[doc_path] = ExtractionResult(
                        data=None,
                        metadata={"error": str(exc)},
                    )
                if self.progress_callback:
                    self.progress_callback(int((idx / len(docs_list)) * 100))

        suggestions: list[SchemaSuggestion] = []
        if self.enable_suggestions:
            suggestions = self._suggest_schema_improvements(results, schema)

        return BatchExtractionResult(results=results, suggestions=suggestions)

    async def extract_batch_async(
        self,
        documents: Iterable[str],
        schema: dict[str, Any],
        prompt: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        include_extra: bool = False,
    ) -> BatchExtractionResult:
        """Async batch extraction using asyncio.gather for I/O concurrency."""
        docs_list = list(documents)

        PlanValidator.raise_for_invalid(self.plan)

        async def _run_one(doc: str) -> tuple[str, ExtractionResult]:
            pipeline = ExtractionPipeline(self.plan)
            result = await pipeline.extract_async(
                document=doc,
                schema=schema,
                prompt=prompt,
                examples=examples,
                include_extra=include_extra,
            )
            return doc, result

        tasks = [asyncio.create_task(_run_one(doc)) for doc in docs_list]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, ExtractionResult] = {}
        for doc, task_result in zip(docs_list, task_results):
            if isinstance(task_result, Exception):
                log.error("batch_document_failed", error=str(task_result))
                results[doc] = ExtractionResult(
                    data=None,
                    metadata={"error": str(task_result)},
                )
            else:
                _, result = task_result
                results[doc] = result

        suggestions: list[SchemaSuggestion] = []
        if self.enable_suggestions:
            suggestions = self._suggest_schema_improvements(results, schema)

        return BatchExtractionResult(results=results, suggestions=suggestions)

    def _suggest_schema_improvements(
        self,
        results: dict[str, ExtractionResult],
        schema: dict[str, Any],
    ) -> list[SchemaSuggestion]:
        known_fields = set(schema.get("properties", {}).keys())
        counts: dict[str, int] = {}

        for result in results.values():
            data = result.data
            if isinstance(data, list):
                rows = [row for row in data if isinstance(row, dict)]
                for row in rows:
                    for key in row.keys():
                        if key not in known_fields:
                            counts[key] = counts.get(key, 0) + 1
            elif isinstance(data, dict):
                for key in data.keys():
                    if key not in known_fields:
                        counts[key] = counts.get(key, 0) + 1

        suggestions: list[SchemaSuggestion] = []
        total = max(len(results), 1)
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            impact = round((count / total) * 100, 2)
            suggestions.append(
                SchemaSuggestion(
                    description=f"Consider adding field '{key}' observed in {count} results.",
                    impact=impact,
                )
            )

        return suggestions
