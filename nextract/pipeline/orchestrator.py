from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import structlog

from nextract.core import ExtractionPlan, ExtractionResult, ExtractorConfig, ProviderConfig
import nextract.chunking  # noqa: F401
import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401
from nextract.ingest.loaders import load_documents
from nextract.merge import merge_partial_outputs
from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry
from nextract.schema import SchemaSuggestion
from nextract.validate import PlanValidator

log = structlog.get_logger(__name__)


@dataclass
class BatchExtractionResult:
    results: Dict[str, ExtractionResult]
    suggestions: List[SchemaSuggestion] = field(default_factory=list)


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
        document: str | List[str],
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        include_extra: bool = False,
    ) -> ExtractionResult:
        documents = [document] if isinstance(document, str) else list(document)
        artifacts = load_documents(documents)

        from nextract.ingest import DocumentValidator

        validator = DocumentValidator()
        for artifact in artifacts:
            validation = validator.validate(artifact)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))

        chunks: List[Any] = []
        for artifact in artifacts:
            chunks.extend(self.chunker.chunk(artifact, self.plan.chunker))

        if not chunks:
            log.warning("no_chunks_generated", files=documents)
            return ExtractionResult(data={}, metadata={"chunks": 0})

        prompt_text = prompt or "Extract the requested fields."
        if self.plan.num_passes > 1:
            pass_outputs: List[Any] = []
            pass_usage: List[Dict[str, Any]] = []

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
                merged_data = []
                for payload in pass_outputs:
                    if isinstance(payload, list):
                        merged_data.extend(payload)
            else:
                merged_data = merge_partial_outputs([p for p in pass_outputs if isinstance(p, dict)])
            usage = self._aggregate_usage_from_passes(pass_usage)
            metadata = {
                "chunks": len(chunks),
                "extractor": extractor_name,
                "provider": provider_name,
                "usage": usage,
                "passes": self.plan.num_passes,
                "include_confidence": self.plan.include_confidence,
                "include_citations": self.plan.include_citations,
            }

            if self.plan.schema_validation and isinstance(schema, dict):
                from nextract.validate import SchemaValidator

                validator = SchemaValidator()
                validation = validator.validate(merged_data, schema)
                metadata["validation"] = validation
                if self.plan.include_confidence:
                    metadata["confidence"] = validation.metadata.get("completeness")

            if self.plan.include_citations:
                metadata.setdefault("citations", [])

            return ExtractionResult(data=merged_data, metadata=metadata)

        merged_data, usage, extractor_name, provider_name = self._run_pass(
            chunks=chunks,
            schema=schema,
            prompt=prompt_text,
            examples=examples,
            include_extra=include_extra,
        )

        metadata = {
            "chunks": len(chunks),
            "extractor": extractor_name,
            "provider": provider_name,
            "usage": usage,
            "include_confidence": self.plan.include_confidence,
            "include_citations": self.plan.include_citations,
        }

        if self.plan.schema_validation and isinstance(schema, dict):
            from nextract.validate import SchemaValidator

            validator = SchemaValidator()
            validation = validator.validate(merged_data, schema)
            metadata["validation"] = validation
            if self.plan.include_confidence:
                metadata["confidence"] = validation.metadata.get("completeness")

        if self.plan.include_citations:
            metadata.setdefault("citations", [])

        return ExtractionResult(data=merged_data, metadata=metadata)

    def _merge_chunk_results(self, results: List[Dict[str, Any]], schema: Dict[str, Any]) -> Any:
        payloads = [r.get("response") for r in results if isinstance(r.get("response"), (dict, list))]
        if schema.get("type") == "array":
            merged: List[Any] = []
            for payload in payloads:
                if isinstance(payload, list):
                    merged.extend(payload)
            return merged

        dict_payloads = [p for p in payloads if isinstance(p, dict)]
        if not dict_payloads:
            return {}
        return merge_partial_outputs(dict_payloads)

    def _run_pass(
        self,
        chunks: List[Any],
        schema: Dict[str, Any],
        prompt: str,
        examples: Optional[List[Dict[str, Any]]],
        include_extra: bool,
    ) -> tuple[Any, Dict[str, Any], str, str]:
        extractor_result = self.extractor.run(
            chunks,
            provider=self.provider,
            prompt=prompt,
            schema=schema,
            examples=examples,
            include_extra=include_extra,
        )

        merged_data = self._merge_chunk_results(extractor_result.results, schema)
        usage = self._aggregate_usage(extractor_result.results)
        return merged_data, usage, extractor_result.name, extractor_result.provider_name

    def _aggregate_usage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        totals = {"requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        for result in results:
            usage = result.get("usage") or {}
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
        return totals

    def _aggregate_usage_from_passes(self, passes: List[Dict[str, Any]]) -> Dict[str, Any]:
        totals = {"requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        for usage in passes:
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
        return totals

    def _build_extractor(self, config: ExtractorConfig):
        extractor_class = ExtractorRegistry.get_instance().get(config.name)
        if extractor_class is None:
            raise ValueError(f"Unknown extractor: {config.name}")
        extractor = extractor_class()
        extractor.initialize(config)
        return extractor

    def _build_provider(self, config: ProviderConfig):
        provider_class = ProviderRegistry.get_instance().get(config.name)
        if provider_class is None:
            raise ValueError(f"Unknown provider: {config.name}")
        provider = provider_class()
        provider.initialize(config)
        return provider

    def _build_chunker(self, name: str):
        chunker_class = ChunkerRegistry.get_instance().get(name)
        if chunker_class is None:
            raise ValueError(f"Unknown chunker: {name}")
        return chunker_class()


class BatchPipeline:
    """Batch extraction pipeline with simple parallelism."""

    def __init__(
        self,
        plan: ExtractionPlan,
        max_workers: int = 4,
        enable_suggestions: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self.plan = plan
        self.max_workers = max_workers
        self.enable_suggestions = enable_suggestions
        self.progress_callback = progress_callback

    def extract_batch(
        self,
        documents: Iterable[str],
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        include_extra: bool = False,
    ) -> BatchExtractionResult:
        docs_list = list(documents)
        results: Dict[str, ExtractionResult] = {}

        def _run(doc: str) -> tuple[str, ExtractionResult]:
            pipeline = ExtractionPipeline(self.plan)
            return doc, pipeline.extract(
                document=doc,
                schema=schema,
                prompt=prompt,
                examples=examples,
                include_extra=include_extra,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run, doc) for doc in docs_list]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                doc, result = future.result()
                results[doc] = result
                if self.progress_callback:
                    self.progress_callback(int((idx / len(docs_list)) * 100))

        suggestions: List[SchemaSuggestion] = []
        if self.enable_suggestions:
            suggestions = self._suggest_schema_improvements(results, schema)

        return BatchExtractionResult(results=results, suggestions=suggestions)

    def _suggest_schema_improvements(
        self,
        results: Dict[str, ExtractionResult],
        schema: Dict[str, Any],
    ) -> List[SchemaSuggestion]:
        known_fields = set(schema.get("properties", {}).keys())
        counts: Dict[str, int] = {}

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

        suggestions: List[SchemaSuggestion] = []
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
