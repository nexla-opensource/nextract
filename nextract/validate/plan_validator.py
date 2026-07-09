from __future__ import annotations

from typing import Any

from nextract.core import ExtractionPlan, Modality, ValidationResult
from nextract.core.exceptions import PlanError
from nextract.core.model_capabilities import get_model_capability
from nextract.registry import ChunkerRegistry, ExtractorRegistry
from nextract.registry.bootstrap import ensure_plugins_loaded

ensure_plugins_loaded()

# OCR providers that bypass the LLM model capability table
_OCR_PROVIDERS = frozenset({"tesseract", "easyocr", "paddleocr", "textract"})


class PlanValidator:
    """Validates extraction plans based on modalities and compatibility.

    Uses static lookups (model capability registry, provider class metadata)
    instead of instantiating providers, to avoid side effects.
    """

    @staticmethod
    def validate_extraction_plan(plan: ExtractionPlan) -> ValidationResult:
        # plan.validate() enforces num_passes/backoff bounds and applies the
        # plan retry policy (fills unset provider retries, or forces 1 attempt
        # when retry_on_failure is False).
        try:
            plan.validate()
        except ValueError as exc:
            return ValidationResult(valid=False, errors=[str(exc)])

        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(plan.extractor.name)
        if not extractor_class:
            available = ", ".join(extractor_registry.list_extractors()) or "(none)"
            return ValidationResult(
                valid=False,
                errors=[
                    f"Unknown extractor: {plan.extractor.name}. "
                    f"Available extractors: {available}"
                ],
            )

        modality = extractor_class.get_modality()

        chunker_registry = ChunkerRegistry.get_instance()
        chunker_class = chunker_registry.get(plan.chunker.name)

        if chunker_class is None:
            available = ", ".join(chunker_registry.list_items()) or "(none)"
            return ValidationResult(
                valid=False,
                errors=[
                    f"Unknown chunker: '{plan.chunker.name}'. "
                    f"Available chunkers: {available}"
                ],
            )

        applicable = chunker_class.get_applicable_modalities()
        if modality not in applicable:
            return ValidationResult(
                valid=False,
                errors=[
                    f"Chunker '{plan.chunker.name}' is not "
                    f"applicable to modality '{modality.value}'. "
                    f"Available chunkers: "
                    f"{chunker_registry.get_chunkers_for_modality(modality)}"
                ],
            )

        supported_providers = extractor_class.get_supported_providers()
        if plan.extractor.provider.name not in supported_providers:
            return ValidationResult(
                valid=False,
                errors=[
                    f"Extractor '{plan.extractor.name}' does not support "
                    f"provider '{plan.extractor.provider.name}'. "
                    f"Supported providers: {supported_providers}"
                ],
            )

        # Vision capability check using static lookups (no provider instantiation)
        provider_name = plan.extractor.provider.name
        if modality in {Modality.VISUAL, Modality.HYBRID}:
            if provider_name in _OCR_PROVIDERS:
                has_vision = True  # OCR providers are vision-capable by design
            else:
                has_vision = get_model_capability(
                    model=plan.extractor.provider.model,
                    capability="vision",
                    default=False,
                    provider=provider_name,
                )
            if not has_vision:
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Provider '{provider_name}' does not "
                        f"support vision, but extractor requires {modality.value.upper()} modality"
                    ],
                )

        try:
            plan.chunker.validate(modality)
        except ValueError as exc:
            return ValidationResult(valid=False, errors=[str(exc)])

        return ValidationResult(valid=True, errors=[])

    @staticmethod
    def raise_for_invalid(plan: ExtractionPlan) -> None:
        result = PlanValidator.validate_extraction_plan(plan)
        if not result.valid:
            raise PlanError("; ".join(result.errors))


class CapabilityDetector:
    """Detects and reports available capabilities based on a plan.

    Uses static lookups to avoid side effects from provider instantiation.
    """

    @staticmethod
    def detect_capabilities(plan: ExtractionPlan) -> dict[str, Any]:
        extractor_class = ExtractorRegistry.get_instance().get(plan.extractor.name)
        provider_name = plan.extractor.provider.name

        modality = extractor_class.get_modality() if extractor_class else Modality.TEXT

        # Static capability detection without instantiation
        if provider_name in _OCR_PROVIDERS:
            provider_capabilities: dict[str, Any] = {
                "vision": True,
                "structured_output": False,
                "ocr": True,
            }
        else:
            provider_capabilities = {
                "vision": get_model_capability(
                    model=plan.extractor.provider.model,
                    capability="vision",
                    default=False,
                    provider=provider_name,
                ),
                "structured_output": get_model_capability(
                    model=plan.extractor.provider.model,
                    capability="structured_output",
                    default=True,
                    provider=provider_name,
                ),
            }

        capabilities = {
            "modality": modality.value,
            "supported_chunkers": ChunkerRegistry.get_instance().get_chunkers_for_modality(modality),
            "provider_capabilities": provider_capabilities,
            "multi_pass_extraction": plan.num_passes > 1,
            "has_fallback": plan.extractor.fallback_provider is not None,
            "confidence_scoring": plan.include_confidence,
            "citation_tracking": plan.include_citations,
        }
        return capabilities
