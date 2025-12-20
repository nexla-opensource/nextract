from __future__ import annotations

from typing import Any, Dict

from nextract.core import ExtractionPlan, Modality, ValidationResult
from nextract.core.exceptions import PlanError
from nextract.registry import ChunkerRegistry, ExtractorRegistry, ProviderRegistry
import nextract.chunking  # noqa: F401
import nextract.extractors  # noqa: F401
import nextract.providers  # noqa: F401


class PlanValidator:
    """Validates extraction plans based on modalities and compatibility."""

    @staticmethod
    def validate_extraction_plan(plan: ExtractionPlan) -> ValidationResult:
        try:
            plan.extractor.validate()
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(valid=False, errors=[str(exc)])

        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(plan.extractor.name)
        if not extractor_class:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown extractor: {plan.extractor.name}"],
            )

        modality = extractor_class.get_modality()

        chunker_registry = ChunkerRegistry.get_instance()
        chunker_class = chunker_registry.get(plan.chunker.name)

        if chunker_class:
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

        provider_registry = ProviderRegistry.get_instance()
        provider_class = provider_registry.get(plan.extractor.provider.name)

        if provider_class and modality == Modality.VISUAL:
            provider = provider_class()
            provider.initialize(plan.extractor.provider)
            if not provider.supports_vision():
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Provider '{plan.extractor.provider.name}' does not "
                        f"support vision, but extractor requires VISUAL modality"
                    ],
                )

        try:
            plan.chunker.validate(modality)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(valid=False, errors=[str(exc)])

        return ValidationResult(valid=True, errors=[])

    @staticmethod
    def raise_for_invalid(plan: ExtractionPlan) -> None:
        result = PlanValidator.validate_extraction_plan(plan)
        if not result.valid:
            raise PlanError("; ".join(result.errors))


class CapabilityDetector:
    """Detects and reports available capabilities based on a plan."""

    @staticmethod
    def detect_capabilities(plan: ExtractionPlan) -> Dict[str, Any]:
        extractor_class = ExtractorRegistry.get_instance().get(plan.extractor.name)
        provider_class = ProviderRegistry.get_instance().get(plan.extractor.provider.name)

        modality = extractor_class.get_modality() if extractor_class else Modality.TEXT

        provider_capabilities: Dict[str, Any] = {}
        if provider_class:
            provider = provider_class()
            provider.initialize(plan.extractor.provider)
            provider_capabilities = provider.get_capabilities()

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
