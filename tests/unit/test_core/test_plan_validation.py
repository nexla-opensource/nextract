from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.validate import PlanValidator


def test_vlm_extractor_with_page_chunker_valid():
    plan = ExtractionPlan(
        extractor=ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        ),
        chunker=ChunkerConfig(name="page", pages_per_chunk=3),
    )

    result = PlanValidator.validate_extraction_plan(plan)
    assert result.valid is True


def test_vlm_extractor_with_semantic_chunker_invalid():
    plan = ExtractionPlan(
        extractor=ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        ),
        chunker=ChunkerConfig(name="semantic"),
    )

    result = PlanValidator.validate_extraction_plan(plan)
    assert result.valid is False
    assert "not applicable to modality" in result.errors[0]


def test_text_extractor_with_semantic_chunker_valid():
    plan = ExtractionPlan(
        extractor=ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
        ),
        chunker=ChunkerConfig(name="semantic", chunk_size=2000),
    )

    result = PlanValidator.validate_extraction_plan(plan)
    assert result.valid is True


def test_extractor_provider_compatibility():
    plan = ExtractionPlan(
        extractor=ExtractorConfig(
            name="textract",
            provider=ProviderConfig(name="openai", model="gpt-4o"),
            extractor_params={
                "aws_access_key": "x",
                "aws_secret_key": "y",
                "region": "us-east-1",
            },
        ),
        chunker=ChunkerConfig(name="page"),
    )

    result = PlanValidator.validate_extraction_plan(plan)
    assert result.valid is False
    assert "does not support provider" in result.errors[0]


def test_hybrid_extractor_requires_vision_capable_provider():
    plan = ExtractionPlan(
        extractor=ExtractorConfig(
            name="hybrid",
            provider=ProviderConfig(name="local", model="llama3"),
        ),
        chunker=ChunkerConfig(name="hybrid"),
    )

    result = PlanValidator.validate_extraction_plan(plan)
    assert result.valid is False
    assert "support vision" in result.errors[0]
