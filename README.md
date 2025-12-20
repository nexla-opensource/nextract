# nextract

nextract is an open-source Python package for intelligent document ingestion and structured data extraction. It provides a modular architecture with extractors, providers, and chunkers so you can mix and match techniques and backends while keeping a consistent API.

## Features

- Dual-layer abstraction: extractors (technique) + providers (LLM or API backend)
- Plugin registry for new extractors, providers, and chunkers
- Modality-aware chunking (visual, text, hybrid)
- Extraction plans with validation and capability checks
- CLI and Python SDK

## Installation

```bash
pip install nextract
```

For development:

```bash
pip install -e .[dev]
```

Python 3.10+ is required.

## Quick Start

### Simple extraction

```python
from nextract import extract_simple

schema = {
    "title": "Invoice",
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"}
    },
    "required": ["invoice_number", "total"]
}

result = extract_simple(
    document="invoice.pdf",
    schema=schema,
    provider="openai",
    model="gpt-4o",
    prompt="Extract invoice fields"
)

print(result.data)
```

### Full control with an extraction plan

```python
from nextract import ExtractionPipeline
from nextract.core import ExtractionPlan, ExtractorConfig, ChunkerConfig, ProviderConfig

plan = ExtractionPlan(
    extractor=ExtractorConfig(
        name="vlm",
        provider=ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="your-key"
        )
    ),
    chunker=ChunkerConfig(
        name="page",
        pages_per_chunk=3,
        page_overlap=1
    ),
    num_passes=1
)

pipeline = ExtractionPipeline(plan)
result = pipeline.extract(
    document="contract.pdf",
    schema={"type": "object", "properties": {"party": {"type": "string"}}},
    prompt="Extract contract details"
)

print(result.data)
```

## CLI

```bash
# Extract with defaults
nextract extract invoice.pdf --schema schema.json --provider openai

# Explicit extractor and chunker
nextract extract contract.pdf \
  --schema contract_schema.json \
  --extractor vlm \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --chunker page \
  --pages-per-chunk 3

# List available extractors and chunkers
nextract list extractors
nextract list chunkers --extractor text

# Check provider capabilities
nextract check-provider openai --model gpt-4o

# Convert a document to Markdown or HTML
nextract convert docs/report.pdf --format markdown

# Suggest a schema from samples
nextract suggest-schema sample1.pdf sample2.pdf --prompt "Extract vendor and totals"

# Validate a plan file
nextract validate-config plan.json
```

## Configuration

Provider and extractor settings are configured via `ProviderConfig`, `ExtractorConfig`, and `ChunkerConfig`. Environment variables can still be used for legacy runtime settings:

- `NEXTRACT_MODEL`
- `NEXTRACT_MAX_CONCURRENCY`
- `NEXTRACT_MAX_RUN_RETRIES`
- `NEXTRACT_PER_CALL_TIMEOUT_SECS`
- `NEXTRACT_PRICING`

## Development

```bash
pytest
ruff check nextract
mypy nextract
```
