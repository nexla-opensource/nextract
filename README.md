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

`extract_simple` uses `mode="auto"` by default: **PDFs and images use visual (VLM) extraction**; plain text files use the text extractor. Pass `mode="text"` to force text-only.

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
# Metadata includes both "provider" and "provider_name"
print(result.metadata.get("provider_name"))
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

Structured logs go to **stderr** so JSON on stdout stays pipe-friendly. Use `nextract --version` / `-V` for the package version.

```bash
# Version
nextract --version

# Extract (default extractor: auto — PDF/images→vlm, text files→text)
nextract extract invoice.pdf --schema schema.json --provider openai

# Explicit extractor/chunker (omit --chunker to auto-select by modality)
nextract extract contract.pdf \
  --schema contract_schema.json \
  --extractor vlm \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --chunker page \
  --pages-per-chunk 3

# Batch extract (default extractor: auto; exits 1 if any document fails)
nextract batch doc1.pdf doc2.pdf --schema schema.json --provider openai

# List extractors, providers, chunkers
nextract list extractors
nextract list providers
nextract list chunkers --extractor vlm   # marks section/table_aware as experimental

# Check provider capabilities (and required credentials)
nextract check-provider openai --model gpt-4o
nextract check-provider openai --smoke   # optional connectivity probe

# Convert: extracts text, then formats it (not layout-preserving conversion)
nextract convert docs/report.pdf --format markdown

# Suggest a schema from samples
nextract suggest-schema sample1.pdf sample2.pdf --prompt "Extract vendor and totals"

# Validate a plan file (exits 1 if invalid)
nextract validate-config plan.json
```

### CLI defaults and behavior

| Topic | Behavior |
|-------|----------|
| Default extractor | `auto` (PDF/images → `vlm`, text files → `text`). Override with `--extractor`. Mixed-type batches must set `--extractor` explicitly. |
| Auto-chunker | If `--chunker` is omitted: `page` for vlm/ocr/textract, `semantic` for text/llamaindex, `hybrid` for hybrid. |
| `batch` exit code | Non-zero if any document fails. |
| `validate-config` | Prints validity; **exit code 1** when the plan is invalid. |
| `convert` | Text extraction + format only (`markdown`, `html`, `csv`, `json`). Fails on empty text or unknown format. Not full document conversion. |
| Logs | Structured logs on stderr; JSON results on stdout (unless `-o`). |

### Credentials

Set provider API keys via environment variables (or pass `api_key` in `ProviderConfig`). Missing keys produce a non-fatal warning before the run.

| Provider | Env vars |
|----------|----------|
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| google | `GOOGLE_API_KEY` |
| azure | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| cohere | `CO_API_KEY` |
| bedrock / aws | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

`nextract check-provider <name>` reports capabilities, required env vars, and which are missing.

## Configuration

Provider and extractor settings use `ProviderConfig`, `ExtractorConfig`, `ChunkerConfig`, and `ExtractionPlan` (retries, multipass, validation).

Env vars used by the public pipeline path:

- `NEXTRACT_PRICING` — optional cost estimate map for usage metadata

Legacy `RuntimeConfig` env vars (`NEXTRACT_MODEL`, `NEXTRACT_MAX_CONCURRENCY`, `NEXTRACT_MAX_RUN_RETRIES`, `NEXTRACT_PER_CALL_TIMEOUT_SECS`, multipass/provenance flags) apply only to the older agent_runner path, not `ExtractionPipeline` / CLI extract.

SDK notes:

- `extract_simple(..., mode=...)` — `auto` (default), `text`, `visual`, `ocr`, `textract`, `hybrid`
- `batch_extract(..., enable_suggestions=False)` — set `True` for post-batch schema suggestions
- Exceptions: `NextractError`, `PipelineError`, `PlanError`, `ProviderAuthError`, `ProviderRequestError` (exported from `nextract`)

## Development

```bash
pytest
ruff check nextract
mypy nextract
```
