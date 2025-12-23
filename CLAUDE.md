# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Python 3.10+ is required
```

### Testing
```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_adaptive_extraction.py -v

# Run with coverage report
pytest tests/ --cov=nextract --cov-report=html

# Run single test
pytest tests/test_integration.py::test_function_name -v
```

### Code Quality
```bash
# Lint with ruff
ruff check nextract

# Type check with mypy
mypy nextract
```

### CLI Usage
```bash
# Extract structured data from document
nextract extract document.pdf --schema schema.json --provider openai

# Convert document to Markdown/HTML
nextract convert document.pdf --format markdown

# List available extractors and chunkers
nextract list extractors
nextract list chunkers --extractor vlm

# Check provider capabilities
nextract check-provider openai --model gpt-4o

# Suggest schema from sample documents
nextract suggest-schema sample1.pdf sample2.pdf --prompt "Extract vendor and totals"

# Validate extraction plan configuration
nextract validate-config plan.json
```

## High-Level Architecture

### Core Abstraction: Dual-Layer Design

Nextract uses a **dual-layer abstraction** that separates extraction **techniques** (Extractors) from **LLM/API backends** (Providers):

**Extractors** = Extraction techniques/algorithms
- `VLMExtractor`: Vision-language model extraction (uses images)
- `TextExtractor`: Text-only extraction
- `OCRExtractor`: OCR-based extraction
- `HybridExtractor`: Combines multiple techniques
- `TextractExtractor`: AWS Textract integration
- `LlamaIndexExtractor`: LlamaIndex integration

**Providers** = LLM/API backend implementations
- `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`
- `AWSProvider` (Bedrock), `AzureProvider`
- `LocalProvider` (Ollama, vLLM)
- OCR providers: `TesseractProvider`, `EasyOCRProvider`, `PaddleOCRProvider`

**Key Relationship**: One Extractor can work with multiple Providers. For example, `VLMExtractor` can use OpenAI's GPT-4o, Anthropic's Claude, or Google's Gemini.

### Modality System

Extraction behavior is determined by **Modality** (defined in `nextract/core/base.py:9`):
- `VISUAL`: Image-based extraction (uses vision models)
- `TEXT`: Text-only extraction
- `HYBRID`: Combines visual and text techniques

Each Extractor declares its modality via `get_modality()`, which determines:
1. Compatible Chunkers (visual chunkers vs text chunkers)
2. Required Provider capabilities (vision support, structured output support)
3. Input data format (images vs text)

### Plugin Registry System

The registry system (`nextract/registry/`) enables plugin discovery and validation:
- `ExtractorRegistry`: Registers and retrieves extractors
- `ProviderRegistry`: Registers and retrieves providers
- `ChunkerRegistry`: Registers and retrieves chunkers, filtered by modality

Registries validate compatibility at runtime:
- Extractors declare supported providers via `get_supported_providers()`
- Chunkers declare applicable modalities via `get_applicable_modalities()`
- Config validation happens early in `ExtractionPlan.validate()`

### Core Configuration Objects

Extraction is configured through a hierarchy of config objects (`nextract/core/config.py`):

```
ExtractionPlan
├── ExtractorConfig
│   ├── ProviderConfig (primary)
│   └── ProviderConfig (fallback, optional)
├── ChunkerConfig
└── Execution parameters (num_passes, retry settings, etc.)
```

- `ProviderConfig`: Model, API keys, timeout, temperature, etc.
- `ExtractorConfig`: Links extractor to provider, enables caching, batch size
- `ChunkerConfig`: Chunking parameters (pages_per_chunk, chunk_size, overlap, etc.)
- `ExtractionPlan`: Complete extraction configuration with validation rules

### Pipeline Orchestration

The pipeline layer (`nextract/pipeline/`) orchestrates the extraction workflow:

1. **ExtractionPipeline**: Single-document extraction
   - Loads document via ingest layer
   - Validates plan and checks provider/extractor compatibility
   - Chunks document based on modality
   - Runs extractor on chunks
   - Merges results with confidence scores
   - Returns `ExtractionResult`

2. **BatchPipeline**: Multi-document parallel extraction
   - Processes multiple documents concurrently
   - Aggregates results across documents
   - Returns `BatchExtractionResult`

3. **PipelineRouter**: Routes documents to appropriate extractors based on capabilities

### Layer Responsibilities

**Ingest Layer** (`nextract/ingest/`):
- File loading and format detection
- Document conversion (DOCX/PPTX → PDF → images)
- Document validation (corruption, password protection)
- Creates `DocumentArtifact` objects

**Parse Layer** (`nextract/parse/`):
- OCR and layout parsing
- Text extraction from PDFs
- Table detection and extraction

**Chunking Layer** (`nextract/chunking/`):
- Splits documents for processing
- Visual modality: `PageChunker` (splits by page groups)
- Text modality: `SemanticChunker`, `FixedSizeChunker`, `SectionChunker`, `TableAwareChunker`
- Hybrid modality: `HybridChunker`

**Extractor Layer** (`nextract/extractors/`):
- Implements extraction techniques
- All extractors extend `BaseExtractor` (`nextract/core/base.py:66`)
- Routes to providers via normalized `ProviderRequest`/`ProviderResponse`

**Provider Layer** (`nextract/providers/`):
- Implements LLM/API clients
- All providers extend `BaseProvider` (`nextract/core/base.py:37`)
- Handles model-specific request formatting
- Implements via pydantic-ai for LLM providers

**Schema Layer** (`nextract/schema/`):
- Schema suggestion from sample documents
- Schema validation helpers
- Structured output parsing

**Validation Layer** (`nextract/validate/`):
- JSON schema validation
- Business rule validation
- Quality scoring

**Output Layer** (`nextract/output/`):
- Formatters: `JsonFormatter`, `MarkdownFormatter`, `HtmlFormatter`, `CsvFormatter`
- Citation rendering
- Metadata enrichment

### Key Data Structures

**Request/Response Objects** (`nextract/core/base.py`):
- `ProviderRequest`: Normalized request to any provider (messages, images, schema, options)
- `ProviderResponse`: Normalized response from provider (text, structured_output, usage, raw)

**Document Objects** (`nextract/core/artifacts.py`):
- `DocumentArtifact`: Canonical representation of loaded document
- `DocumentChunk`: Chunk of document with metadata
- `ExtractorResult`: Result from a single extractor run
- `ExtractionResult`: Final merged result with confidence and provenance

### Extension Points

To add new functionality:

1. **New Extractor**: Extend `BaseExtractor`, implement required methods, register via `@register_extractor`
2. **New Provider**: Extend `BaseProvider`, implement required methods, register via `@register_provider`
3. **New Chunker**: Extend `BaseChunker`, implement required methods, register via `@register_chunker`
4. **New Formatter**: Extend `BaseFormatter`, implement `format()` method

Templates are provided:
- `nextract/extractors/custom_extractor_template.py`
- `nextract/providers/custom_provider_template.py`

## Important Implementation Notes

### Test Status
- Current test pass rate: ~85% (138/163 tests passing)
- 12 tests are marked as "Skip pending implementation" (see recent commit)
- Known issues documented in `tests/README.md`
- When adding new features, ensure tests are added and existing tests pass

### Legacy Code
Several "legacy" files exist during the V2 architecture transition:
- `legacy_core.py`, `legacy_chunking.py`, `legacy_schema.py`, `legacy_cli.py`
- These are deprecated and should not be extended
- Use the new modular architecture in `nextract/core/`, `nextract/extractors/`, etc.

### Async Considerations
- Some tests fail due to async configuration (pytest-asyncio)
- Pipeline execution can be async-aware for future improvements
- Provider implementations may use async clients (pydantic-ai supports async)

### Configuration and Environment
Environment variables (legacy, prefer config objects):
- `NEXTRACT_MODEL`: Default model
- `NEXTRACT_MAX_CONCURRENCY`: Max concurrent requests
- `NEXTRACT_MAX_RUN_RETRIES`: Retry attempts
- `NEXTRACT_PER_CALL_TIMEOUT_SECS`: Request timeout
- `NEXTRACT_PRICING`: Pricing configuration

### Provider-Specific Notes
- Most LLM providers are thin wrappers around pydantic-ai
- OCR providers have specialized implementations with preprocessing
- Provider compatibility is validated at plan validation time
- Fallback providers can be configured in `ExtractorConfig`

### Modality Validation
When modifying extractors/chunkers:
1. Ensure `get_modality()` or `get_applicable_modalities()` is correctly implemented
2. Chunker-extractor compatibility is validated via modality matching
3. Provider capabilities (vision, structured output) are checked against extractor requirements
