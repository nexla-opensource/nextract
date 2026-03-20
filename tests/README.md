# Nextract Test Suite

**Total Tests**: ~400+
**Unit Tests**: 282 passing, 12 skipped
**Integration Tests**: Require API credentials
**Last Updated**: 2026-03-20

---

## Overview

This directory contains comprehensive unit and integration tests for Nextract's V2 architecture, organized into two main categories:

- **Unit Tests** (`tests/unit/`): Self-contained tests that don't require external services
- **Integration Tests** (`tests/integration/`): Tests that require API credentials for providers

### Running Tests

```bash
# Install package in editable mode
pip install -e .[dev]

# Run all unit tests (no credentials needed)
pytest tests/unit/ -v

# Run unit tests with coverage
pytest tests/unit/ --cov=nextract --cov-report=html

# Collect integration tests (verify imports work)
pytest tests/integration/ --collect-only

# Run integration tests (requires API keys)
pytest tests/integration/ -v
```

---

## Test Structure

```
tests/
├── __init__.py              # Makes tests a proper Python package
├── README.md                # This file
├── unit/                    # Unit tests (no external dependencies)
│   ├── conftest.py          # Unit test fixtures
│   ├── test_adaptive/       # Adaptive extraction tests
│   ├── test_chunking/       # Chunking algorithm tests
│   ├── test_core/           # Core config and artifact tests
│   ├── test_error/          # Error handling tests
│   ├── test_merge/          # Partial output merge tests
│   ├── test_multipass/      # Multi-pass extraction tests
│   ├── test_parallel/       # Parallel processing tests
│   ├── test_provenance/     # Provenance tracking tests
│   └── test_schema/         # Schema utility tests
└── integration/             # Integration tests (require credentials)
    ├── conftest.py          # Integration fixtures & skip markers
    ├── fixtures/            # Test schemas and data
    ├── test_cli/            # CLI command tests
    ├── test_chunkers/       # Chunker integration tests
    ├── test_edge_cases/     # Edge case tests
    ├── test_extractors/     # Extractor integration tests
    ├── test_pipelines/      # Pipeline integration tests
    ├── test_plans/          # Plan validation tests
    ├── test_providers/      # Provider integration tests
    └── test_schemas/        # Schema suggestion/validation tests
```

---

## Unit Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| Adaptive Extraction | ~60 | Passing |
| Chunking | 32 | Passing |
| Core (Config, Artifacts) | ~30 | Passing |
| Error Handling | ~15 | Passing |
| Merge Logic | ~20 | Passing |
| Multi-pass | ~15 | Passing |
| Parallel Processing | 20 | Passing |
| Provenance | ~20 | Passing (1 skipped) |
| Schema Utils | ~15 | Passing |
| Component Integration | ~10 | Passing |
| **TOTAL** | **~282** | **Passing** |

---

## Integration Tests

Integration tests require API credentials. Set environment variables before running:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# Azure
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="..."

# AWS (Bedrock)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

Tests will be skipped if credentials are not available.

---

## Running Specific Test Categories

```bash
# Unit tests by category
pytest tests/unit/test_adaptive/ -v
pytest tests/unit/test_chunking/ -v
pytest tests/unit/test_core/ -v
pytest tests/unit/test_provenance/ -v

# Integration tests by category
pytest tests/integration/test_providers/ -v
pytest tests/integration/test_extractors/ -v
pytest tests/integration/test_pipelines/ -v
pytest tests/integration/test_cli/ -v
```

---

## Fixtures

### Unit Test Fixtures (`tests/unit/conftest.py`)
- `simple_schema()`, `nested_schema()`, `array_schema()` - JSON schemas
- `mock_chunk()`, `mock_chunks()` - Mock document chunks
- `complete_extraction_result()`, `partial_extraction_result()` - Result fixtures

### Integration Test Fixtures (`tests/integration/conftest.py`)
- `has_provider_credentials(provider)` - Check if credentials exist
- `requires_openai`, `requires_anthropic`, etc. - Skip markers
- `simple_schema()`, `nested_schema()` - Schema fixtures
- `sample_pdf_path()`, `sample_text_content()` - Document fixtures

---

## Known Considerations

1. **Integration tests require credentials**: Without API keys, integration tests will be skipped
2. **Some tests marked skip**: 12 unit tests are skipped pending implementation
3. **Async support**: Tests use pytest-asyncio with `asyncio_mode = "auto"`

---

**See CLAUDE.md for architecture documentation**
