# Nextract Test Suite

**Total Tests**: 163  
**Passing**: 138 (84.7%)  
**Failing**: 25 (15.3%)  
**Last Updated**: 2025-11-17

---

## Overview

This directory contains comprehensive unit and integration tests for Nextract's core functionality including adaptive extraction, sentence-aware chunking, parallel processing, and provenance tracking.

### Running Tests

```bash
# Install package in editable mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_adaptive_extraction.py -v

# Run with coverage
pytest tests/ --cov=nextract --cov-report=html
```

---

## Test Summary by Category

| Category | Total | Passing | Failing | Status |
|----------|-------|---------|---------|--------|
| **Adaptive Extraction** | 29 | 25 | 4 | ⚠️ Prompt format |
| **Nested Adaptive** | 34 | 34 | 0 | ✅ Perfect |
| **Sentence Chunking** | 32 | 32 | 0 | ✅ Perfect |
| **Parallel Processing** | 20 | 20 | 0 | ✅ Perfect |
| **Provenance** | 20 | 19 | 1 | ⚠️ Citation |
| **Integration** | 4 | 0 | 4 | ⚠️ Async |
| **Multi-pass** | 9 | 1 | 8 | ⚠️ Async |
| **Error Handling** | 5 | 0 | 5 | ⚠️ Async |
| **TOTAL** | **163** | **138** | **25** | **84.7%** |

---

## Test Files

### 1. `test_adaptive_extraction.py` (29 tests, 25 passing)

**Purpose**: Test two-pass adaptive extraction functionality

**Coverage**:
- ✅ Field identification (missing vs populated)
- ✅ Focused schema creation
- ⚠️ Focused prompt creation (4 tests failing)
- ✅ Result merging
- ✅ Nested schema support

**Status**: ⚠️ Minor failures (prompt format validation needs update)

---

### 2. `test_adaptive_extraction_nested.py` (34 tests, 34 passing ✅)

**Purpose**: Test adaptive extraction with deeply nested schemas

**Coverage**:
- ✅ Recursive field counting (52 leaf fields across 4 levels)
- ✅ Nested field identification
- ✅ Dot-notation path handling
- ✅ Deep merge of results

**Status**: ✅ All passing

---

### 3. `test_sentence_chunking.py` (32 tests, 32 passing ✅)

**Purpose**: Test sentence-aware chunking (LangExtract-inspired)

**Coverage**:
- ✅ Character interval tracking
- ✅ Sentence boundary detection
- ✅ LangExtract Rule A (long sentences break at newlines)
- ✅ LangExtract Rule B (oversized sentences standalone)
- ✅ LangExtract Rule C (short sentences packed)

**Status**: ✅ All passing

---

### 4. `test_parallel_processing.py` (20 tests, 20 passing ✅)

**Purpose**: Test parallel file/chunk processing

**Coverage**:
- ✅ Concurrent extraction
- ✅ Error handling per item
- ✅ Batch result aggregation

**Status**: ✅ All passing

---

### 5. `test_provenance.py` (20 tests, 19 passing)

**Purpose**: Test provenance tracking

**Coverage**:
- ✅ Field provenance tracking
- ✅ Source file/chunk/page tracking
- ✅ Confidence scores
- ⚠️ Citation generation (1 test failing)

**Status**: ⚠️ Minor failure (citation formatting)

---

### 6. `test_integration.py` (4 tests, 0 passing)

**Purpose**: End-to-end integration tests

**Status**: ⚠️ Async configuration needed (all 4 tests failing)

---

### 7. `test_multipass.py` (9 tests, 1 passing)

**Purpose**: Test multi-pass extraction with merge strategies

**Status**: ⚠️ Async configuration needed (8 tests failing)

---

### 8. `test_error_handling.py` (5 tests, 0 passing)

**Purpose**: Test error handling and recovery

**Status**: ⚠️ Async configuration needed (all 5 tests failing)

---

## Known Issues

### 1. Async Test Configuration (17 tests failing)

**Issue**: Tests using `@pytest.mark.asyncio` failing

**Solution**: Configure pytest-asyncio in `pyproject.toml`

---

### 2. Prompt Format Validation (4 tests failing)

**Issue**: Adaptive extraction prompt format changed with anti-hallucination rules

**Solution**: Update test assertions to match new prompt format

---

### 3. Citation Ellipsis Logic (1 test failing)

**Issue**: Citation generation doesn't add ellipsis for short text

**Solution**: Update citation generation logic or test expectations

---

### 4. Multi-pass Validation (1 test failing)

**Issue**: `fail_threshold` validation not raising ValueError

**Solution**: Add validation in `MultiPassExtractor.__init__`

---

## Test Coverage Goals

**Current**: 84.7%  
**Target**: 95%+

**To Achieve**:
1. Fix 17 async tests (pytest-asyncio configuration)
2. Fix 4 prompt format tests (update assertions)
3. Fix 1 citation test (update logic)
4. Fix 1 validation test (add validation)

---

## Running Specific Test Categories

```bash
# Adaptive extraction tests
pytest tests/test_adaptive_extraction.py -v

# Nested schema tests
pytest tests/test_adaptive_extraction_nested.py -v

# Chunking tests
pytest tests/test_sentence_chunking.py -v

# Parallel processing tests
pytest tests/test_parallel_processing.py -v

# Provenance tests
pytest tests/test_provenance.py -v
```

---

**See ARCHITECTURE.md for detailed test documentation**
