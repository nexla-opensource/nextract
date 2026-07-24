# Provenance

Modules in this package were extracted **verbatim** from the production
monolith `chunking_code.py` (Nexla transform). Body code is unchanged except
for the documented seam edits (see package README section / PR description).

| module | monolith lines | sha256 (extracted body) |
|---|---|---|
| `config.py` | 104-199 | `9025908c398c79f4` |
| `models.py` | 200-217 | `d3c6a934ad4b6723` |
| `chunker_utils.py` | 218-546 | `696875bb05293904` |
| `prompts.py` | 547-1537 | `f8a0dd9317482620` |
| `llm.py` | 1538-1675 | `2804cf471c53df30` |
| `pdf.py` | 1676-1862 | `074edc532fdfbc46` |
| `merger.py` | 1863-2042 | `a4d43154e3983721` |
| `pipeline.py` | 2043-6189 | `8b0613d2df8e60c1` |

Notes on completeness and conventions:

- Hashes are truncated sha256 (first 16 hex chars) of each extracted body and
  refer to the pre-seam extraction; the seam edits below are the only body
  deltas since. Bodies begin immediately after each module's generated header
  (everything through the `logger = structlog.get_logger(__name__)` line; the
  blank line after it belongs to the monolith segment).
- Deliberately NOT ported: monolith lines 1-103 (import block, import-time
  `logging.basicConfig(DEBUG)` + pdfminer/pdfplumber log suppression, and
  DependencyManager runtime pip-install machinery, inert with
  REQUIRED_PACKAGES=[]) and lines 6190-6272 (the legacy Nexla `process()`
  entrypoint with platform credential lookup); the `RagDocumentChunker`
  facade in `__init__.py` replaces the entrypoint, and log configuration is
  left to the host application.
- Adaptation for nextract: three seam edits made during the extraction
  (config env reads moved to instantiation time via `field(default_factory)`;
  `LLMService`/`DocumentPipeline` gained a `use_vertex` client-mode flag; the
  `process_file` nexla_meta/tags full-dump log block — PII into logs — was
  deleted), plus two header-only changes: the stdlib-logging -> structlog
  logger swap and an unused-import (F401) prune. Body code is otherwise
  byte-identical to the monolith ranges above.
