# nextract

**nextract** is a small, pragmatic framework for **structured data extraction** from files using the **Pydantic AI Agent**. It focuses on clean boundaries, strong typing, and JSON Schema/Pydantic-driven outputs—while keeping file handling simple and predictable.

> **Scope of this build**
>
> * Uses **Pydantic AI Agent** only.
> * Takes **local file paths** and feeds content to the Agent:
>
>   * Text files are read as text and wrapped in delimiters.
>   * PDFs and images are attached as **binary bytes**.
>   * Office docs (`.doc/.docx/.ppt/.pptx`) are converted to **PDF** first when a converter is available.
>   * Excel files: `.xlsx` is extracted to **TSV** (in-process); `.xls` attempts **CSV** via LibreOffice/unoconv.
> * **No OCR**, no large-file chunking yet (TODO).
> * Returns a **`dict`** by default, or a **Pydantic model instance** if you pass a model and request it.
> * Tracing via **structlog**; usage & **cost estimation** from Agent usage + a simple model pricing table.

---

## Table of Contents

* [Features](#features)
* [What’s in / out of scope](#whats-in--out-of-scope)
* [Installation](#installation)
* [Quick Start](#quick-start)

  * [JSON Schema output (default dict)](#json-schema-output-default-dict)
  * [Pydantic model output](#pydantic-model-output)
  * [Batch extraction (parallel)](#batch-extraction-parallel)
* [CLI](#cli)
* [Configuration](#configuration)

  * [Environment variables](#environment-variables)
  * [Pricing configuration](#pricing-configuration)
  * [Model selection](#model-selection)
* [How it works](#how-it-works)
* [File type handling](#file-type-handling)
* [Office → PDF conversion](#office--pdf-conversion)
* [Examples & Few-shot Hints](#examples--few-shot-hints)
* [Return shape](#return-shape)
* [Logging & Tracing](#logging--tracing)
* [Retries, Rate Limits & Timeouts](#retries-rate-limits--timeouts)
* [Large files (TODO)](#large-files-todo)
* [Limitations](#limitations)
* [FAQ](#faq)
* [License](#license)

---

## Features

* **Structured extraction** for small files with:

  * **JSON Schema** (output as `dict[str, Any]`), or
  * **Pydantic v2 models** (output as dict by default; optional model instance).
* **Pydantic AI Agent** integration:

  * Raw **binary attachments** for PDFs/images.
  * **StructuredDict** for JSON Schema outputs.
  * Usage metrics retrieved from the run.
* **Batch** mode runs one file per Agent call in **parallel**.
* **Cost** estimation via a simple **pricing map** (optional).
* **Structlog** logging to console.
* **ZIP** files: extract to `/tmp` and process each contained file “as-is”.

---

## What’s in / out of scope

**Supported file types**

* **Read Text Directly**: 

`.txt`, `.md`, `.csv`, `.tsv`, `.xls`, `.xlsx`, `.json`, `.xml`, `.yaml`, `.yml`, `.html`, `.htm`

* **Upload Directly (binary)**: Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tiff`), **PDF** (`.pdf`)

* **ZIP**: Extracted to `/tmp/nextract-zip-<name>` and each file inside is processed “as-is”.

* **Accepted as binary, converted to PDF before uploading to LLMs**
`.doc`, `.docx`, `.ppt`, `.pptx`

**Not supported for now**

* **Audio/Video** processing
* **OCR** for scanned PDFs/images
* **Large-file chunking & merging** (design is stubbed; not implemented)

---

## Installation

```bash
# Install from PyPI
pip install nextract

# Or install from source for development
git clone https://github.com/your-username/nextract.git
cd nextract
pip install -e .[dev]
```

> **Python**: 3.10+

---

## Quick Start

### JSON Schema output (default dict)

```python
from nextract import extract

schema = {
    "title": "Invoice",
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total": {"type": "number"}
    },
    "required": ["invoice_number", "total"]
}

res = extract(
    files=["./docs/invoice.pdf"],
    schema_or_model=schema,
    user_prompt="Extract the invoice fields exactly as defined.",
    include_extra=True,  # adds a top-level `extra` bag for helpful unmodeled fields
)

print(res["data"])   # dict[str, Any] matching your schema (+ optional `extra`)
print(res["report"]) # model, usage, cost_estimate_usd, warnings
```

### Pydantic model output

```python
from pydantic import BaseModel
from nextract import extract

class Invoice(BaseModel):
    invoice_number: str
    date: str | None = None
    total: float

res = extract(
    files=["./docs/invoice.pdf"],
    schema_or_model=Invoice,
    user_prompt="Extract the invoice fields."
    # include_extra is ignored for Pydantic model mode
)

# Default behavior returns a dict
print(res["data"])  # -> {'invoice_number': '...', 'date': '...', 'total': ...}
```

To get the **Pydantic model instance** instead of a dict:

```python
res = extract(
    files=["./docs/invoice.pdf"],
    schema_or_model=Invoice,
    user_prompt="Extract the invoice fields.",
    return_pydantic=True,
)
invoice_obj = res["data"]  # -> Invoice instance
```

### Batch extraction (parallel)

Process each file **independently** (one Agent call per file):

```python
from nextract import batch_extract

schema = {
    "title": "DocSummary",
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title"]
}

res = batch_extract(
    batch=["./a.pdf", "./b.png", "./c.txt"],   # or [["./a1.pdf","./a2.pdf"], ["./b1.pdf"]] to group
    schema_or_model=schema,
    user_prompt="Summarize each document with title + summary.",
    include_extra=False,
    max_concurrency=4,
)

# result is a dict keyed by the first file in each item
print(res.keys())  # -> {"./a.pdf", "./b.png", "./c.txt"}
```

---

## CLI

```bash
# JSON Schema
nextract extract ./invoice.pdf \
  --schema ./invoice.schema.json \
  --prompt "Extract the invoice fields." \
  --include-extra

# Pydantic model (module:Class or module.Class)
nextract extract ./invoice.pdf \
  --pydantic-model mypkg.models:Invoice

# Batch (parallel): schema mode
nextract batch ./a.pdf ./b.png ./c.txt \
  --schema ./summary.schema.json \
  --prompt "Summarize each document." \
  --max-concurrency 4
```

> Run `nextract --help`, `nextract extract --help`, or `nextract batch --help` for more.

---

## Configuration

### Environment variables

| Variable                         | Default         | Description                                         |
| -------------------------------- | --------------- | --------------------------------------------------- |
| `NEXTRACT_MODEL`                 | `openai:gpt-4o` | Pydantic AI **model string** (`provider:model-id`). |
| `NEXTRACT_MAX_CONCURRENCY`       | `4`             | Max parallel Agent calls in `batch_extract`.        |
| `NEXTRACT_MAX_RUN_RETRIES`       | `5`             | Max retry attempts around Agent runs.               |
| `NEXTRACT_PER_CALL_TIMEOUT_SECS` | `120`           | Per-call timeout in seconds.                        |
| `NEXTRACT_PRICING`               | *(unset)*       | JSON map for cost estimation (see below).           |

> Also set **provider credentials** as expected by Pydantic AI for your chosen provider.
> Example: for OpenAI: `OPENAI_API_KEY=...`

### Pricing configuration

`NEXTRACT_PRICING` expects a JSON string like:

```json
{
  "openai:gpt-4o": { "input_per_1k": 0.005, "output_per_1k": 0.015 },
  "openai:gpt-4.1-mini": { "input_per_1k": 0.003, "output_per_1k": 0.006 }
}
```

This is used to compute `cost_estimate_usd` from the Agent’s token **usage**. If the current model is missing in this map, cost will be `null`.

### Model selection

By default, `nextract` uses `openai:gpt-4o` (vision-capable). You can override per process:

```bash
export NEXTRACT_MODEL="provider:model-id"
```

Or in Python, construct and pass your own `RuntimeConfig` (advanced—optional).

---

## How it works

1. **You pass file paths** (single or multiple).
2. `nextract` prepares content:

   * **Textual** files are read and injected as-is into the prompt, wrapped by:

     ```
     --- BEGIN FILE: <name> (mime) ---
     <file contents>
     --- END FILE: <name> ---
     ```
   * **Binary** files (PDFs, images, Office docs, others) are attached as **binary parts** using Pydantic AI’s `BinaryContent`.
3. An **Agent** is created with:

   * a **system prompt** that instructs strict, schema-aligned extraction,
   * an `output_type` of either:

     * **StructuredDict(JSON Schema)** → outputs a dict, or
     * **Your Pydantic Model** → outputs a model instance (dumped to dict by default).
4. For JSON Schema mode, a **jsonschema validator** runs as an **output validator**. On failure, the Agent is asked to **retry** briefly (limited rounds).
5. The result is **validated** again before returning. You get:

   * `data` (dict by default),
   * `report` with usage and optional cost estimate.

---

## File type handling

* **Text**: `.txt`, `.md`, `.json`, `.yaml`, `.yml`, `.xml`, `.csv`, `.tsv`, `.html`, `.htm`
  → Read as text (UTF‑8 with fallback), injected verbatim with file delimiters.
* **Excel**: `.xlsx` (parsed to TSV via in‑process XML), `.xls` (CSV via CLI if available; else raw fallback)
  → Read as text and injected like other textual files. Best‑effort extraction (no styling/formatting).
* **PDF / Images**: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tiff`
  → Attached as **binary bytes** for the model (vision-capable models recommended).
* **Office docs**: `.doc`, `.docx`, `.ppt`, `.pptx`
  → Converted to **PDF** via LibreOffice/soffice or unoconv if available; on failure, attached as original binary.
* **ZIP**: Extracted to `/tmp/nextract-zip-<name>`; each inner file is processed as above. No nested recursion.

> **No OCR**: Scanned PDFs/images are **not** OCR’d. If the model can’t read them natively, fields may be missing.

---

## Office → PDF conversion

`nextract` attempts to convert `.doc/.docx/.ppt/.pptx` to PDF using system tools. These are external dependencies and are not installed via pip.

- Preferred: `soffice` (LibreOffice) in headless mode.
- Fallback: `unoconv` (uses LibreOffice UNO).

Installation hints:

- macOS (Homebrew):
  - `brew install --cask libreoffice`
  - Ensure `soffice` is on your `PATH`. If not, you can symlink:
    - `ln -s "/Applications/LibreOffice.app/Contents/MacOS/soffice" /usr/local/bin/soffice`  (adjust for Apple Silicon/Homebrew prefix)
- Ubuntu/Debian:
  - `sudo apt-get update && sudo apt-get install -y libreoffice`
  - Optional: `sudo apt-get install -y unoconv`
- Fedora/CentOS/RHEL:
  - `sudo dnf install -y libreoffice`  (or `yum` on older systems)
  - Optional: install `unoconv` from your distro repos if available.
- Windows:
  - Install LibreOffice from https://www.libreoffice.org/download/ and add `soffice.exe` to your `PATH`.

If neither tool is found, `nextract` logs a warning and falls back to attaching the original Office binary.

## Examples & Few-shot Hints

You can supply **examples** to guide the model:

**Programmatic** (`examples` argument):

* Output-only examples: `list[dict]`
* Paired input/output: `list[tuple[str | None, dict]]`

**CLI** (`--examples` JSON file):

* Output-only examples:

  ```json
  [
    { "invoice_number": "INV-001", "total": 123.45 }
  ]
  ```
* Paired input/output (use a two-element **array**):

  ```json
  [
    ["Item: Widget A, Total: 123.45", { "invoice_number": "INV-001", "total": 123.45 }]
  ]
  ```

**“Extra” fields** (JSON Schema mode):
If you pass `include_extra=True`, your schema is augmented with a top-level:

```json
"extra": { "type": "object", "additionalProperties": true }
```

so the model can place relevant-but-unspecified fields there.

---

## Return shape

All entry points return a **dict** with this structure:

```json
{
  "data": { /* your structured result (dict by default) */ },
  "report": {
    "model": "provider:model-id",
    "files": ["..."],
    "usage": {
      "requests": 1,
      "tool_calls": 0,
      "input_tokens": 123,
      "output_tokens": 456,
      "details": { /* provider-dependent */ }
    },
    "cost_estimate_usd": 0.0123,
    "warnings": []
  }
}
```

* In **Pydantic model** mode, `data` is still a dict **unless** you passed `return_pydantic=True`, in which case it’s the **model instance**.

---

## Logging & Tracing

* Uses **structlog**; logs are JSON-formatted to stdout.
* Each extraction logs: model, files, usage, warnings, and cost estimate.
* You can set up your own logging before calling `extract`/`batch_extract`. By default, the library initializes logging for you (toggle via `setup_logs=False`).

---

## Retries, Rate Limits & Timeouts

* Each Agent call is wrapped with **exponential backoff** (max attempts from `NEXTRACT_MAX_RUN_RETRIES`).
* **Timeout** per call is `NEXTRACT_PER_CALL_TIMEOUT_SECS` (default 120s).
* In **batch** mode, up to `NEXTRACT_MAX_CONCURRENCY` tasks run in parallel (default 4).

---

## Large files (TODO)

Planned design (not implemented in this build):

* **Chunking** (semantic/page) for large inputs.
* Per-chunk extraction with **all fields optional**, then **merge** into a full model validated against the target schema/model.
* Pluggable **conflict resolution** & optional provenance.

---

## Limitations

* No **OCR**, no **readability parsing** for HTML.
* Office conversions require `soffice` (LibreOffice) or `unoconv` installed; otherwise we fall back to attaching the original binary.
* Office file understanding depends on the model/provider.
* Very large inputs may exceed model or provider limits.
* ZIP extraction writes to `/tmp/nextract-zip-<name>`; these temp files are **not auto-deleted** by the library.

---

## FAQ

**Q: Which providers/models can I use?**
A: Any supported by **Pydantic AI Agent**. Select via `NEXTRACT_MODEL="provider:model-id"` and set the provider’s expected credentials (e.g., `OPENAI_API_KEY`).

**Q: What happens if schema validation keeps failing?**
A: The Agent is asked to retry a couple of times. Final results are validated once more; if still invalid, you’ll see a `final_validation_error` entry under `report.warnings`.

**Q: Can I store or inspect attachments that `nextract` sends?**
A: This build sends raw text or binary bytes directly to the Agent. If you need durable storage or redaction, wrap `nextract` in your own pipeline.

**Q: Can I get a Pydantic model out?**
A: Yes—pass your model class to `schema_or_model` and set `return_pydantic=True`.

---

## Development

### Building & Testing

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
ruff check nextract

# Build package
python -m build

# Test installation
pip install dist/*.whl
```

### CI/CD

This project uses GitHub Actions for continuous integration and automated PyPI publishing:

- **CI**: Runs on every push/PR, testing across Python 3.10-3.12
- **Release**: Automatically publishes to PyPI when GitHub releases are created
- **Versioning**: Managed statically in `pyproject.toml` (current: `0.0.1`)

### Creating a Release

1. Bump `version` in `pyproject.toml`, commit, and push.
2. Create a GitHub release (with notes) — this triggers automatic PyPI publishing.

---

## License

**MIT**. Feel free to adapt and extend.

---

## Project Structure (for reference)

```
nextract/
  ├─ nextract/
  │  ├─ __init__.py            # exports extract, batch_extract
  │  ├─ version.py
  │  ├─ config.py              # RuntimeConfig (model, concurrency, timeouts, pricing)
  │  ├─ logging.py             # structlog setup
  │  ├─ mimetypes_map.py       # simple mapping & helpers
  │  ├─ schema.py              # JSON Schema/Pydantic utilities
  │  ├─ prompts.py             # system prompt + examples builder
  │  ├─ files.py               # read-as-is; BinaryContent or text
  │  ├─ pricing.py             # usage → cost estimate
  │  ├─ agent_runner.py        # Agent wiring, retries, validation, metrics
  │  ├─ core.py                # public API: extract, batch_extract
  │  └─ cli.py                 # Typer CLI
  └─ pyproject.toml
```
