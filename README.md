# nextract

Extract structured data from documents (text, PDFs, images, spreadsheets) using LLMs, OCR, and schema-guided extraction.

This package mirrors the design of langextract while extending support to more document types and providers.

## Features

- Unified `extract()` API for files, directories, or in-memory streams
- Schema-driven extraction with provider-specific adaptation (OpenAI, Gemini, Ollama)
- OCR support for PDFs and images (PyMuPDF, Tesseract)
- Intelligent chunking for large documents
- Few-shot examples to guide complex extractions
- JSONL save/load and simple HTML visualization

## Installation

```bash
pip install -e .
```

This project depends on optional native components (PyMuPDF, Tesseract). You may need:
- macOS: `brew install tesseract`
- Linux: `apt-get install tesseract-ocr`

## Quick Start

```python
import nextract as nx

# Example 1: Simple extraction from a text file with a prompt
result = nx.extract(
    documents="invoice.txt",
    prompt="Extract the invoice number, date, vendor name, and total amount.",
    model_id="gemini-2.5-flash"
)

# Example 2: Extraction using a predefined JSON Schema from a PDF
invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "vendor": {"type": "string"},
        "total": {"type": "number"}
    },
    "required": ["invoice_number", "date", "vendor", "total"]
}

result = nx.extract(
    documents="scanned_invoice.pdf",
    schema=invoice_schema,
    model_id="gemini-1.5-pro",
    use_ocr=True
)

# Example 3: Using few-shot examples for a complex extraction from an image
examples = [
    nx.ExampleData(
        document="sample_receipt_1.jpg",
        extractions=[
            nx.Extraction(
                extraction_class="item",
                extraction_text="Coffee",
                attributes={"price": "3.50", "quantity": "1"}
            ),
            nx.Extraction(
                extraction_class="item",
                extraction_text="Bagel",
                attributes={"price": "2.75", "quantity": "1"}
            ),
            nx.Extraction(
                extraction_class="total",
                extraction_text="$6.25"
            )
        ]
    )
]

result = nx.extract(
    documents="new_receipt.jpg",
    prompt="Extract all items with their prices and quantities, and the final total.",
    examples=examples,
    model_id="gpt-4o"
)
```

## API

```python
nx.extract(
    documents,                       # file path, list, directory, or BytesIO
    schema=None,                     # dict|str|Path|BaseSchema
    prompt=None,                     # natural language instruction
    examples=None,                   # list[ExampleData]
    model_id="gemini-2.5-flash",
    provider=None,                   # override provider
    api_key=None,                    # provider-specific api key
    extraction_passes=1,             # multiple passes to improve recall
    max_workers=10,                  # parallel workers
    max_char_buffer=2000,            # chunk size per prompt
    temperature=0.0,
    use_ocr=None,                    # default True for pdf/image
    ocr_provider="pymupdf",
    fence_output=None,               # auto by provider & schema
    debug=False
)
```

Return type: list of `AnnotatedDocument` with extractions and metadata. See `nextract/core/data.py`.

## Providers and API Keys

- Gemini: set `GEMINI_API_KEY`
- OpenAI: set `OPENAI_API_KEY`
- Ollama: local models via the `ollama` HTTP API (no key by default)

You can also pass `api_key` directly to `extract()`.

## Saving and Visualization

```python
from nextract.io import save_jsonl, load_jsonl
from nextract.visualization import visualize

save_jsonl(result, "extractions.jsonl")
docs = load_jsonl("extractions.jsonl")
html = visualize(docs[0])
```

## Testing

```bash
pytest -q
```

## License

Apache 2.0
