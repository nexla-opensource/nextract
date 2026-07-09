from __future__ import annotations

from pathlib import Path

import typer

from nextract.ingest import DocumentValidator, load_documents
from nextract.mimetypes_map import is_image, is_office_binary, is_pdf, is_textual
from nextract.output import CsvFormatter, HtmlFormatter, JsonFormatter, MarkdownFormatter
from nextract.parse import extract_text

app = typer.Typer(
    add_completion=False,
    help=(
        "Format extracted document text as Markdown, HTML, CSV, or JSON "
        "(not layout-preserving conversion)"
    ),
)

_SUPPORTED_FORMATS = frozenset({"markdown", "html", "csv", "json"})
_SUPPORTED_THEMES = frozenset({"light", "dark", "system"})


@app.command(
    "convert",
    help=(
        "Format extracted document text as Markdown, HTML, CSV, or JSON. "
        "Extracts text then formats it — not full layout-preserving document conversion."
    ),
)
def cli_convert(
    document: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input document"),
    output_format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, html, csv, or json"
    ),
    theme: str = typer.Option("system", "--theme", help="Theme for HTML output: light, dark, or system"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file instead of stdout"),
) -> None:
    try:
        formatter = _select_formatter(output_format)
        validated_theme = _validate_theme(theme)

        artifacts = load_documents([document])
        if not artifacts:
            raise typer.BadParameter("No documents provided")

        validator = DocumentValidator()
        for artifact in artifacts:
            validation = validator.validate(artifact)
            if not validation.valid:
                _fail_validation(validation.errors)

        text = extract_text(artifacts[0])
        if text is None or (isinstance(text, str) and not text.strip()):
            typer.echo(_empty_text_message(Path(artifacts[0].source_path)), err=True)
            raise typer.Exit(code=1)

        result_payload = formatter.format(_build_result(text), theme=validated_theme)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result_payload)
            typer.echo(f"Wrote {output}", err=True)
        else:
            typer.echo(result_payload)
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Conversion failed: {exc}", err=True)
        raise typer.Exit(code=1)


def _select_formatter(output_format: str):
    name = output_format.lower()
    if name not in _SUPPORTED_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise typer.BadParameter(
            f"Unknown format '{output_format}'. Supported formats: {supported}",
            param_hint="--format",
        )
    if name == "markdown":
        return MarkdownFormatter()
    if name == "html":
        return HtmlFormatter()
    if name == "csv":
        return CsvFormatter()
    return JsonFormatter()


def _validate_theme(theme: str) -> str:
    name = theme.lower()
    if name not in _SUPPORTED_THEMES:
        supported = ", ".join(sorted(_SUPPORTED_THEMES))
        raise typer.BadParameter(
            f"Invalid theme '{theme}'. Supported themes: {supported}",
            param_hint="--theme",
        )
    return name


def _fail_validation(errors: list[str]) -> None:
    message = "; ".join(errors)
    lower = message.lower()
    if "encrypted" in lower or "password" in lower:
        message = (
            f"{message}. Provide an unencrypted copy or decrypt the PDF first "
            "(password-protected PDFs are not supported by convert)."
        )
    typer.echo(f"Conversion failed: document validation error: {message}", err=True)
    raise typer.Exit(code=1)


def _empty_text_message(path: Path) -> str:
    """Build an actionable error when text extraction yields nothing."""
    parts = [
        "Conversion failed: extracted text is empty.",
        "Convert formats extracted text only (Markdown/HTML/CSV/JSON), "
        "not layout-preserving conversion.",
    ]
    specific = False
    if is_textual(path):
        parts.append(
            "The file appears empty or contains only whitespace; "
            "provide a document with readable text content."
        )
        specific = True
    if is_office_binary(path):
        parts.append(
            "For Office documents (DOCX/PPTX/XLSX), install LibreOffice (`soffice`) or unoconv."
        )
        specific = True
    if is_pdf(path):
        parts.append(
            "For scanned/image PDFs, install Tesseract OCR (`tesseract`) and pytesseract."
        )
        specific = True
    if is_image(path):
        parts.append(
            "Images: convert extracts text only and does not OCR images; "
            "use `nextract extract` with a VLM extractor (e.g. --extractor vlm) for images."
        )
        specific = True
    if not specific:
        parts.append(
            "For Office documents (DOCX/PPTX/XLSX), install LibreOffice (`soffice`) or unoconv. "
            "For scanned/image PDFs, install Tesseract OCR (`tesseract`) and pytesseract. "
            "For images, use `nextract extract` with a VLM extractor."
        )
    return " ".join(parts)


def _build_result(text: str):
    from nextract.core import ExtractionResult

    return ExtractionResult(data={"content": text}, metadata={})
