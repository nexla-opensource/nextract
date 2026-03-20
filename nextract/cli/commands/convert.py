from __future__ import annotations

from pathlib import Path

import typer

from nextract.ingest import load_documents
from nextract.output import CsvFormatter, HtmlFormatter, JsonFormatter, MarkdownFormatter
from nextract.parse import extract_text

app = typer.Typer(add_completion=False)


@app.command("convert")
def cli_convert(
    document: Path = typer.Argument(..., exists=True, readable=True),
    output_format: str = typer.Option("markdown", "--format", "-f"),
    theme: str = typer.Option("system", "--theme", help="Theme for HTML output: light, dark, or system"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    try:
        artifacts = load_documents([document])
        if not artifacts:
            raise typer.BadParameter("No documents provided")

        text = extract_text(artifacts[0])
        if text is None:
            text = ""

        formatter = _select_formatter(output_format)
        result_payload = formatter.format(_build_result(text), theme=theme)

        if output:
            output.write_text(result_payload)
        else:
            typer.echo(result_payload)
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Conversion failed: {exc}", err=True)
        raise typer.Exit(code=1)


def _select_formatter(output_format: str):
    name = output_format.lower()
    if name == "markdown":
        return MarkdownFormatter()
    if name == "html":
        return HtmlFormatter()
    if name == "csv":
        return CsvFormatter()
    return JsonFormatter()


def _build_result(text: str):
    from nextract.core import ExtractionResult

    return ExtractionResult(data={"content": text}, metadata={})
