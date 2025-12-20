from __future__ import annotations

import csv
import html
import json
from io import StringIO
from typing import Any

from nextract.core import BaseFormatter, ExtractionResult


class JsonFormatter(BaseFormatter):
    """Format extraction results as JSON."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        indent = kwargs.get("indent", 2)
        return json.dumps(result.data, ensure_ascii=False, indent=indent)


class MarkdownFormatter(BaseFormatter):
    """Format extraction results as Markdown."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        payload = json.dumps(result.data, ensure_ascii=False, indent=2)
        return "\n".join(
            [
                "# Extraction Result",
                "",
                "```json",
                payload,
                "```",
            ]
        )


class HtmlFormatter(BaseFormatter):
    """Format extraction results as HTML."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        payload = html.escape(json.dumps(result.data, ensure_ascii=False, indent=2))
        return f"<html><body><pre>{payload}</pre></body></html>"


class CsvFormatter(BaseFormatter):
    """Format extraction results as CSV."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        data = result.data
        output = StringIO()
        writer = csv.writer(output)

        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
            if not rows:
                return ""
            headers = sorted({key for row in rows for key in row.keys()})
            writer.writerow(headers)
            for row in rows:
                writer.writerow([row.get(header, "") for header in headers])
            return output.getvalue()

        if isinstance(data, dict):
            writer.writerow(["field", "value"])
            for key, value in data.items():
                writer.writerow([key, json.dumps(value, ensure_ascii=False)])
            return output.getvalue()

        writer.writerow(["value"])
        writer.writerow([data])
        return output.getvalue()
