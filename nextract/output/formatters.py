from __future__ import annotations

import csv
import html
import json
from io import StringIO
from pathlib import Path
from typing import Any

from nextract.core import BaseFormatter, ExtractionResult

_HTML_TEMPLATE: str | None = None


def _load_html_template() -> str:
    """Load the HTML template from the companion template file."""
    global _HTML_TEMPLATE
    if _HTML_TEMPLATE is None:
        template_path = Path(__file__).parent / "html_template.html"
        _HTML_TEMPLATE = template_path.read_text(encoding="utf-8")
    return _HTML_TEMPLATE


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for non-standard types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


class JsonFormatter(BaseFormatter):
    """Format extraction results as JSON."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        indent = kwargs.get("indent", 2)
        return json.dumps(result.data, ensure_ascii=False, indent=indent, default=_json_default)


class MarkdownFormatter(BaseFormatter):
    """Format extraction results as Markdown."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        payload = json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
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
        theme = str(kwargs.get("theme") or "system").lower()
        if theme not in {"light", "dark", "system"}:
            theme = "system"

        payload = json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
        payload_html = html.escape(payload)
        payload_script = (
            json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

        template = _load_html_template()

        return (
            template
            .replace("__NEXTRACT_THEME__", theme)
            .replace("__NEXTRACT_PAYLOAD_HTML__", payload_html)
            .replace("__NEXTRACT_PAYLOAD_JSON__", payload_script)
        )


class CsvFormatter(BaseFormatter):
    """Format extraction results as CSV."""

    @staticmethod
    def _safe_csv_cell(value: object) -> object:
        """Sanitize cell value to prevent CSV formula injection.

        Prefix formula-injection markers Excel may interpret (=, +, @, -, tab, CR).
        Legitimate numeric literals (including negatives like -12.5) are left alone.
        Hybrids such as ``-1+cmd`` are still sanitized.
        """
        import re

        s = str(value) if value is not None else ""
        if not s:
            return s
        # Full numeric literal: optional leading -, digits, optional fraction/exponent.
        if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", s):
            return s
        if s[0] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + s
        return s

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        data = result.data
        flatten_nested = kwargs.get("flatten_nested", True)
        output = StringIO()
        writer = csv.writer(output)

        if isinstance(data, list):
            rows = []
            for row in data:
                if isinstance(row, dict):
                    rows.append(self._flatten_dict(row) if flatten_nested else row)
                else:
                    rows.append({"value": row})
            if not rows:
                return ""
            headers = sorted({key for row in rows for key in row.keys()})
            writer.writerow([self._safe_csv_cell(h) for h in headers])
            for row in rows:
                writer.writerow([self._safe_csv_cell(row.get(header, "")) for header in headers])
            return output.getvalue()

        if isinstance(data, dict):
            flat = self._flatten_dict(data) if flatten_nested else data
            writer.writerow(["field", "value"])
            for key, value in flat.items():
                writer.writerow([self._safe_csv_cell(key), self._safe_csv_cell(value)])
            return output.getvalue()

        writer.writerow(["value"])
        writer.writerow([self._safe_csv_cell(data)])
        return output.getvalue()

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten nested dicts using dotted keys."""
        items: dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(CsvFormatter._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, list):
                items[new_key] = json.dumps(v, ensure_ascii=False, default=_json_default)
            else:
                items[new_key] = v
        return items
