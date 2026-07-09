"""Unit tests for output formatters."""

from __future__ import annotations

from nextract.core import ExtractionResult
from nextract.output.formatters import CsvFormatter, HtmlFormatter


class TestCsvSafeCell:
    """CSV formula-injection sanitization."""

    def test_negative_numbers_not_mangled(self) -> None:
        assert CsvFormatter._safe_csv_cell(-12.5) == "-12.5"
        assert CsvFormatter._safe_csv_cell("-12.5") == "-12.5"
        assert CsvFormatter._safe_csv_cell("-0") == "-0"
        assert CsvFormatter._safe_csv_cell("-999") == "-999"
        assert CsvFormatter._safe_csv_cell(-42) == "-42"

    def test_formula_prefixes_sanitized(self) -> None:
        assert CsvFormatter._safe_csv_cell("=1+1") == "'=1+1"
        assert CsvFormatter._safe_csv_cell("+cmd") == "'+cmd"
        assert CsvFormatter._safe_csv_cell("@SUM(A1)") == "'@SUM(A1)"
        assert CsvFormatter._safe_csv_cell("\tformula") == "'\tformula"
        assert CsvFormatter._safe_csv_cell("\rcmd") == "'\rcmd"

    def test_dash_formula_still_sanitized(self) -> None:
        """Leading '-' that is not a pure number should still be sanitized."""
        assert CsvFormatter._safe_csv_cell("-@SUM(A1)") == "'-@SUM(A1)"
        assert CsvFormatter._safe_csv_cell("-=1+1") == "'-=1+1"
        assert CsvFormatter._safe_csv_cell("-") == "'-"
        assert CsvFormatter._safe_csv_cell("-abc") == "'-abc"
        # Digit-prefixed formula hybrids must not bypass sanitization.
        assert CsvFormatter._safe_csv_cell("-1+cmd") == "'-1+cmd"
        assert CsvFormatter._safe_csv_cell("-2|cmd") == "'-2|cmd"
        assert CsvFormatter._safe_csv_cell("-3=HYPERLINK()") == "'-3=HYPERLINK()"

    def test_plain_values_unchanged(self) -> None:
        assert CsvFormatter._safe_csv_cell("hello") == "hello"
        assert CsvFormatter._safe_csv_cell(42) == "42"
        assert CsvFormatter._safe_csv_cell("") == ""
        assert CsvFormatter._safe_csv_cell(None) == ""

    def test_format_preserves_negative_numbers_in_output(self) -> None:
        result = ExtractionResult(
            data=[{"amount": -12.5, "label": "refund"}, {"amount": 10, "label": "=evil"}],
            metadata={},
        )
        csv_text = CsvFormatter().format(result)
        assert "-12.5" in csv_text
        assert "'-12.5" not in csv_text
        assert "'=evil" in csv_text


class TestHtmlTheme:
    def test_valid_themes_applied(self) -> None:
        result = ExtractionResult(data={"x": 1}, metadata={})
        for theme in ("light", "dark", "system"):
            html = HtmlFormatter().format(result, theme=theme)
            assert f'data-theme="{theme}"' in html or theme in html
