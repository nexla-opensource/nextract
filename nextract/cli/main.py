from __future__ import annotations

import logging

import typer

from nextract import __version__
from nextract.cli.commands import batch as batch_cmd
from nextract.cli.commands import check_provider as check_provider_cmd
from nextract.cli.commands import convert as convert_cmd
from nextract.cli.commands import extract as extract_cmd
from nextract.cli.commands import listing as listing_cmd
from nextract.cli.commands import suggest_schema as suggest_schema_cmd
from nextract.cli.commands import validate_config as validate_cmd
from nextract.telemetry.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="nextract — intelligent document extraction",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nextract {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """nextract — intelligent document extraction.

    Structured logs are written to stderr so stdout JSON stays pipe-clean.
    """
    setup_logging(level=logging.INFO)


app.add_typer(extract_cmd.app)
app.add_typer(batch_cmd.app)
app.add_typer(listing_cmd.app, name="list", help="List extractors, providers, and chunkers")
app.add_typer(validate_cmd.app)
app.add_typer(convert_cmd.app)
app.add_typer(suggest_schema_cmd.app)
app.add_typer(check_provider_cmd.app)
