from __future__ import annotations

import typer

from nextract.cli.commands import batch as batch_cmd
from nextract.cli.commands import check_provider as check_provider_cmd
from nextract.cli.commands import convert as convert_cmd
from nextract.cli.commands import extract as extract_cmd
from nextract.cli.commands import listing as listing_cmd
from nextract.cli.commands import suggest_schema as suggest_schema_cmd
from nextract.cli.commands import validate_config as validate_cmd

app = typer.Typer(add_completion=False, help="nextract — intelligent document extraction")

app.add_typer(extract_cmd.app)
app.add_typer(batch_cmd.app)
app.add_typer(listing_cmd.app, name="list")
app.add_typer(validate_cmd.app)
app.add_typer(convert_cmd.app)
app.add_typer(suggest_schema_cmd.app)
app.add_typer(check_provider_cmd.app)
