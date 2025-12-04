# Repository Guidelines

## Project Structure & Modules
- `nextract/`: core library and CLI
  - `core.py` (public API: `extract`, `batch_extract`), `agent_runner.py` (agent wiring), `files.py` (file handling), `schema.py` (JSON Schema/Pydantic utils), `pricing.py`, `config.py`, `prompts.py`, `logging.py`, `cli.py` (Typer app)
- `usage/`: runnable examples (e.g., `python usage/pydantic_usage.py`)
- `pyproject.toml`: packaging, deps, console script `nextract`
- `dist/`: build artifacts (wheel/sdist)

## Build, Test, and Development
- Create env and install dev deps: `pip install -e .[dev]`
- Run CLI locally:
  - After install: `nextract --help`
  - Without install: `python -m nextract.cli --help`
- Tests: `pytest`
- Lint: `ruff check nextract`
- Type check (optional but encouraged): `mypy nextract`
- Build package: `python -m build`

## Coding Style & Naming
- Python 3.10+; 4‑space indentation; UTF‑8.
- Use type hints for all public functions. Prefer small, pure helpers.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Logging via `structlog.get_logger(__name__)`; avoid print.
- Run `ruff check` before pushing; `ruff format` is acceptable if you use it.

## Testing Guidelines
- Framework: `pytest` (with `pytest-mock` as needed).
- Location: create `tests/` at repo root; files named `test_*.py`.
- Focus: unit tests for `files.py`, `schema.py`, `agent_runner.py`, and CLI argument parsing. Use temp dirs for I/O and mock network/LLM calls.
- Aim for meaningful coverage with clear arrange‑act‑assert structure.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject line; include a short body when helpful. Example: `Add JSON Schema validator retries in agent_runner`.
- PRs: small and focused; include description, linked issues, and before/after examples (CLI or code). Update `README.md` and `usage/` when user‑facing behavior changes.
- CI: ensure `ruff` and `pytest` pass locally before opening a PR.

## Security & Configuration Tips
- Do not commit secrets. Use environment variables; `.env` is for local only. Mirror new keys in `.env.example`.
- Key envs: `NEXTRACT_MODEL`, `NEXTRACT_MAX_CONCURRENCY`, `NEXTRACT_MAX_RUN_RETRIES`, `NEXTRACT_PER_CALL_TIMEOUT_SECS`, `NEXTRACT_PRICING`.
- Office→PDF conversion requires system tools (`soffice`/LibreOffice or `unoconv`) if you touch those paths.

## Agent‑Specific Notes
- Keep changes minimal and targeted; do not reformat unrelated files.
- Follow this guide’s naming, typing, and lint rules for any new or modified code.

