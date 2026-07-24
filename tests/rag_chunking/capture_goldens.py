#!/usr/bin/env python
"""Capture golden outputs + LLM cassettes from the ORIGINAL monolith.

Runs the original production monolith (chunking_code.py) against a set of
fixture files, recording every LLM interaction into a "cassette" and every
output DataFrame into a golden JSON file. The migrated nextract.rag_chunking package
can then be parity-tested offline by replaying the cassettes and comparing
its outputs against the goldens.

Requires a real Gemini API key (GOOGLE_API_KEY / GEMINI_API_KEY or
--api-key). Use --dry-run to verify the monolith loads and the pipeline
constructs without making any API calls.

Example:
    RAG_CHUNKING_MONOLITH=/path/to/chunking_code.py \
        python tests/rag_chunking/capture_goldens.py --api-key "$GOOGLE_API_KEY"
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The monolith is not part of this repo; point at your local copy via env or --monolith.
DEFAULT_MONOLITH = os.getenv("RAG_CHUNKING_MONOLITH", "")
DEFAULT_OUT = str(HERE / "goldens")
FIXTURES_DIR = HERE / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "loan-extraction.pdf"

# Capture-only timeout headroom: transient live timeouts (e.g. a slow
# structured-tables batch) would otherwise fail-open into silently degraded
# goldens. Timeouts never enter cassette keys and replay ignores them, so
# goldens captured with headroom remain valid under the package defaults.
CAPTURE_TIMEOUT_STANDARD = 300
CAPTURE_TIMEOUT_LARGE = 600


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Capture golden outputs + LLM cassettes from the original monolith."
    )
    p.add_argument(
        "--monolith",
        default=DEFAULT_MONOLITH,
        help="Path to the monolith chunking_code.py "
        "(default: $RAG_CHUNKING_MONOLITH; required if the env var is unset)",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory for goldens + cassettes (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--fixtures",
        nargs="*",
        default=None,
        help=(
            "Fixture files to process (default: all files in "
            f"{FIXTURES_DIR} plus {SAMPLE_PDF})"
        ),
    )
    p.add_argument(
        "--allow-degraded",
        action="store_true",
        default=False,
        help="Accept goldens containing fail-open/degraded rows (error chunks, "
        "ERROR: sentinels). By default such captures exit nonzero: degraded "
        "goldens are a bad parity baseline — re-run the capture instead.",
    )
    p.add_argument(
        "--no-vertex",
        action="store_true",
        default=False,
        help="Swap in a plain developer-key genai.Client for ordinary Gemini API "
        "keys (default: keep the monolith's vertexai=True express-mode client, "
        "matching production)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (default: GOOGLE_API_KEY or GEMINI_API_KEY env var)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Stop after loading the monolith and constructing the pipeline "
        "(no fixtures processed, no API calls)",
    )
    return p.parse_args(argv)


def resolve_api_key(args):
    key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit(
            "ERROR: no API key. Pass --api-key or set GOOGLE_API_KEY / "
            "GEMINI_API_KEY in the environment."
        )
    return key


def default_fixtures():
    fixtures = []
    if FIXTURES_DIR.is_dir():
        fixtures.extend(sorted(p for p in FIXTURES_DIR.iterdir() if p.is_file()))
    else:
        print(f"WARNING: fixtures dir not found: {FIXTURES_DIR}", file=sys.stderr)
    if SAMPLE_PDF.is_file():
        fixtures.append(SAMPLE_PDF)
    else:
        print(f"WARNING: sample PDF not found: {SAMPLE_PDF}", file=sys.stderr)
    return fixtures


def load_monolith(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"ERROR: monolith not found: {path}")
    spec = importlib.util.spec_from_file_location("chunking_monolith", str(path))
    if spec is None or spec.loader is None:
        sys.exit(f"ERROR: could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunking_monolith"] = mod
    spec.loader.exec_module(mod)
    # The monolith calls logging.basicConfig(level=DEBUG) at import time; tame it.
    logging.getLogger().setLevel(logging.INFO)
    return mod


def serialize_df(df):
    # to_json handles NaN -> null and nested dicts/lists inside cells.
    return {
        "columns": list(df.columns),
        "records": json.loads(df.to_json(orient="records")),
    }


DEGRADED_MARKERS = (
    '"chunker_result": "Fail"',
    "ERROR:",
    "Processing failed",
    "Event loop is closed",
    "Cassette miss",
)


def degraded_rows(outputs) -> list:
    """Rows whose serialized content shows fail-open/degraded output.

    A golden containing such rows replays consistently, but it enshrines a
    transient live failure as the parity baseline — reject by default.
    """
    hits = []
    for out_name, o in outputs.items():
        for i, record in enumerate(o["records"]):
            blob = json.dumps(record, default=str)
            for marker in DEGRADED_MARKERS:
                if marker in blob:
                    hits.append(f"{out_name} records[{i}]: contains {marker!r}")
                    break
    return hits


class _LoopChurnRetryShim:
    """Capture-only mitigation for the monolith's asyncio.run loop churn.

    The pipeline's handlers call asyncio.run() repeatedly against one cached
    genai client; its aio transport can bind to an already-closed loop, and the
    first async call on a fresh loop then fails instantly with 'Event loop is
    closed' (raised by agen_text; returned as an 'ERROR: ...' sentinel string
    by the vision methods). The monolith treats this as non-retryable and
    fail-opens a degraded row — a race we must not bake into parity goldens.
    On that specific signature, rebuild the client and retry the call once.
    The monolith stays verbatim; the packaged fix is tracked in the bug
    register (retry classification + loop handling).
    """

    def __init__(self, inner, rebuild_client):
        self._inner = inner
        self._rebuild = rebuild_client

    @staticmethod
    def _is_loop_churn_sentinel(resp) -> bool:
        return (
            isinstance(resp, str)
            and resp.startswith("ERROR:")
            and "event loop is closed" in resp.lower()
        )

    async def _call(self, fn, *args, **kwargs):
        try:
            resp = await fn(*args, **kwargs)
        except RuntimeError as exc:
            if "event loop is closed" not in str(exc).lower():
                raise
            self._rebuild()
            return await fn(*args, **kwargs)
        if self._is_loop_churn_sentinel(resp):
            self._rebuild()
            return await fn(*args, **kwargs)
        return resp

    async def agen_text(self, *args, **kwargs):
        return await self._call(self._inner.agen_text, *args, **kwargs)

    async def agen_vision(self, *args, **kwargs):
        return await self._call(self._inner.agen_vision, *args, **kwargs)

    async def agen_vision_batch(self, *args, **kwargs):
        return await self._call(self._inner.agen_vision_batch, *args, **kwargs)

    def count_tokens(self, text):
        return self._inner.count_tokens(text)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run:
        api_key = args.api_key or "dry"
    else:
        api_key = resolve_api_key(args)

    if not args.monolith:
        sys.exit(
            "ERROR: no monolith path. Set RAG_CHUNKING_MONOLITH or pass --monolith "
            "(the production chunking_code.py is not part of this repo)."
        )

    mono = load_monolith(args.monolith)
    cfg = mono.Config()
    # Capture-only headroom (see module docstring constants): avoid transient
    # timeouts fail-opening into degraded goldens. Not a behavior change for
    # replay — timeouts never enter cassette keys.
    cfg.GEMINI_TIMEOUT_STANDARD = max(cfg.GEMINI_TIMEOUT_STANDARD, CAPTURE_TIMEOUT_STANDARD)
    cfg.GEMINI_TIMEOUT_LARGE = max(cfg.GEMINI_TIMEOUT_LARGE, CAPTURE_TIMEOUT_LARGE)
    pipeline = mono.DocumentPipeline(cfg, api_key, None)

    if args.dry_run:
        print("DRY RUN OK")
        return 0

    if args.no_vertex:
        # The monolith hardcodes genai.Client(vertexai=True, ...). Swap in a
        # plain developer-key client so ordinary Gemini API keys work without
        # editing the monolith.
        from google import genai

        pipeline.llm.client = genai.Client(api_key=api_key)

    # Make the sibling cassette.py importable.
    sys.path.insert(0, str(HERE))
    from cassette import CassetteRecorder

    fixtures = [Path(f) for f in args.fixtures] if args.fixtures else default_fixtures()
    if not fixtures:
        sys.exit("ERROR: no fixtures to process.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_llm = pipeline.llm

    def _rebuild_client():
        from google import genai

        print("    [loop-churn] stale aio transport detected; rebuilding genai client, retrying once")
        base_llm.client = (
            genai.Client(api_key=api_key)
            if args.no_vertex
            else genai.Client(vertexai=True, api_key=api_key)
        )

    shimmed_llm = _LoopChurnRetryShim(base_llm, _rebuild_client)
    summary = []  # (fixture_name, status, n_outputs, n_rows, n_calls)
    any_failed = False

    for fixture in fixtures:
        name = fixture.name
        # Include the extension in the artifact name: fixtures like tiny.csv /
        # tiny.xlsx / tiny.png share a stem and would otherwise overwrite each other.
        stem = f"{fixture.stem}_{fixture.suffix.lstrip('.')}" if fixture.suffix else fixture.stem
        cassette_path = out_dir / f"{stem}.cassette.json"
        golden_path = out_dir / f"{stem}.golden.json"
        print(f"\n=== Capturing: {fixture} ===")
        try:
            if not fixture.is_file():
                raise FileNotFoundError(f"fixture not found: {fixture}")

            rec = CassetteRecorder(shimmed_llm, cassette_path)
            pipeline.llm = rec

            # process_file resets token counters internally per handler.
            results = pipeline.process_file(str(fixture), {"tags": {"display_path": name}})

            outputs = {out_name: serialize_df(df) for out_name, df in (results or {}).items()}
            golden_path.write_text(
                json.dumps({"fixture": name, "outputs": outputs}, indent=2, default=str)
            )
            rec.save()

            # Degraded detection, two layers:
            # (1) golden-row markers — fail-open rows that embed error text;
            # (2) cassette error entries — ANY recorded live failure means some
            #     pipeline stage fail-opened (possibly with no row marker at
            #     all, e.g. structured-tables batch failure just leaves
            #     structured_tables empty).
            degraded = degraded_rows(outputs)
            for err in rec.recorded_errors():
                degraded.append(
                    f"cassette: {err['method']} ({err['model']}) failed live: "
                    f"{err['error'][:120]}"
                )
            if degraded:
                print(f"    DEGRADED capture ({len(degraded)} issue(s)):")
                for hit in degraded:
                    print(f"      - {hit}")

            n_rows = sum(len(o["records"]) for o in outputs.values())
            n_calls = rec.entry_count()
            status = "DEGRADED" if degraded else "ok"
            summary.append((name, status, len(outputs), n_rows, n_calls))
            if degraded and not args.allow_degraded:
                # Quarantine so the parity gate's *.golden.json glob can never
                # pick up a rejected baseline.
                golden_path.rename(golden_path.with_suffix(".rejected.json"))
                cassette_path.rename(cassette_path.with_suffix(".rejected.json"))
                print("    quarantined rejected artifacts as *.rejected.json")
            else:
                print(f"    wrote {golden_path}")
                print(f"    wrote {cassette_path}")
        except Exception:
            traceback.print_exc()
            summary.append((name, "FAILED", 0, 0, 0))
        finally:
            pipeline.llm = base_llm

    print("\n=== Summary ===")
    print(f"{'fixture':<40} {'status':<8} {'outputs':>8} {'rows':>8} {'llm_calls':>10}")
    for name, status, n_outputs, n_rows, n_calls in summary:
        calls = "?" if n_calls < 0 else str(n_calls)
        print(f"{name:<40} {status:<8} {n_outputs:>8} {n_rows:>8} {calls:>10}")

    any_failed = any(status == "FAILED" for _, status, *_ in summary)
    any_degraded = any(status == "DEGRADED" for _, status, *_ in summary)
    if any_degraded and not args.allow_degraded:
        print(
            "\nDegraded goldens detected (transient live failures baked into "
            "outputs). Re-run the capture for those fixtures, or pass "
            "--allow-degraded to accept them anyway."
        )
        return 1
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
