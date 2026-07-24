"""Offline parity gate: migrated package vs. monolith golden outputs.

For every ``goldens/*.golden.json`` with a sibling ``*.cassette.json`` this test
replays the recorded LLM responses through the migrated
``nextract.rag_chunking`` package and asserts the resulting
DataFrames match the golden byte-for-byte, modulo timing-related fields.

Goldens are captured against the original monolith by ``capture_goldens.py``
(requires a Gemini API key); this test itself is fully offline.
"""

import json
import math
import re
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

HERE = Path(__file__).resolve().parent
GOLDENS_DIR = HERE / "goldens"
FIXTURES_DIR = HERE / "fixtures"
REPO_ROOT = HERE.parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"

# Keys dropped (at any nesting depth, incl. debug_info / llm_stats) before
# comparison because they are timing-dependent and legitimately differ run-to-run.
TIMING_KEY_RE = re.compile(r"(^|_)(time|elapsed|duration|timestamp|wallclock)", re.I)

MAX_REPORTED_MISMATCHES = 20


def _discover_pairs():
    """Return sorted list of (golden_path, cassette_path) pairs."""
    if not GOLDENS_DIR.is_dir():
        return []
    pairs = []
    for golden in sorted(GOLDENS_DIR.glob("*.golden.json")):
        cassette = golden.with_name(golden.name[: -len(".golden.json")] + ".cassette.json")
        if cassette.is_file():
            pairs.append((golden, cassette))
    return pairs


PAIRS = _discover_pairs()

if not PAIRS:
    pytest.skip(
        "no goldens captured yet — run capture_goldens.py with a Gemini API key",
        allow_module_level=True,
    )

# The gate's membership is pinned: _discover_pairs silently drops any golden
# missing its cassette sibling, so a refresh/merge mishap could otherwise
# shrink the gate without failing anything.
EXPECTED_GATE_IDS = {"tiny_csv", "tiny_xlsx", "tiny_png", "loan-extraction_pdf"}

# Config reads GEMINI_* / CHUNKER_* env vars at instantiation; ambient values
# would change replay keying or pipeline routing. Replay must be hermetic.
CONFIG_ENV_VARS = (
    "GEMINI_MODEL",
    "GEMINI_LITE_MODEL",
    "GEMINI_METADATA_MODEL",
    "CHUNKER_USE_HEADING_DRIVEN",
    "CHUNKER_VERIFY_CHUNKS",
    "CHUNKER_STRUCTURED_TABLES",
)


def test_gate_membership():
    actual = {g.name[: -len(".golden.json")] for g, _ in PAIRS}
    assert actual == EXPECTED_GATE_IDS, (
        f"parity gate membership drifted: missing={EXPECTED_GATE_IDS - actual}, "
        f"unexpected={actual - EXPECTED_GATE_IDS} — a golden or its cassette "
        "sibling was lost/added without updating EXPECTED_GATE_IDS"
    )


@pytest.mark.parametrize(
    "cassette_path", sorted({c for _, c in PAIRS}), ids=lambda p: p.name
)
def test_cassettes_contain_no_recorded_failures(cassette_path):
    """A cassette error entry means the capture baked a live failure into the
    baseline (the pipeline fail-opens, sometimes with no row-level marker at
    all). Shipped baselines must come from fully-healthy captures."""
    entries = json.loads(cassette_path.read_text())
    errors = [
        f"{e.get('method')}({e.get('model')}): {e.get('error', '')[:100]}"
        for e in entries.values()
        if "error" in e
    ]
    assert not errors, (
        f"{cassette_path.name} records live failures — re-capture this fixture:\n  "
        + "\n  ".join(errors)
    )


def _resolve_fixture_path(recorded: str) -> Path:
    """Resolve the input-file path recorded in the golden json.

    Tries the recorded path as-is (if absolute), then relative to the fixtures
    dir, the samples dir, and the repo root; finally falls back to matching by
    basename under fixtures/ and samples/.
    """
    rec = Path(recorded)
    candidates = []
    if rec.is_absolute():
        candidates.append(rec)
    candidates.extend(
        [
            FIXTURES_DIR / recorded,
            SAMPLES_DIR / recorded,
            REPO_ROOT / recorded,
            FIXTURES_DIR / rec.name,
            SAMPLES_DIR / rec.name,
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"cannot resolve fixture path {recorded!r}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _is_timing_key(key) -> bool:
    return isinstance(key, str) and TIMING_KEY_RE.search(key) is not None


def _compare(actual, expected, path, mismatches):
    """Recursively compare, appending '(json path, detail)' strings to mismatches."""
    if len(mismatches) >= MAX_REPORTED_MISMATCHES:
        return

    if isinstance(actual, dict) and isinstance(expected, dict):
        a_keys = [k for k in actual.keys() if not _is_timing_key(k)]
        e_keys = [k for k in expected.keys() if not _is_timing_key(k)]
        for k in e_keys:
            if k not in actual:
                mismatches.append(f"{path}.{k}: missing in actual")
        for k in a_keys:
            if k not in expected:
                mismatches.append(f"{path}.{k}: unexpected key in actual")
        for k in e_keys:
            if k in actual:
                _compare(actual[k], expected[k], f"{path}.{k}", mismatches)
        return

    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            mismatches.append(
                f"{path}: length mismatch actual={len(actual)} expected={len(expected)}"
            )
        for i in range(min(len(actual), len(expected))):
            _compare(actual[i], expected[i], f"{path}[{i}]", mismatches)
        return

    # Numeric comparison (bool is an int subclass — keep bools strict).
    a_num = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    e_num = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    if a_num and e_num:
        af, ef = float(actual), float(expected)
        if math.isnan(af) and math.isnan(ef):
            return
        if not math.isclose(af, ef, rel_tol=1e-9):
            mismatches.append(f"{path}: {actual!r} != {expected!r}")
        return

    if type(actual) is not type(expected) or actual != expected:
        mismatches.append(f"{path}: {actual!r} != {expected!r} (actual is what the package produced)")


def _serialize_outputs(outputs):
    """Serialize {name: DataFrame} exactly like capture_goldens.py does."""
    serialized = {}
    for name, df in outputs.items():
        serialized[name] = {
            "columns": list(df.columns),
            "records": json.loads(df.to_json(orient="records")),
        }
    return serialized


@pytest.mark.parametrize(
    "golden_path,cassette_path",
    PAIRS,
    ids=[g.name[: -len(".golden.json")] for g, _ in PAIRS],
)
def test_parity(golden_path, cassette_path, monkeypatch):
    from nextract.rag_chunking import RagDocumentChunker, Config  # noqa: F401
    from cassette import CassetteReplayer  # conftest puts the goldens dir on sys.path

    # Hermetic replay: ambient env would alter Config at instantiation
    # (model names enter cassette keys; feature toggles alter routing).
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    golden = json.loads(golden_path.read_text())

    recorded_fixture = None
    for key in ("fixture", "fixture_path", "input", "input_path", "source_file", "file_path"):
        if golden.get(key):
            recorded_fixture = golden[key]
            break
    assert recorded_fixture, (
        f"{golden_path.name}: no fixture path recorded in golden json "
        "(expected one of: fixture, fixture_path, input, input_path, source_file, file_path)"
    )
    fixture_path = _resolve_fixture_path(recorded_fixture)

    expected_outputs = golden.get("outputs", golden.get("dataframes"))
    assert isinstance(expected_outputs, dict), (
        f"{golden_path.name}: golden json has no 'outputs' mapping"
    )

    chunker = RagDocumentChunker(api_key="cassette-replay", use_vertex=False)
    cfg = chunker._pipeline.cfg
    replayer = CassetteReplayer(cassette_path, cfg=cfg)
    chunker._pipeline.llm = replayer

    outputs = chunker.process_to_dataframes(str(fixture_path))
    actual_outputs = _serialize_outputs(outputs)

    # Misses are swallowed by the pipeline's fail-open handlers (CassetteMiss
    # is a KeyError), so they MUST be asserted explicitly: any miss means the
    # port issued a request the monolith never made — prompt/routing drift —
    # even if the output comparison below happens to pass.
    assert not replayer.misses, (
        f"{cassette_path.name}: {len(replayer.misses)} cassette miss(es) — the "
        "port issued LLM requests the monolith capture never made: "
        + "; ".join(f"{m['method']}:{m['key'][:12]}" for m in replayer.misses[:5])
    )

    mismatches = []
    _compare(actual_outputs, expected_outputs, "$", mismatches)
    assert not mismatches, (
        f"parity mismatch vs {golden_path.name} "
        f"(first {min(len(mismatches), MAX_REPORTED_MISMATCHES)} shown):\n"
        + "\n".join(mismatches[:MAX_REPORTED_MISMATCHES])
    )
