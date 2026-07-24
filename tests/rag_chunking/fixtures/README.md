# rag_chunking test fixtures

Small deterministic input files used for golden-output capture and parity tests
between the original monolith and the ported `nextract.rag_chunking` package.
Regenerate the tiny.* files with `make_fixtures.py` (fixed literal data only; csv is byte-stable).
`loan-extraction.pdf` (130KB) is the PDF fixture, vendored here so the parity
suite is self-contained.
