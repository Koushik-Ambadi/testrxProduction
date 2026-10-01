# TESTRX production application

This directory is the production-only release candidate. The parent project
remains the authority for experiments and evaluation; this package contains the
frozen runtime contract and the minimal parser/chunker implementation needed to
rebuild its immutable corpus index. It contains no golden set or tuning runner.

## Frozen release

`config.yaml` is the runtime and index-build contract. It pins the TESTRX
manual checksum, hierarchy-384 chunking, tokenizer, BGE and MiniLM revisions,
exact cosine search, Qdrant payload/collection schema, candidate K=10, final
K=5, evaluation record, and the measured selection rationale. The runtime
refuses to start if the collection dimension, distance, or stored release
contract differs. A new retrieval configuration or corpus
requires a new `config_id`, `index_release_id`, and collection.

`release-lock.json` is the matching copy of the development-owned
`configs/retrieval/production.lock.json`. The lock records source, parser code,
canonical document, chunker/tokenizer code, ordered chunk text/metadata, exact
model artifact file hashes, runtime package versions, and the config hashes.
The production config pins the canonical JSON SHA-256 of the lock, independent
of platform line endings. At startup, the service verifies
the config, runtime dependencies, copied parser/chunker and application code,
model artifacts, and Qdrant release contract before accepting requests.

The encoders and reranker are loaded from the platform's read-only model mount:
`MODEL_ROOT/encoders/...` and `MODEL_ROOT/rerankers/...`. No model download or
fallback is allowed at runtime.

## Offline index build

Install dependencies, mount the shared models and point at a staging Qdrant
instance, then run:

```sh
pip install -r requirements.txt
pip install -e .
MODEL_ROOT=/models QDRANT_URL=http://localhost:6333 \
  python -m testrx_prod.build_index /data/TESTRX_User_Manual.pdf
```

The builder verifies the PDF checksum, runs the source parser and validation,
compares the canonical document hash, applies the frozen chunker, compares the
ordered chunk and metadata hash, hashes the model artifacts and generated
vectors, then checks the stored Qdrant vectors and release contract. It writes
a detailed PASS/FAIL JSON report and prints each check; a failed check prevents
publication. It refuses to replace an existing collection unless `--replace`
is given. Promote the completed collection as an immutable release; do not run
this builder in the online service container.

Before promotion from the development repository, run
`python scripts/freeze_production_release.py --check`. It re-parses and
re-chunks the source PDF, hashes the shared model artifacts, checks source and
production code parity, and confirms the development and production lock copies
match. To intentionally promote a parser/chunker change, copy the reviewed
implementation into this package first, then run the script without `--check`
to create the new lock. The command refuses to lock if the development and
production implementations differ.

## Online API

Run with Docker or `uvicorn testrx_prod.application:api`. `POST /prompt` accepts
`{"query":"...","top_k":5,"candidate_k":10}`; either K may be omitted. It
returns the final system/user messages, selected evidence with source metadata,
dense and reranker scores/ranks, a compact trace for all candidates, stage
latency, config/index IDs, and an `invocation_id`. The platform should use this ID to join its session/model
telemetry; timings are returned for that purpose and are not duplicated in the
TESTRX audit database. `POST /invocations/{invocation_id}/answer` stores the
eventual LLM answer against that invocation.

The app audit store is normalized SQLite: one invocation row, prompt-message
rows, and one row per dense candidate, including both model rankings and
selected rank. Mount `/data`
persistently. In a multi-host deployment, replace the store adapter with the
platform's shared application-data database while retaining the same keys and
tables; operational/session telemetry remains platform-owned.

## Container configuration

Required environment: `QDRANT_URL`, `MODEL_ROOT` (defaults to `/models`).
Optional: `QDRANT_API_KEY`, `MODEL_DEVICE`, `TESTRX_CONFIG`, `TESTRX_LOG_DB`,
`TESTRX_BUILD_REPORT`.
Mount shared models read-only at `/models`, and persistent app data at `/data`.
The image starts only the online API; index construction is an explicit offline
command.

## Verification

```sh
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```
