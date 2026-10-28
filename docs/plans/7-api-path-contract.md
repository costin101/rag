# Issue #7 — Prove RAGResponse contract via one API-path query (R2)

Zone: `rag` (python, TDD). Rules in play: 1 (single response contract), 3, 4 (config only in config.py), 5 (provider/retriever interfaces), 6 (shared retrieval), 7 (stub generation).

## Decisions (human-confirmed 2026-10-28)
- Provider: Anthropic behind a `Provider` interface. Retriever: Qdrant behind a `Retriever` interface.
- Query embedding: host-side `fastembed` (via `qdrant-client[fastembed]`). Python pinned to 3.14 (`requires-python = ">=3.11,<3.15"`, dev/CI on 3.14): onnxruntime ships cp311–cp314 wheels only. This is an exception to the constitution's "latest stable" Python policy and needs `/forge:constitution` (human-only) before PR 1 merges.
- Retrieval happens ONCE on the host, before a path is chosen; #8 passes the retrieved chunks to the container, which stays a pure stub generator (no fastembed/onnxruntime in the image; rule 6 holds by construction).
- One story, three stacked PRs. Retitled `[rag] Prove RAGResponse contract via one API-path query`.

## Module layout (flat)
| File | Responsibility |
|---|---|
| `config.py` | `Settings`: `ANTHROPIC_API_KEY` (SecretStr), `RAG_PROVIDER` (default `anthropic`), `RAG_MODEL`, `RAG_MAX_TOKENS`, `RAG_PROVIDER_TIMEOUT_S`, `QDRANT_URL` (http/https; https required when a key is set), `QDRANT_API_KEY` (optional SecretStr), `QDRANT_COLLECTION`, `QDRANT_TIMEOUT_S`, `RAG_TOP_K`, `RAG_EMBED_MODEL` (pinned name+revision; the default revision is the commit of `Qdrant/bge-small-en-v1.5-onnx-Q`, the repo fastembed downloads from; PR 2 must enforce it at load time, e.g. `snapshot_download(revision=…)` + `specific_model_path`, since `TextEmbedding` takes no revision), `RAG_EMBED_CACHE_DIR`. `hide_input_in_errors=True`; `ConfigError` names the variable and reason, never the value, and reports all problems together. No Nomad fields. |
| `normalizer.py` | Frozen, `extra="forbid"` models `RAGResponse`, `ContextChunk`, `Source`, `Usage`, `Metrics`; `ErrorCode` closed `StrEnum`; `RAGError` hierarchy; path adapter `from_generation(generation, chunks, latency_ms)`. #8 adds its CLI-stdout adapter here (this repo owns that format; it will be versioned JSON). Vendor payload formats never live here. A committed `RAGResponse` JSON-schema snapshot test fails on drift. |
| `provider.py` | `Provider` protocol: `generate(query, chunks) -> Generation{text, input_tokens, output_tokens}` (provider-neutral). `AnthropicProvider` maps the SDK response to `Generation` (text blocks only, no `tools`), raising `ProviderPayloadError`/`ProviderError`. Factory keyed on `RAG_PROVIDER`. Client built explicitly from `Settings` (no implicit env reads): key, timeout, `max_retries`. Prompt: instructions in `system`; chunks in the user turn inside delimited, id-labelled data tags. |
| `retriever.py` | `Retriever` protocol: `retrieve(query, k) -> list[ContextChunk]`; `QdrantRetriever` (explicit timeout/API key; embedding model from `RAG_EMBED_MODEL`; verifies collection vector config at first use). |
| `pipeline.py` | Path-neutral `answer(query, retriever, generator) -> RAGResponse`: validate query, retrieve, generate, normalize, time (`latency_ms` measured here so both paths are comparable). |
| `rag_api_fallback.py` | Wires the API generator (`AnthropicProvider`) into `pipeline.answer`. |
| `main.py` | Single-query CLI; stdout is one JSON document; on `RAGError` prints only `code` + message to stderr, non-zero exit, no traceback/`__cause__`/SDK repr. |
| `tests/` | `contract.py` (shared `assert_valid_rag_response`), unit/integration/live tests, fixtures, `seed_qdrant.py`. |

## RAGResponse
```
RAGResponse { answer: str (non-blank), chunks: list[ContextChunk], sources: list[Source],
              usage: Usage{input_tokens:int>=0, output_tokens:int>=0}, metrics: Metrics{latency_ms:float>=0} }
ContextChunk { id: str, text: str, score: float (finite), source_id: str }
Source { id: str, title: str|None, uri: str|None }   # uri limited to http/https, else None
```
No `path`/`provider` field. "Caller cannot tell" = same shape AND same value semantics: the container stub (#8) reports deterministic non-zero token counts. `sources` = deduplicated, order-stable; "populated" = at least one source whose id derives from a returned chunk.

## Qdrant collection contract (external, ingestion out of scope)
Vector config compatible with `RAG_EMBED_MODEL`; point payload keys `text`, `source_id`, optional `title`, `uri`. Mismatch → `retrieval_failed`. `tests/seed_qdrant.py` builds a fixture collection with a mandatory test prefix and refuses to touch any other collection.

## Error contract (`ErrorCode`)
`config_invalid`, `invalid_query` (blank query), `retrieval_failed`, `empty_retrieval` (zero chunks: raise, never call the provider ungrounded), `provider_failed` (network/auth/rate/context-too-long), `provider_payload_invalid` (non-text-only/blank answer, missing/invalid usage, `stop_reason=max_tokens` truncation). Public set only; container failures (#8/#12) are internal fallback signals, never `RAGError`. No bare exception escapes `pipeline.answer`; a `RAGError` is not double-wrapped.

## Security requirements (from review)
SecretStr keys; secrets never in errors/repr/logs; `.gitignore` adds `.env*` (keep `!.env.example`), `.venv/`, embed cache; fake test keys only; hash-locked deps (`uv lock`, install with hashes), pinned model revision, `pip-audit` in CI; GitHub secret scanning + push protection on; live test never on `pull_request_target`; README notes queries and chunk text go to Anthropic.

## Stacked PRs
1. `config.py`, `normalizer.py` (models, errors, schema snapshot), packaging/lockfile, `.gitignore`, pytest+markers, CI workflow. Needs the constitution Python exception.
2. `retriever.py`, `provider.py`, `from_generation`, `pipeline.py`, `rag_api_fallback.py`, contract helper, seed script, tests.
3. `main.py`, subprocess tests, gated live test, overview amendment (`docs/architecture/overview.md`: Qdrant + Anthropic nodes/edges with evidence).

## Tests (TDD; full plan from the tester consult)
- Lanes: unit (`pytest-socket` disables network), `integration` (real Qdrant: `QDRANT_URL` else pinned testcontainer else skip; `RAG_REQUIRE_INTEGRATION=1` in CI turns skips into failures), `live` (`RAG_LIVE_ANTHROPIC=1`).
- Anthropic stubbed at the HTTP boundary with `httpx.MockTransport` (real SDK parsing; `max_retries=0`); 401/429/500/400/ConnectError/ReadTimeout → `provider_failed`.
- Config, `Generation` mapping, normalizer, retriever, pipeline, `main.py` subprocess, contract-helper negative tests, and the schema snapshot — as listed in the tester report; each acceptance criterion maps to at least one test.
- `tests/contract.py` accepts model or dict, checks the exact top-level key set (no `path`), non-bool ints, finite floats, unique source ids, every chunk `source_id` in `sources`, JSON round-trip.

## Risks accepted
fastembed model download size/first-run latency (pinned, cached in CI); live test cost (gated); Python 3.14 pin until onnxruntime ships cp315 wheels.
