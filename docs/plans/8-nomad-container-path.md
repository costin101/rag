# Issue #8 — Prove container path via one Nomad query (R1 design)

Zone: `rag` (python, TDD). Rules in play: 1 (single response contract), 2 (bounded, ephemeral), 3 (no wait, fallback signal), 4 (config only), 5 (interfaces), 6 (shared retrieval), 7 (stub generation). Touches outside the zone regex, declared as in #7: `.github/workflows/ci.yml`, `docs/architecture/overview.md`.

## Decisions (human-confirmed 2026-10-28)
- Retitled `[rag] Prove container path via one Nomad query`.
- Tests: unit lane runs against a fake Nomad (`tests/fake_nomad.py`, `httpx.MockTransport`); integration lane runs against `nomad agent -dev` in CI and locally. No agent is installed locally today; Docker is.
- Container failure is an internal `ContainerError`, never a public `RAGError`/`ErrorCode` (plan 7).

## Decisions taken in this design (from consults; open to the design gate)
- `pipeline.py` is unchanged. A `ContainerError` raised inside `generate` becomes the fixed `ProviderError` there; callers keep seeing only `RAGResponse` or `RAGError` (rules 1, 3). Tests target `ContainerSession`/`NomadGenerator`/`parse_container_output` directly. The component that wants the raw `ContainerError` as a fallback trigger is the router/session wrapper of #12, which catches it before the pipeline.
- ~~No `--path` flag in `main.py`~~ **Superseded 2026-10-29 by the tech lead (costin101)**: `main.py [--path api|container] QUERY` ships in #8 so a container session is runnable from the CLI. The default stays the API path; `--path` is an explicit operator choice, not routing (automatic path choice and fallback stay #12). No new CLI error line and no new public `ErrorCode`: container failures print the existing fixed lines. #12 will own whether `--path` survives next to routing.
- Signal/crash purge stays in #11; a `kill -9` leaks a job until then (accepted risk, owner #10/#11).

## Shape
Retrieval stays on the host (plan 7). The container only generates from the query and chunks it is given. The container path is a `Provider` implementation, so `pipeline.answer` and `from_generation` produce the `RAGResponse` unchanged: no second adapter into `RAGResponse`.

```
ContainerPath.answer(query):
  with ContainerSession(settings) as session:      # close() purges, idempotent
      pipeline.answer(query, retriever, NomadGenerator(session), k)
        retrieve -> NomadGenerator.generate -> session.run(query, chunks) -> Generation
        -> from_generation -> RAGResponse
```

`ContainerSession` owns the job lifecycle (`close()` purges, safe to call twice, context manager). In #8 `run` registers the batch job with the first query's input and reads its result; #9 replaces the input channel so one allocation serves N queries. Until then a session serves exactly one `run` (a second raises `ContainerError`).

## Units
| File | Responsibility |
|---|---|
| `container/entrypoint.py` | In-image, stdlib only. Reads base64-JSON `local/input.json`, deletes it, prints one JSON line `{"v":1,"answer":str,"input_tokens":int,"output_tokens":int}`. Deterministic stub: answer derived from query and chunk ids; token counts = word counts, floor 1 (non-zero, same value semantics as the API path). |
| `Dockerfile` | `python:3.14-slim` pinned by digest, numeric non-root `USER`, `ENTRYPOINT ["python","-I","/app/entrypoint.py"]`, `PYTHONDONTWRITEBYTECODE=1`, a single `COPY`. No fastembed/onnxruntime, vector client or secrets. |
| `normalizer.py` | `ContainerError` (NOT a `RAGError`; fixed-text messages, never Nomad bodies, host, token or stdout). `parse_container_output(stdout: bytes) -> Generation`: strict, `v == 1`, byte-capped, exactly one JSON object line, rejects empty/truncated/non-JSON/BOM/NaN/duplicate keys/extra keys/wrong types/bool or negative ints/blank answer (`is_blank`). Docstring on the public error set reworded. |
| `nomad_client.py` | Direct HTTP over `httpx` (declared in `pyproject.toml` and relocked, not left transitive): `trust_env=False`, `follow_redirects=False`, `X-Nomad-Token` header, explicit timeout, optional `NOMAD_CACERT`, never `verify=False`. Register, eval/alloc status, streamed stdout read that stops at the byte cap, deregister `purge=true`, best-effort per-allocation GC. Alloc ids validated as UUIDs before use in a path. Owns `build_job` (Nomad's JSON shape lives here only). |
| `container_session.py` | `ContainerSession` lifecycle + `run`. Hard deadline from config; fails fast on an unplaceable eval (`FailedTGAllocs`/blocked eval with no allocation) so rule 3 ("never wait") holds. Exit code and failure read from alloc `TaskStates` events, not logs. Purge runs in `close()`; a failed purge never masks the original error. |
| `container_generator.py` | `NomadGenerator(session).generate(query, chunks) -> Generation`; refuses blank query / empty chunks before any Nomad call. |
| `rag_container_path.py` | `ContainerPath.answer(query)`: the same wiring as `ApiPath` with `NomadGenerator` as the provider; checks Nomad settings at construction (`ConfigError` naming every missing variable). |
| `config.py` | Optional fields (the API path must keep working without Nomad; `ANTHROPIC_API_KEY`/`QDRANT_URL` stay required): `NOMAD_TOKEN` (declared before `NOMAD_ADDR`; https required when set), `NOMAD_ADDR` (`url_problem`), `NOMAD_CACERT`, `NOMAD_NAMESPACE` (`[A-Za-z0-9-]{1,128}`, sent via httpx `params`), `RAG_CONTAINER_IMAGE` (full `name@sha256:<64 hex>` only), `RAG_CONTAINER_CPU_MHZ`, `RAG_CONTAINER_MEMORY_MB` (bounded: Nomad floor and the interpreter's need), `RAG_CONTAINER_TIMEOUT_S`, `RAG_CONTAINER_MAX_OUTPUT_BYTES`. Format is checked whenever a value is present. `tests/test_config.py` iterates `model_fields` and is updated. |

## Job spec (JSON, PascalCase; hardened)
Batch, one group, `Count=1`, restart `Attempts=0 Mode=fail`, reschedule `Attempts=0 Unlimited=false`, docker driver: `network_mode="none"`, `readonly_rootfs`, `cap_drop=["ALL"]`, `privileged=false`, `security_opt=["no-new-privileges"]`, `pids_limit`, numeric `user`, `cpu_hard_limit=true`, no `volumes`/`mount`/host `pid_mode`/`ipc_mode`, small `EphemeralDisk`, `LogConfig` `MaxFiles=1`/`MaxFileSizeMB=1`, `memory_max` never set; CPU and memory from `Settings`. Input is base64(JSON) in the template `EmbeddedTmpl` (the alphabet carries no `{{` or `${`, so Nomad's template engine and interpolation cannot evaluate user text), capped in bytes on the host before registering. Job id `rag-session-<uuid4>`, `Meta={"rag":"session","rag_instance":<per-process id>}`, dedicated namespace.

## Purge semantics
"Purged" = `DELETE /v1/job/<id>` (stop; task `KillTimeout` 1 s, below the wait), a short bounded poll while the job is still registered until no allocation of that job is non-terminal (after a purge the list is always `[]`), then `DELETE /v1/job/<id>?purge=true` for the registered id (with namespace), then best-effort per-allocation GC of the client alloc dir. Done on success and on every failure path including timeout. The entrypoint deletes `input.json` after reading. Terminal allocation records and logs may persist until Nomad's own GC; Raft snapshots keep the job body until compaction. Accepted and documented.

## Tests (TDD)
- Unit (`--disable-socket`):
  - `parse_container_output` table: valid; empty; truncated; non-JSON; BOM; NaN/Infinity; duplicate keys; second JSON line; depth bomb; invalid UTF-8; wrong `v`; extra key; bool/float/negative/huge/numeric-string tokens; blank and invisible-char answer; oversize. A canary string in stdout must not appear in `str`, `repr`, `__cause__` or `__context__` of the error.
  - `build_job`: limits, namespace and image come from non-default `Settings`; every hardening flag; no `volumes`/`mount`; canary `ANTHROPIC_API_KEY`/`QDRANT_API_KEY`/`NOMAD_TOKEN` appear in no spec, body or header except `X-Nomad-Token`; a leaking builder fails (negative control); base64 input round-trips `{{ env "X" }}`, `${NOMAD_ALLOC_ID}`, quotes, newlines, unicode and a large payload.
  - Session against the fake Nomad: success; register 5xx; unplaceable eval; alloc `pending` forever (fake clock, deadline fires); alloc `lost`; 0 or 2 allocs; non-zero exit with valid JSON; exit 137; exit 0 with garbage; empty stdout with stderr data; logs non-200 or truncated. Each failure raises the fixed-text `ContainerError`; stop then purge asserted as the exact requests `DELETE /v1/job/<registered id>` and `DELETE /v1/job/<registered id>?purge=true` with namespace, once each, in every path, with the allocation reads between them; a failing purge does not mask the original error; blank query or empty chunks register nothing; two sessions get distinct ids.
  - Redirect from Nomad is not followed; `NOMAD_TOKEN` over http is rejected; a tag-only image is rejected.
  - `pipeline.answer` with a `NomadGenerator` that raises `ContainerError` yields the fixed `ProviderError` (unchanged pipeline).
  - `tests/gating.py`: `nomad_decision(env)` (needs `NOMAD_ADDR`, `RAG_CONTAINER_IMAGE`) with a `test_gating.py` table; separate `nomad` marker so `-m "integration and not nomad"` works.
  - Entrypoint as a subprocess: output parses via `parse_container_output`; two runs byte-identical; non-ASCII query; answer depends on chunk ids.
  - Dockerfile static test: digest-pinned `FROM`, numeric non-root `USER`, one `COPY`.
- Contract: `assert_valid_rag_response` on the container path result; same query and chunks through the stub API provider and the container give the same shape, `chunks` and `sources`; schema snapshot unchanged.
- Integration (`nomad` marker; skip when unset, fail under `RAG_REQUIRE_INTEGRATION=1`): real agent and image, one query through `ContainerPath` on seeded Qdrant; assert contract; record the registered id and assert it existed during the run, then `GET /v1/job/<id>` is 404 and no non-terminal allocation remains; bad-image failure still purges; a probe task shows the running task has no network; autouse teardown sweeps `rag-session-*` jobs and fails if any remain; an image that prints constant valid JSON fails the derived-answer assertion.
- CI: `registry:2` service container, build and push the image, read the digest, pin it; image checks (`find_spec('fastembed'/'onnxruntime') is None`, no secret env vars); install Nomad with a SHA256 hardcoded in the workflow; `sudo nomad agent -dev -bind=127.0.0.1` in the background, wait for health, node `ready` and docker driver `Healthy`; `timeout-minutes`; agent log uploaded on failure.

## Priorities (tech lead, 2026-10-29)
- Priority is a functional Nomad - Docker - CLI session. The deadline bound on a session (review of PR #25, red-team claim 7: register, the stop and its allocation reads, and the purge sit outside the deadline, worst case about 16x `RAG_CONTAINER_TIMEOUT_S`) is explicitly deprioritized and moves to a follow-up; it is not a merge condition for #8.
- New requirement, **not yet scoped**: the CLI must be reachable from a web session in a browser. This is a different deliverable from #8 (a network-facing service plus a browser client, with session identity, auth and abuse limits) and is tracked as a separate story, not built in #8.
  - **Shape (decided 2026-10-29, tech lead costin101)**: an HTTP service with a query endpoint that returns the same `RAGResponse` (rules 1 and 3: the container-or-API choice stays server-side, the browser never branches on it) plus a small static page that calls it. A terminal-in-browser (websocket to a session container) is rejected: it exposes a shell-like surface and needs its own threat model.
- Remaining open questions for that story's design:
  - Session mapping: one browser session = one container session (rules 2, 3 and #9 reuse), with an idle TTL and purge on disconnect (#11 signals/TTL).
  - Auth and tenancy: who may open a session, per-user limits against `max_containers` (#10), and what happens to an unauthenticated visitor when the pool is full (API fallback costs money).
  - Data: queries and chunks sit in the job spec until purge (see Risks); a web surface widens who can send them.
  - Hosting: Nomad agent and vector store stay external (CLAUDE.md); where the web service runs is out of scope until decided.
  - Dependencies: #9 (reuse), #10 (cap), #11 (TTL and signals), #12 (routing and fallback) come first; the web story builds on them.

## Stacked PRs
0. Spike (throwaway, first): a CI job that starts the dev agent and registry, registers the hardened job, reads logs, purges, and records the real responses (log framing, `FailedTGAllocs`, per-alloc GC, stdout at task death, template into `local/` under `readonly_rootfs`/`cap_drop`) as fixtures for `tests/fake_nomad.py`. The fake is built from these recordings, not from assumptions.
1. `ContainerError`, `parse_container_output`, `container/entrypoint.py`, `Dockerfile`, config fields, `httpx` dependency and relock, `build_job`, tests.
2. `nomad_client.py`, `container_session.py`, `container_generator.py`, `rag_container_path.py`, `tests/fake_nomad.py`, tests.
3. CI (registry, Nomad agent, image checks), integration lane, gating, `docs/architecture/overview.md` amendment (Nomad node/edges, job spec and stdout v1 under Queues/jobs), `rag_api_fallback.py` docstring fix (routing is #12).

## Risks accepted
- One-shot input channel (accepted by the tech lead, costin101, 2026-10-28, answering the contrarian): the query rides in the job spec of a one-shot batch task. #9 (one allocation for N queries) and #18 (warm container holds no session data) replace the channel; surviving from #8: `ContainerError`, `parse_container_output`, stdout v1, `build_job` hardening, `nomad_client`, purge semantics, fake Nomad, CI lane. Replaced by #9/#18: the entrypoint input path, the job type, and where `ContainerSession` starts the container (here inside generate, after retrieval; #9/#18 start it at session start).
- Where #12 detects a failed container start: not decided in #8. In #8 a start failure surfaces inside generate as a fixed `ProviderError` via the pipeline; #12 owns moving container start to session start and catching `ContainerError` there.
- Nomad dev agent + docker driver on macOS may not fingerprint reliably: CI is the proof, local run is best effort.
- Query and chunks sit in the job spec while the job exists (Nomad API, Raft until compaction); `local/input.json` is deleted by the entrypoint, logs persist until client GC. A Nomad Variable instead of the embedded template is the upgrade path.
- Leaked jobs after `kill -9` or SIGTERM until #11; no sweeper in #8.
- Follow-ups from the PR #25 review, accepted for #8 and tracked outside it: session deadline bound (above); stdout-read deadline test; `NOMAD_CACERT` load check at `ContainerPath` construction (`ConfigError`); clear the job body in `run` on a register failure so captured tracebacks do not carry the query.
- A container-only CLI run still needs `ANTHROPIC_API_KEY` set (config unchanged).
- API behaviour of Nomad (alloc logs, per-alloc GC, eval failure fields) is from consultant knowledge; verified for the first time by the integration lane.
- Story size (about 8 files, CI, docs): kept as one story in three stacked PRs, as #7.
- #9, #10, #11, #12 and #18 still carry "pending the rule 2 amendment" notes although the amendment is in `CLAUDE.md`; a follow-up edit on those issues, not part of #8.
