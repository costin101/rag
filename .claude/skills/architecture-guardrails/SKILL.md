---
name: architecture-guardrails
description: rag architecture guardrails. Use before writing or changing any code in this repo — the enforcement checklist of the root CLAUDE.md constitution. Also use when evaluating whether a proposed design fits the system.
---
# rag architecture guardrails

Read the root CLAUDE.md first; this skill is its ENFORCEMENT CHECKLIST — one testable bullet per constitutional rule, phrased so a reviewer can answer yes/no on a diff.

## Before writing code, verify the change against these boundaries
- Response contract (rule 1): callers receive only `RAGResponse`. A caller importing a path-specific payload type or branching on path → wrong; add or extend an adapter in normalizer.py. (Enforced: review-only; contract test planned.)
- Container lifecycle (rule 2): every container creation has a matching purge reachable from error, exit, and signal paths. A job registered without limits, or a purge only on the happy path → wrong. (Enforced: review-only; cleanup test planned.)
- Routing (rule 3): path is chosen once per session. Re-deciding per query, or surfacing pool exhaustion as an error → wrong; fall back. (Enforced: review-only.)
- Config (rule 4): no literal hosts, limits, ports, or keys outside config.py. (Enforced: review-only.)
- Interfaces (rule 5): provider- or vector-store-specific code outside its implementation module → wrong. (Enforced: review-only.)
- Retrieval (rule 6): the container path and the API path use the same retriever. A second retrieval implementation → wrong.
- Generation (rule 7): non-deterministic or network-calling generation inside the container image without a scoped story → wrong.

## Smells that mean "stop and re-read the design"
- A caller checking `if used_container` or similar.
- A `finally`-less allocation start, or cleanup that depends on the process exiting normally.
- A limit, host, or key appearing as a literal.
- Two code paths that format the answer differently before normalization.
- Retry loops waiting for capacity (queueing is a non-goal).
