# rag — Engineering Constitution

Hybrid RAG service: answers queries through a pool of ephemeral, resource-bounded Nomad session containers and falls back to a configurable external LLM API when the pool is full. Both paths return an identical `RAGResponse`. Python, Docker, Nomad.

## Non-negotiable architecture rules
1. **Single response contract**: `RAGResponse` (normalizer.py) is the only shape callers see. Path-specific output is converted by an adapter; no caller branches on which path ran.
2. **Bounded, ephemeral containers**: every session container carries explicit CPU and memory limits from config, is created at session start, reused for that session's queries, and purged at session end, close, error, or process exit (including SIGINT/SIGTERM). Live containers never exceed `max_containers`.
3. **Sticky session routing**: path (container or API) is decided once at session start; a session keeps its path. Pool exhaustion, container start failure, or mid-session loss falls back — never fails the request.
4. **Config is the only source of settings**: Nomad host, resource bounds, pool capacity, vector store URL, provider and keys come from `config.py` (env-validated). No hardcoded hosts, limits, or secrets.
5. **Provider and retriever are interfaces**: the fallback LLM provider and the vector store client sit behind one interface each. Adding one must not touch routing or normalization.
6. **Shared retrieval**: both paths read the same networked vector store, so identical queries see identical context.
7. **Deterministic local generation**: the container's generator is a stub until a real model is explicitly scoped; swapping it is an image change only.
8. **Human gate**: AI proposes, human decides — design approval and releases require explicit human confirmation.

## Stack
Python 3.14 (deliberate exception to "latest stable": `onnxruntime`, required by `fastembed`, ships cp311–cp314 wheels only; revisit when cp315 wheels ship), Pydantic for schemas, `python-nomad` or direct Nomad HTTP, Docker. Nomad agent and vector store are external; provisioning them is out of scope.

## Folder-specific rules
Single zone `rag` (see .claude/charter.json); canon is this file. Add a folder CLAUDE.md when a folder gains its own conventions.

## Process (see .claude/charter.json)
Trunk on `main`, feature/fix branches via PR. Lifecycle: /brainstorm → human approves design (label) → /implement → /review → /feedback → merge. Releases via /release with explicit human confirmation. /adhoc triages ad-hoc changes into linked, prioritized issues.

## Communication & CLI Output Contract
- **Model Badge Attribution**: Always prefix CLI responses with the active model badge: `[Opus]` (or `[O]`), `[Sonnet]` (or `[S]`), or `[Haiku]` (or `[H]`).
- **Succinct & Formatted**: CLI responses must be ultra-short, concise, scannable, and formatted with minimal high-signal icons (✅ pass/done, ❌ fail/blocker, ⚠️ warning/drift, 🐛 bug, ⏳ in-progress). Provide deep details only when explicitly asked.
- **Mandatory Clickable Links**: Every issue ID (e.g. `[#26](https://github.com/costin101/rag/issues/26)`) and PR ID (e.g. `[#133](https://github.com/costin101/rag/pull/133)`) MUST ALWAYS be a clickable GitHub link.
