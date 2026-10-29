# rag

Hybrid local/API RAG service with ephemeral Nomad session containers.

## Why this project exists

The goal is to reduce API call costs. Serving every query through a paid LLM API is the simplest design but the most expensive. When concurrency is below hundreds of concurrent executions per second, local or rented hardware can serve most traffic at a fraction of the cost. The API is only needed for the peaks.

This project is a tiered system: low-concurrency periods run on local or rented hardware, and high-concurrency periods run on the API. A CLI-based local path does not scale to high concurrency (per-execution process and container startup, file or text output handling, and per-machine limits all cap throughput), so the API absorbs the overflow.

### Estimated savings

Illustrative estimates, not measured results:

| Execution mix | Monthly API cost | Local infrastructure cost | Estimated total savings |
|---|---|---|---|
| 100% cloud API | ~$1,500 – $5,000+ | $0 | 0% (baseline) |
| Hybrid (80% local / 20% API fallback) | ~$300 – $1,000 | ~$100 – $300 (hosting/server) | ~60% – 75% |
| Hybrid (95% local / 5% API fallback) | ~$75 – $250 | ~$100 – $300 (hosting/server) | ~75% – 85% |

### How it works

- **Session-scoped containers**: each session gets its own ephemeral Docker container with explicit CPU and memory limits, started once per session and purged when the session ends. This removes per-execution startup cost and cross-session state pollution.
- **Nomad orchestration**: the container pool is scheduled by HashiCorp Nomad Community Edition (free, single binary, native Docker driver).
- **API fallback**: when the pool is exhausted, requests go to an external LLM API instead of failing or queueing.
- **Output normalization**: both paths return the same `RAGResponse`, so callers never branch on which path ran.

## Data handling

Queries and retrieved chunk text are sent to the Anthropic API (`RAG_PROVIDER_BASE_URL`) on the API path. Do not index or query data that must not leave your environment.

## Usage

`python main.py "your question"` (API path) or `python main.py --path container "your question"` (an ephemeral Nomad session container, needs `NOMAD_ADDR` and `RAG_CONTAINER_IMAGE`) prints one `RAGResponse` JSON document; a failure exits 1 with a final `<code>: <message>` line on stderr (hub download progress or warnings may precede it) (usage errors exit 2). Configuration comes from the environment (see `.env.example`).
