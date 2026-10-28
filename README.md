# rag

Hybrid local/API RAG service with ephemeral Nomad session containers.

## Data handling

Queries and retrieved chunk text are sent to the Anthropic API (`RAG_PROVIDER_BASE_URL`) on the API path. Do not index or query data that must not leave your environment.

## Usage

`python main.py "your question"` prints one `RAGResponse` JSON document; a failure prints one `<code>: <message>` line to stderr and exits 1 (usage errors exit 2). Configuration comes from the environment (see `.env.example`).
