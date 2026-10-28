"""Rules for the integration lane: it needs a real Qdrant and real sockets.

Locally a missing QDRANT_URL skips the lane; CI sets RAG_REQUIRE_INTEGRATION=1 so a missing
service fails the build instead of silently turning the lane into a no-op.
"""

from collections.abc import Mapping


def integration_decision(env: Mapping[str, str]) -> tuple[str, str]:
    """('run' | 'skip' | 'fail', reason) for an integration test given the environment."""
    if env.get("QDRANT_URL", "").strip():
        return "run", ""
    if env.get("RAG_REQUIRE_INTEGRATION") == "1":
        return "fail", "QDRANT_URL is not set but RAG_REQUIRE_INTEGRATION=1"
    return "skip", "QDRANT_URL is not set; integration lane needs a real Qdrant"
