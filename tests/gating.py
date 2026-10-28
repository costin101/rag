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


def live_decision(env: Mapping[str, str]) -> tuple[str, str]:
    """('run' | 'skip', reason) for a live test: it spends real Anthropic tokens, so it is opt-in."""
    if env.get("RAG_LIVE_ANTHROPIC") != "1":
        return "skip", "RAG_LIVE_ANTHROPIC=1 is not set; the live lane calls the real Anthropic API"
    if not env.get("ANTHROPIC_API_KEY", "").strip():
        return "skip", "ANTHROPIC_API_KEY is not set"
    return integration_decision(env)
