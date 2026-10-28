"""Live lane: the whole API path against a real Qdrant and the real Anthropic API.

Opt-in (RAG_LIVE_ANTHROPIC=1 plus ANTHROPIC_API_KEY and QDRANT_URL); never run on pull_request_target.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from config import load_settings
from rag_api_fallback import answer_via_api
from retriever import FastEmbedder
from tests.contract import assert_valid_rag_response
from tests.seed_qdrant import drop, seed, unique_test_collection

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]

ROOT = Path(__file__).parent.parent
PASSTHROUGH = ("ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "RAG_MODEL", "RAG_EMBED_CACHE_DIR")


def env_for(collection):
    env = {name: os.environ[name] for name in PASSTHROUGH if os.environ.get(name)}
    env["QDRANT_COLLECTION"] = collection
    return env


@pytest.fixture(scope="module")
def collection():
    cfg = load_settings(env_for("unused"))
    key = cfg.QDRANT_API_KEY.get_secret_value() if cfg.QDRANT_API_KEY else None
    client = QdrantClient(url=cfg.QDRANT_URL, api_key=key, check_compatibility=False)
    name = unique_test_collection()
    seed(client, name, FastEmbedder(cfg).embed_documents)
    yield name
    drop(client, name)


def test_a_real_query_returns_a_response_that_passes_the_contract(collection):
    response = answer_via_api("Why is the sky blue?", load_settings(env_for(collection)))
    assert_valid_rag_response(response)
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0


def test_the_cli_prints_the_same_contract(collection):
    clean = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME")}
    done = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "Why is the sky blue?"],
        cwd=ROOT,
        env={**clean, **env_for(collection)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert_valid_rag_response(json.loads(done.stdout))
