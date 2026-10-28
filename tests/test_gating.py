import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gating import integration_decision, live_decision

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    "env,verdict",
    [
        ({"QDRANT_URL": "http://localhost:6333"}, "run"),
        ({"QDRANT_URL": "http://localhost:6333", "RAG_REQUIRE_INTEGRATION": "1"}, "run"),
        ({}, "skip"),
        ({"QDRANT_URL": "  "}, "skip"),
        ({"RAG_REQUIRE_INTEGRATION": "0"}, "skip"),
        ({"RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
        ({"QDRANT_URL": "", "RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
    ],
)
def test_integration_decision(env, verdict):
    assert integration_decision(env)[0] == verdict


def run_integration_lane(extra_env):
    env = {k: v for k, v in os.environ.items() if k not in ("QDRANT_URL", "RAG_REQUIRE_INTEGRATION")}
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         "tests/test_integration_retrieval.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_conftest_skips_the_lane_without_qdrant():
    result = run_integration_lane({})
    assert result.returncode == 0, result.stdout
    assert "skipped" in result.stdout and "failed" not in result.stdout


def test_conftest_fails_the_lane_when_integration_is_required():
    result = run_integration_lane({"RAG_REQUIRE_INTEGRATION": "1"})
    assert result.returncode != 0
    assert "RAG_REQUIRE_INTEGRATION=1" in result.stdout
    assert "skipped" not in result.stdout


def test_integration_settings_pass_the_embed_cache_dir_through(monkeypatch, tmp_path):
    from tests.test_integration_retrieval import settings_for

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("RAG_EMBED_CACHE_DIR", str(tmp_path))
    assert settings_for("docs").RAG_EMBED_CACHE_DIR == tmp_path
    monkeypatch.delenv("RAG_EMBED_CACHE_DIR")
    assert settings_for("docs").RAG_EMBED_CACHE_DIR is None


def test_ci_requires_the_integration_lane_against_a_pinned_qdrant():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'RAG_REQUIRE_INTEGRATION: "1"' in ci
    assert "QDRANT_URL: http://localhost:6333" in ci
    assert "RAG_EMBED_CACHE_DIR:" in ci
    assert "image: qdrant/qdrant:v" in ci  # a released version tag, never latest


@pytest.mark.parametrize(
    "env,verdict",
    [
        ({}, "skip"),
        ({"RAG_LIVE_ANTHROPIC": "0", "ANTHROPIC_API_KEY": "k", "QDRANT_URL": "http://q"}, "skip"),
        ({"RAG_LIVE_ANTHROPIC": "1", "QDRANT_URL": "http://q"}, "skip"),
        ({"RAG_LIVE_ANTHROPIC": "1", "ANTHROPIC_API_KEY": "  ", "QDRANT_URL": "http://q"}, "skip"),
        ({"RAG_LIVE_ANTHROPIC": "1", "ANTHROPIC_API_KEY": "k"}, "skip"),
        ({"RAG_LIVE_ANTHROPIC": "1", "ANTHROPIC_API_KEY": "k", "RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
        ({"RAG_LIVE_ANTHROPIC": "1", "ANTHROPIC_API_KEY": "k", "QDRANT_URL": "http://q"}, "run"),
    ],
)
def test_live_decision(env, verdict):
    assert live_decision(env)[0] == verdict


def test_the_live_lane_skips_by_default():
    env = {k: v for k, v in os.environ.items() if k not in ("RAG_LIVE_ANTHROPIC", "QDRANT_URL")}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_live_api.py", "-q", "-rs"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0 and "skipped" in done.stdout and "RAG_LIVE_ANTHROPIC=1" in done.stdout


def test_ci_never_runs_the_live_lane():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "RAG_LIVE_ANTHROPIC" not in ci and "pull_request_target" not in ci.replace("Never pull_request_target", "")
