import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gating import integration_decision, live_decision, nomad_decision

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


NOMAD = {"NOMAD_ADDR": "http://127.0.0.1:4646", "RAG_CONTAINER_IMAGE": "127.0.0.1:5000/rag@sha256:" + "a" * 64}


@pytest.mark.parametrize(
    "env,verdict",
    [
        (NOMAD, "run"),
        ({**NOMAD, "RAG_REQUIRE_INTEGRATION": "1"}, "run"),
        ({}, "skip"),
        ({"NOMAD_ADDR": NOMAD["NOMAD_ADDR"]}, "skip"),
        ({"RAG_CONTAINER_IMAGE": NOMAD["RAG_CONTAINER_IMAGE"]}, "skip"),
        ({**NOMAD, "NOMAD_ADDR": "  "}, "skip"),
        ({**NOMAD, "RAG_CONTAINER_IMAGE": ""}, "skip"),
        ({"RAG_REQUIRE_INTEGRATION": "0"}, "skip"),
        ({"RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
        ({"NOMAD_ADDR": NOMAD["NOMAD_ADDR"], "RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
        ({**NOMAD, "RAG_CONTAINER_IMAGE": " ", "RAG_REQUIRE_INTEGRATION": "1"}, "fail"),
    ],
)
def test_nomad_decision(env, verdict):
    assert nomad_decision(env)[0] == verdict


def test_nomad_decision_names_what_is_missing():
    _, reason = nomad_decision({"NOMAD_ADDR": NOMAD["NOMAD_ADDR"]})
    assert "RAG_CONTAINER_IMAGE" in reason and "NOMAD_ADDR" not in reason
    _, reason = nomad_decision({"RAG_REQUIRE_INTEGRATION": "1"})
    assert "NOMAD_ADDR" in reason and "RAG_CONTAINER_IMAGE" in reason


def run_nomad_lane(extra_env):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("QDRANT_URL", "RAG_REQUIRE_INTEGRATION", "NOMAD_ADDR", "RAG_CONTAINER_IMAGE")
    }
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         "tests/test_integration_container_path.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_conftest_skips_the_nomad_lane_without_an_agent():
    result = run_nomad_lane({})
    assert result.returncode == 0, result.stdout
    assert "skipped" in result.stdout and "failed" not in result.stdout


def test_conftest_skips_the_nomad_lane_with_qdrant_but_no_agent():
    result = run_nomad_lane({"QDRANT_URL": "http://localhost:6333"})
    assert result.returncode == 0, result.stdout
    assert "NOMAD_ADDR" in result.stdout and "failed" not in result.stdout


def test_conftest_fails_the_nomad_lane_when_integration_is_required():
    result = run_nomad_lane({"RAG_REQUIRE_INTEGRATION": "1"})
    assert result.returncode != 0
    assert "RAG_REQUIRE_INTEGRATION=1" in result.stdout
    assert "skipped" not in result.stdout


def test_the_nomad_marker_selects_a_lane_apart_from_the_qdrant_one():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "integration and not nomad", "tests/test_integration_container_path.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert "::" not in result.stdout  # nothing selected: every test there carries `nomad`
    assert "deselected" in result.stdout


def test_ci_runs_the_nomad_lane_against_a_pinned_agent_and_a_registry():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "image: registry:2" in ci
    assert "NOMAD_ADDR: http://127.0.0.1:4646" in ci
    assert "timeout-minutes:" in ci
    assert "nomad agent -dev -bind=127.0.0.1" in ci
    assert "sha256sum -c" in ci and "NOMAD_SHA256:" in ci
    assert "RAG_CONTAINER_IMAGE=" in ci  # the pushed digest, never a tag
    assert "if: failure()" in ci and "nomad-agent.log" in ci
