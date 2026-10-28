import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import main
from normalizer import ContextChunk, ProviderError, RAGResponse, RetrievalError, Source, Usage, Metrics
from tests.contract import assert_valid_rag_response

ROOT = Path(__file__).parent.parent
KEY = "test-not-a-key-LEAK-main"
ENV = {"ANTHROPIC_API_KEY": KEY, "QDRANT_URL": "http://localhost:6333", "QDRANT_COLLECTION": "docs"}


def response():
    return RAGResponse(
        answer="Blue.",
        chunks=[ContextChunk(id="c1", text="The sky is blue.", score=0.9, source_id="weather")],
        sources=[Source(id="weather")],
        usage=Usage(input_tokens=3, output_tokens=2),
        metrics=Metrics(latency_ms=1.5),
    )


class FakePath:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.queries = result, error, []

    def answer(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result


def run(argv, *, env=ENV, path=None):
    out, err = io.StringIO(), io.StringIO()
    built = []

    def factory(settings):
        built.append(settings)
        return path

    code = main.run(argv, env=env, out=out, err=err, api_path=factory)
    return code, out.getvalue(), err.getvalue(), built


def test_success_prints_one_json_document_that_passes_the_contract():
    path = FakePath(result=response())
    code, out, err, _ = run(["Why is the sky blue?"], path=path)
    assert (code, err) == (0, "")
    assert out.endswith("\n") and out.count("\n") == 1
    assert_valid_rag_response(json.loads(out))
    assert path.queries == ["Why is the sky blue?"]


def test_settings_come_from_the_given_environment():
    _, _, _, built = run(["q"], path=FakePath(result=response()))
    assert built[0].QDRANT_COLLECTION == "docs"


@pytest.mark.parametrize("argv", [[], ["a", "b"]])
def test_wrong_argument_count_is_a_usage_error(argv):
    code, out, err, built = run(argv, path=FakePath(result=response()))
    assert (code, out, built) == (2, "", [])
    assert err.startswith("usage:") and err.count("\n") == 1


@pytest.mark.parametrize("query", ["", "   ", "​", "\n\t"])
def test_blank_query_is_invalid_and_nothing_is_built(query):
    code, out, err, built = run([query], path=FakePath(result=response()))
    assert (code, out, built) == (1, "", [])
    assert err.startswith("invalid_query: ") and err.count("\n") == 1


def test_a_rag_error_prints_only_code_and_message():
    path = FakePath(error=RetrievalError("the vector store search failed"))
    code, out, err, _ = run(["q"], path=path)
    assert (code, out) == (1, "")
    assert err == "retrieval_failed: the vector store search failed\n"


def test_bad_config_fails_naming_the_variable_without_building_anything():
    code, out, err, built = run(["q"], env={"ANTHROPIC_API_KEY": KEY, "QDRANT_URL": "ftp://x"})
    assert (code, out, built) == (1, "", [])
    assert err.startswith("config_invalid: ") and "QDRANT_URL" in err
    assert KEY not in err and err.count("\n") == 1


def test_an_unexpected_exception_prints_a_fixed_line_without_its_message():
    path = FakePath(error=RuntimeError(f"boom {KEY}"))
    code, out, err, _ = run(["q"], path=path)
    assert (code, out) == (1, "")
    assert KEY not in err and "boom" not in err and "Traceback" not in err
    assert err.count("\n") == 1


def test_a_failure_while_building_the_path_is_reported_the_same_way():
    def factory(settings):
        raise ProviderError("the provider could not be built")

    out, err = io.StringIO(), io.StringIO()
    code = main.run(["q"], env=ENV, out=out, err=err, api_path=factory)
    assert (code, out.getvalue(), err.getvalue()) == (1, "", "provider_failed: the provider could not be built\n")


def test_default_wiring_builds_the_api_path_from_settings(monkeypatch):
    built = []

    class Stub(FakePath):
        def __init__(self, settings):
            super().__init__(result=response())
            built.append(settings)

    monkeypatch.setattr(main, "ApiPath", Stub)
    out, err = io.StringIO(), io.StringIO()
    assert main.run(["q"], env=ENV, out=out, err=err) == 0
    assert built[0].QDRANT_COLLECTION == "docs"


def subprocess_run(args, env):
    clean = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT")}
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=ROOT,
        env={**clean, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_subprocess_bad_config_exits_nonzero_with_one_clean_line():
    done = subprocess_run(["q"], {"ANTHROPIC_API_KEY": KEY, "QDRANT_URL": "ftp://x"})
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("config_invalid: ") and done.stderr.count("\n") == 1
    assert "Traceback" not in done.stderr and KEY not in done.stderr


def test_subprocess_blank_query_exits_nonzero_before_any_network():
    done = subprocess_run([" "], ENV)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("invalid_query: ") and "Traceback" not in done.stderr
    assert KEY not in done.stderr


def test_subprocess_usage_error_exits_2():
    done = subprocess_run([], ENV)
    assert done.returncode == 2 and done.stdout == "" and done.stderr.startswith("usage:")
