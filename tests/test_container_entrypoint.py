import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from normalizer import Generation, parse_container_output

ENTRYPOINT = Path(__file__).resolve().parent.parent / "container" / "entrypoint.py"
CAP = 65_536

# The image runs the script as __main__ against a fixed path; the tests import it in a real
# interpreter (isolated, like the image's `python -I`) and hand main() a temporary input file.
DRIVER = (
    "import sys; sys.path.insert(0, sys.argv[1]); import entrypoint; "
    "sys.exit(entrypoint.main(sys.argv[2]))"
)


def chunks(*ids):
    return [{"id": cid, "text": f"text of {cid}"} for cid in ids]


def encode(payload):
    return base64.b64encode(json.dumps(payload).encode())


def run(tmp_path, raw):
    """Run the entrypoint on `raw` as the template's input file; return (process, input path)."""
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    process = subprocess.run(
        [sys.executable, "-I", "-c", DRIVER, str(ENTRYPOINT.parent), str(path)],
        capture_output=True,
        timeout=30,
    )
    return process, path


def answer_for(tmp_path, query, chunk_list):
    process, _ = run(tmp_path, encode({"query": query, "chunks": chunk_list}))
    assert process.returncode == 0, process.stderr
    return parse_container_output(process.stdout, max_bytes=CAP)


# --- the stdout contract ---------------------------------------------------------


def test_output_parses_as_a_v1_generation(tmp_path):
    generation = answer_for(tmp_path, "what colour is the sky?", chunks("c1", "c2"))
    assert isinstance(generation, Generation)
    assert generation.text.strip()


def test_output_is_exactly_one_json_line(tmp_path):
    process, _ = run(tmp_path, encode({"query": "q", "chunks": chunks("c1")}))
    assert process.stdout.endswith(b"\n")
    assert process.stdout.count(b"\n") == 1
    assert set(json.loads(process.stdout)) == {"v", "answer", "input_tokens", "output_tokens"}
    assert json.loads(process.stdout)["v"] == 1
    assert process.stderr == b""


def test_two_runs_are_byte_identical(tmp_path):
    raw = encode({"query": "why is it blue?", "chunks": chunks("c1", "c2", "c3")})
    first, _ = run(tmp_path, raw)
    second, _ = run(tmp_path, raw)
    assert first.stdout == second.stdout != b""


def test_non_ascii_query_and_text_round_trip(tmp_path):
    payload = {
        "query": "pourquoi le ciel est-il bleu? 答え \U0001f600",
        "chunks": [{"id": "é1", "text": "ü 答"}],
    }
    process, _ = run(tmp_path, encode(payload))
    assert process.returncode == 0
    generation = parse_container_output(process.stdout, max_bytes=CAP)
    assert "é1" in generation.text
    assert "pourquoi" in generation.text


def test_query_with_newlines_and_quotes_stays_one_line(tmp_path):
    generation = answer_for(tmp_path, 'line one\nline "two"\r\n', chunks("c1"))
    assert "line one" in generation.text


def test_template_syntax_in_the_query_is_data(tmp_path):
    generation = answer_for(tmp_path, '{{ env "X" }} ${NOMAD_ALLOC_ID}', chunks("c1"))
    assert '{{ env "X" }}' in generation.text


# --- determinism and derivation ---------------------------------------------------


def test_answer_depends_on_the_chunk_ids(tmp_path):
    a = answer_for(tmp_path, "same query", chunks("c1", "c2"))
    b = answer_for(tmp_path, "same query", chunks("c3", "c4"))
    assert a.text != b.text


def test_answer_depends_on_the_chunk_order_and_count(tmp_path):
    base = answer_for(tmp_path, "q", chunks("c1", "c2")).text
    assert answer_for(tmp_path, "q", chunks("c2", "c1")).text != base
    assert answer_for(tmp_path, "q", chunks("c1")).text != base


def test_answer_depends_on_the_query(tmp_path):
    a = answer_for(tmp_path, "first question", chunks("c1"))
    b = answer_for(tmp_path, "second question", chunks("c1"))
    assert a.text != b.text


def test_token_counts_are_word_counts_with_a_floor_of_one(tmp_path):
    payload = {"query": "one two three", "chunks": [{"id": "c1", "text": "four five"}]}
    process, _ = run(tmp_path, encode(payload))
    out = json.loads(process.stdout)
    assert out["input_tokens"] == 5
    assert out["output_tokens"] == len(out["answer"].split())
    empty = answer_for(tmp_path, "q", [{"id": "c1", "text": ""}])
    assert empty.input_tokens == 1 and empty.output_tokens >= 1


def test_a_long_query_does_not_blow_up_the_answer(tmp_path):
    generation = answer_for(tmp_path, "word " * 20_000, chunks("c1"))
    assert len(generation.text) < 1000
    assert generation.input_tokens > 20_000


# --- the input file ----------------------------------------------------------------


def test_input_file_is_deleted_after_reading(tmp_path):
    process, path = run(tmp_path, encode({"query": "q", "chunks": chunks("c1")}))
    assert process.returncode == 0
    assert not path.exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not base64 !!!",
        base64.b64encode(b"not json"),
        encode([]),
        encode({"query": "q"}),
        encode({"chunks": chunks("c1")}),
        encode({"query": 3, "chunks": chunks("c1")}),
        encode({"query": "q", "chunks": "c1"}),
        encode({"query": "q", "chunks": [1]}),
        encode({"query": "q", "chunks": [{"id": "c1"}]}),
        encode({"query": "q", "chunks": [{"id": 1, "text": "t"}]}),
        encode({"query": "q", "chunks": [{"id": "c1", "text": None}]}),
        encode({"query": "q", "chunks": []}),
    ],
    ids=[
        "empty",
        "bad base64",
        "base64 of non-json",
        "json array",
        "no chunks key",
        "no query key",
        "query not a string",
        "chunks not a list",
        "chunk not an object",
        "chunk without text",
        "chunk id not a string",
        "chunk text not a string",
        "no chunks",
    ],
)
def test_invalid_input_fails_with_a_fixed_message_and_no_stdout(tmp_path, raw):
    process, path = run(tmp_path, raw)
    assert process.returncode == 2
    assert process.stdout == b""
    assert process.stderr == b"entrypoint: invalid input\n"
    assert not path.exists()


def test_missing_input_file_fails_with_no_stdout(tmp_path):
    process = subprocess.run(
        [sys.executable, "-I", "-c", DRIVER, str(ENTRYPOINT.parent), str(tmp_path / "absent.json")],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 2
    assert process.stdout == b""


def test_invalid_input_never_echoes_the_input(tmp_path):
    process, _ = run(tmp_path, encode({"query": "LEAK-canary", "chunks": [{"id": "LEAK-canary"}]}))
    assert b"LEAK" not in process.stdout + process.stderr


# --- the production path is fixed ---------------------------------------------------


def test_run_as_a_script_reads_only_the_fixed_input_path():
    # No argument or variable can redirect it: running the file bare looks at /local/input.json.
    assert not Path("/local/input.json").exists()
    process = subprocess.run(
        [sys.executable, "-I", str(ENTRYPOINT), str(ENTRYPOINT)],
        capture_output=True,
        env={"INPUT_PATH": str(ENTRYPOINT)},
        timeout=30,
    )
    assert process.returncode == 2
    assert process.stdout == b""


def test_entrypoint_is_stdlib_only():
    source = ENTRYPOINT.read_text()
    imported = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    assert imported <= set(sys.stdlib_module_names)
    assert not imported & {"config", "normalizer", "pipeline", "provider", "retriever", "pydantic"}
