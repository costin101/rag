import json
import traceback

import pytest

from normalizer import ContainerError, Generation, RAGError, parse_container_output

CANARY = "LEAK-canary-stdout"
CAP = 4096


def line(**overrides):
    data = {"v": 1, "answer": "an answer", "input_tokens": 7, "output_tokens": 3}
    data.update(overrides)
    return (json.dumps(data) + "\n").encode()


def parse(stdout, cap=CAP):
    return parse_container_output(stdout, max_bytes=cap)


def rejected(stdout, cap=CAP):
    with pytest.raises(ContainerError) as excinfo:
        parse(stdout, cap)
    return excinfo.value


# --- the valid shape ------------------------------------------------------------


def test_valid_line_becomes_a_generation():
    assert parse(line()) == Generation(text="an answer", input_tokens=7, output_tokens=3)


def test_valid_line_without_trailing_newline_is_accepted():
    assert parse(line().rstrip(b"\n")).text == "an answer"


def test_non_ascii_answer_round_trips():
    assert parse(line(answer="réponse — 答え")).text == "réponse — 答え"
    raw = json.dumps({"v": 1, "answer": "é", "input_tokens": 1, "output_tokens": 1}, ensure_ascii=False)
    assert parse(raw.encode()).text == "é"


def test_zero_token_counts_are_accepted():
    g = parse(line(input_tokens=0, output_tokens=0))
    assert (g.input_tokens, g.output_tokens) == (0, 0)


def test_the_recorded_spike_line_shape_is_rejected_for_its_extra_keys():
    # The spike entrypoint also printed uid/network/rootfs; the production contract has none.
    stdout = line(uid=65532, network="OSError", rootfs="OSError")
    assert isinstance(rejected(stdout), ContainerError)


def test_output_exactly_at_the_cap_is_accepted():
    stdout = line()
    assert parse(stdout, cap=len(stdout)).text == "an answer"


# --- everything else is refused with a fixed message ----------------------------

BAD_OUTPUTS = {
    "empty": b"",
    "newline only": b"\n",
    "truncated": line()[:20],
    "truncated mid-string": b'{"v":1,"answer":"abc',
    "non-json": b"Traceback (most recent call last):\n",
    "json array": b'[{"v":1}]\n',
    "json string": b'"text"\n',
    "json null": b"null\n",
    "bom": b"\xef\xbb\xbf" + line(),
    "nan": b'{"v":1,"answer":"a","input_tokens":NaN,"output_tokens":1}\n',
    "infinity": b'{"v":1,"answer":"a","input_tokens":Infinity,"output_tokens":1}\n',
    "negative infinity": b'{"v":1,"answer":"a","input_tokens":-Infinity,"output_tokens":1}\n',
    "duplicate key": b'{"v":1,"answer":"a","answer":"b","input_tokens":1,"output_tokens":1}\n',
    "duplicate v": b'{"v":1,"v":1,"answer":"a","input_tokens":1,"output_tokens":1}\n',
    "second json line": line() + line(),
    "trailing garbage line": line() + b"extra\n",
    "leading blank line": b"\n" + line(),
    "blank line between": line() + b"\n" + line(),
    "two objects on one line": line().rstrip(b"\n") + line().rstrip(b"\n") + b"\n",
    "trailing text on the line": line().rstrip(b"\n") + b" x\n",
    "leading space": b" " + line(),
    "trailing space": line().rstrip(b"\n") + b" \n",
    "crlf": line().rstrip(b"\n") + b"\r\n",
    "pretty printed": json.dumps(
        {"v": 1, "answer": "a", "input_tokens": 1, "output_tokens": 1}, indent=1
    ).encode(),
    "invalid utf-8": b'{"v":1,"answer":"\xff\xfe","input_tokens":1,"output_tokens":1}\n',
    "overlong utf-8": b'{"v":1,"answer":"\xc0\xaf","input_tokens":1,"output_tokens":1}\n',
    "lone surrogate escape": line(answer="\ud800"),
    "missing v": b'{"answer":"a","input_tokens":1,"output_tokens":1}\n',
    "v is 2": line(v=2),
    "v is 0": line(v=0),
    "v is true": line(v=True),
    "v is float": line(v=1.0),
    "v is string": line(v="1"),
    "v is null": line(v=None),
    "extra key": line(extra="x"),
    "extra key null": line(provider=None),
    "missing answer": b'{"v":1,"input_tokens":1,"output_tokens":1}\n',
    "missing input_tokens": b'{"v":1,"answer":"a","output_tokens":1}\n',
    "missing output_tokens": b'{"v":1,"answer":"a","input_tokens":1}\n',
    "answer is int": line(answer=3),
    "answer is null": line(answer=None),
    "answer is list": line(answer=["a"]),
    "empty answer": line(answer=""),
    "whitespace answer": line(answer="  \t\n "),
    "zero-width answer": line(answer="​‍"),
    "filler answer": line(answer="ㅤ⠀"),
    "bool tokens": line(input_tokens=True),
    "bool output tokens": line(output_tokens=False),
    "float tokens": line(input_tokens=1.0),
    "exponent tokens": b'{"v":1,"answer":"a","input_tokens":1e2,"output_tokens":1}\n',
    "negative tokens": line(input_tokens=-1),
    "negative output tokens": line(output_tokens=-1),
    "huge tokens": line(input_tokens=10**12 + 1),
    "enormous tokens": b'{"v":1,"answer":"a","input_tokens":1,"output_tokens":' + b"9" * 4000 + b"}\n",
    "numeric-string tokens": line(input_tokens="3"),
    "null tokens": line(output_tokens=None),
}


@pytest.mark.parametrize("stdout", BAD_OUTPUTS.values(), ids=BAD_OUTPUTS.keys())
def test_malformed_output_is_a_container_error(stdout):
    assert isinstance(rejected(stdout), ContainerError)


@pytest.mark.parametrize(
    "stdout",
    [
        b'{"v":1,"answer":"a","input_tokens":1,"output_tokens":' + b"9" * 5000 + b"}\n",
        b'{"v":1,"answer":' + b'{"a":' * 100_000 + b"1" + b"}" * 100_000 + b"}\n",
        b'{"v":1,"answer":' + b"[" * 100_000 + b"]" * 100_000 + b"}\n",
    ],
    ids=["integer past the interpreter digit limit", "object depth bomb", "array depth bomb"],
)
def test_pathological_json_within_a_large_cap_is_refused(stdout):
    assert str(rejected(stdout, cap=10**6)) == "container output is not a valid v1 result"


def test_oversize_output_is_refused_before_it_is_parsed():
    stdout = line()
    err = rejected(stdout, cap=len(stdout) - 1)
    assert str(err) == "container output too large"


def test_oversize_output_is_refused_even_when_it_is_not_json():
    assert str(rejected(b"x" * (CAP + 1))) == "container output too large"


@pytest.mark.parametrize("not_bytes", ["text", None, 3, bytearray(b"{}"), [b"{}"]])
def test_non_bytes_input_is_refused(not_bytes):
    assert isinstance(rejected(not_bytes), ContainerError)


def test_refusal_messages_are_fixed_text():
    messages = {str(rejected(stdout)) for stdout in BAD_OUTPUTS.values()}
    assert messages <= {"container output too large", "container output is not a valid v1 result"}


# --- error hygiene: stdout is untrusted and may hold anything -------------------


def _leaking_outputs():
    canary = CANARY.encode()
    return {
        "canary as answer in a rejected line": line(answer=CANARY, extra=CANARY),
        "canary in a non-json line": canary + b" not json\n",
        "canary in a second line": line(answer=CANARY) + canary + b"\n",
        "canary as a bad token": line(answer=CANARY, input_tokens=CANARY),
        "canary as v": line(answer=CANARY, v=CANARY),
        "canary as duplicate key": b'{"%s":1,"%s":2}\n' % (canary, canary),
        "canary in invalid utf-8": canary + b"\xff\n",
        "canary past the cap": canary * 2000,
    }


LEAKING = _leaking_outputs()


def raised_for(name):
    # No local here holds the output, so the frames a locals-capturing reporter walks are clean
    # up to the parser's own.
    with pytest.raises(ContainerError) as excinfo:
        parse_container_output(LEAKING[name], max_bytes=1024)
    return excinfo.value


@pytest.mark.parametrize("name", LEAKING)
def test_stdout_never_reaches_the_error(name):
    err = raised_for(name)
    assert CANARY not in f"{err} {err!r} {err.args}"
    assert err.__cause__ is None
    assert err.__context__ is None
    assert CANARY not in "".join(traceback.format_exception(err))


@pytest.mark.parametrize("name", LEAKING)
def test_stdout_not_reachable_from_traceback_locals(name):
    # Locals-capturing reporters (Sentry, rich show_locals, pytest --showlocals) walk the frames.
    trace = traceback.TracebackException.from_exception(raised_for(name), capture_locals=True)
    assert CANARY not in "".join(trace.format())


def test_container_error_is_not_a_public_rag_error():
    assert not issubclass(ContainerError, RAGError)
    assert issubclass(ContainerError, Exception)


def test_container_error_carries_only_its_message():
    err = ContainerError("the container failed")
    assert str(err) == "the container failed"
    assert err.args == ("the container failed",)
