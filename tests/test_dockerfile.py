import json
import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def instructions():
    """(INSTRUCTION, arguments) per logical line, comments dropped, continuations joined."""
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text())
    lines = [line.strip() for line in text.splitlines()]
    pairs = [line.split(None, 1) for line in lines if line and not line.startswith("#")]
    return [(p[0].upper(), p[1] if len(p) > 1 else "") for p in pairs]


def args(instruction):
    return [a for name, a in instructions() if name == instruction]


def test_exactly_one_stage_pinned_by_digest():
    (base,) = args("FROM")
    assert re.fullmatch(r"python:3\.14-slim@sha256:[0-9a-f]{64}", base)


def test_runs_as_a_numeric_non_root_user():
    (user,) = args("USER")
    assert re.fullmatch(r"[0-9]+", user)
    assert int(user) > 0
    assert user == "65532"


def test_user_is_created_before_it_is_used():
    names = [name for name, _ in instructions()]
    assert names.index("RUN") < names.index("USER")
    (run,) = args("RUN")
    assert "useradd" in run and "--uid 65532" in run
    assert "nologin" in run


def test_a_single_copy_of_the_entrypoint_only():
    (copy,) = args("COPY")
    assert copy == "container/entrypoint.py /app/entrypoint.py"
    assert not args("ADD")


def test_entrypoint_is_the_isolated_interpreter_on_the_script():
    (entrypoint,) = args("ENTRYPOINT")
    assert json.loads(entrypoint) == ["python", "-I", "/app/entrypoint.py"]
    assert not args("CMD")


def test_bytecode_is_not_written():
    assert "PYTHONDONTWRITEBYTECODE=1" in args("ENV")


def test_nothing_is_installed_and_no_secrets_are_baked_in():
    text = DOCKERFILE.read_text().lower()
    for forbidden in ("pip ", "apt", "fastembed", "onnxruntime", "qdrant", "anthropic", "pydantic"):
        assert forbidden not in text
    assert not args("ARG")
    env = " ".join(args("ENV")).upper()
    for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD"):
        assert secret not in env


@pytest.mark.parametrize("instruction", ["EXPOSE", "VOLUME", "ADD", "HEALTHCHECK"])
def test_no_ports_volumes_or_remote_fetches(instruction):
    assert not args(instruction)
