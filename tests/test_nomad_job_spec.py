import base64
import json
import re
import traceback
import uuid

import pytest

from config import load_settings
from container_session import _POLL_S, _PURGE_POLLS
from nomad_client import KILL_TIMEOUT_S, build_job
from normalizer import ContainerError, ContextChunk

DIGEST = "b" * 64
IMAGE = f"registry.example.com:5000/rag/session@sha256:{DIGEST}"
ANTHROPIC_CANARY = "test-not-a-key-LEAK-anthropic"
QDRANT_CANARY = "test-not-a-key-LEAK-qdrant"
NOMAD_CANARY = "test-not-a-key-LEAK-nomad"
QUERY_CANARY = "LEAK-canary-query"


def env(**overrides):
    values = {
        "ANTHROPIC_API_KEY": ANTHROPIC_CANARY,
        "QDRANT_URL": "https://qdrant.example.com:6333",
        "QDRANT_API_KEY": QDRANT_CANARY,
        "QDRANT_COLLECTION": "docs",
        "NOMAD_ADDR": "https://nomad.example.com:4646",
        "NOMAD_TOKEN": NOMAD_CANARY,
        "RAG_CONTAINER_IMAGE": IMAGE,
    }
    values.update(overrides)
    return values


def settings(**overrides):
    return load_settings(env(**overrides))


def chunk(cid="c1", text="the sky is blue"):
    return ContextChunk(id=cid, text=text, score=0.5, source_id="s1")


def job_id():
    return f"rag-session-{uuid.uuid4()}"


def spec(query="why is the sky blue?", chunks=None, s=None, jid=None):
    return build_job(s or settings(), jid if jid is not None else job_id(), query, chunks or [chunk("c1"), chunk("c2")])


CAPPED = settings(RAG_CONTAINER_MAX_INPUT_BYTES="1024")


def parts(body):
    job = body["Job"]
    (group,) = job["TaskGroups"]
    (task,) = group["Tasks"]
    return job, group, task


def decoded_input(body):
    _, _, task = parts(body)
    (template,) = task["Templates"]
    return json.loads(base64.b64decode(template["EmbeddedTmpl"], validate=True))


def keys_anywhere(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from keys_anywhere(value)
    elif isinstance(node, list):
        for item in node:
            yield from keys_anywhere(item)


# --- identity and placement ---------------------------------------------------------


def test_the_body_is_a_job_with_the_given_id():
    jid = job_id()
    job, _, _ = parts(spec(jid=jid))
    assert job["ID"] == jid and job["Name"] == jid
    assert re.fullmatch(r"rag-session-[0-9a-f-]{36}", job["ID"])


@pytest.mark.parametrize(
    "bad", ["", "rag-session-", "rag-session-x", "other-" + str(uuid.uuid4()), "rag-session-../x"]
)
def test_a_job_id_outside_the_session_shape_is_refused(bad):
    with pytest.raises(ContainerError):
        spec(jid=bad)


def test_batch_one_group_one_task_count_one():
    job, group, task = parts(spec())
    assert job["Type"] == "batch"
    assert group["Count"] == 1
    assert task["Driver"] == "docker"


def test_no_retry_and_no_reschedule():
    _, group, _ = parts(spec())
    assert group["RestartPolicy"] == {"Attempts": 0, "Mode": "fail"}
    assert group["ReschedulePolicy"] == {"Attempts": 0, "Unlimited": False}


def test_meta_marks_the_job_as_a_session_of_this_process():
    a = parts(spec())[0]["Meta"]
    b = parts(spec())[0]["Meta"]
    assert a["rag"] == "session"
    assert a["rag_instance"] and a["rag_instance"] == b["rag_instance"]
    assert set(a) == {"rag", "rag_instance"}


def test_namespace_comes_from_settings():
    assert parts(spec(s=settings(NOMAD_NAMESPACE="rag-sessions")))[0]["Namespace"] == "rag-sessions"


def test_without_a_namespace_the_key_is_left_out():
    assert "Namespace" not in parts(spec())[0]


# --- limits and image come from settings -----------------------------------------------


def test_cpu_memory_and_image_come_from_non_default_settings():
    s = settings(RAG_CONTAINER_CPU_MHZ="250", RAG_CONTAINER_MEMORY_MB="192")
    _, _, task = parts(spec(s=s))
    assert task["Resources"] == {"CPU": 250, "MemoryMB": 192}
    assert task["Config"]["image"] == IMAGE


def test_defaults_come_from_settings_too():
    _, _, task = parts(spec())
    assert task["Resources"] == {"CPU": 500, "MemoryMB": 128}


def test_memory_is_a_hard_limit_never_oversubscribed():
    for body in (spec(), spec(s=settings(RAG_CONTAINER_MEMORY_MB="512"))):
        _, _, task = parts(body)
        assert "MemoryMaxMB" not in task["Resources"]
        assert "memory_max" not in json.dumps(body).lower()
        assert "memory_hard_limit" not in task["Config"]


def test_cpu_is_a_hard_limit():
    assert parts(spec())[2]["Config"]["cpu_hard_limit"] is True


def test_the_job_cannot_be_built_without_an_image():
    s = load_settings({k: v for k, v in env().items() if k != "RAG_CONTAINER_IMAGE"})
    with pytest.raises(ContainerError) as excinfo:
        spec(s=s)
    assert str(excinfo.value) == "container image is not configured"


# --- hardening -----------------------------------------------------------------------------


def test_every_hardening_flag_is_set():
    _, group, task = parts(spec())
    config = task["Config"]
    assert config["network_mode"] == "none"
    assert config["readonly_rootfs"] is True
    assert config["cap_drop"] == ["ALL"]
    assert "cap_add" not in config
    assert config["privileged"] is False
    assert config["security_opt"] == ["no-new-privileges"]
    assert config["pids_limit"] > 0
    assert task["User"] == "65532"
    assert task["LogConfig"] == {"MaxFiles": 1, "MaxFileSizeMB": 1}
    assert 0 < group["EphemeralDisk"]["SizeMB"] <= 100


def test_user_is_numeric_and_not_root():
    user = parts(spec())[2]["User"]
    assert user.isdigit() and int(user) != 0


def test_no_volumes_mounts_or_host_namespaces():
    body = spec()
    _, group, task = parts(body)
    for name in ("volumes", "mount", "mounts", "pid_mode", "ipc_mode", "devices", "extra_hosts", "ports"):
        assert name not in task["Config"]
    assert not {"Volumes", "VolumeMounts", "Networks", "Services"} & set(group)
    assert not {"Volumes", "VolumeMounts", "Env", "Artifacts", "Vault", "Identity"} & set(task)
    assert not {"Volumes", "VolumeMounts", "Networks"} & set(keys_anywhere(body))


def test_no_group_level_networks_block():
    # A group Networks block next to docker network_mode="none" fails every allocation (spike).
    assert "Networks" not in set(keys_anywhere(spec()))


def test_the_task_gets_no_command_line():
    config = parts(spec())[2]["Config"]
    assert not {"args", "command", "entrypoint"} & set(config)


# --- secrets never ride in the spec ---------------------------------------------------------


def leaked(body, canaries=(ANTHROPIC_CANARY, QDRANT_CANARY, NOMAD_CANARY)):
    text = json.dumps(body)
    return [c for c in canaries if c in text or base64.b64encode(c.encode()).decode() in text]


def test_no_secret_appears_anywhere_in_the_spec():
    assert leaked(spec()) == []
    assert leaked(spec(s=settings(NOMAD_NAMESPACE="ns", RAG_CONTAINER_CPU_MHZ="300"))) == []


def test_the_leak_check_catches_a_leaking_builder():
    # Negative control: a builder that forwarded settings into the task environment must fail.
    body = spec()
    _, _, task = parts(body)
    task["Env"] = {"ANTHROPIC_API_KEY": settings().ANTHROPIC_API_KEY.get_secret_value()}
    assert leaked(body) == [ANTHROPIC_CANARY]
    body = spec()
    parts(body)[0]["Meta"]["token"] = base64.b64encode(NOMAD_CANARY.encode()).decode()
    assert leaked(body) == [NOMAD_CANARY]


def test_the_input_carries_only_the_query_and_chunk_ids_and_text():
    sent = decoded_input(spec(chunks=[chunk("c1", "hello")]))
    assert sent == {"query": "why is the sky blue?", "chunks": [{"id": "c1", "text": "hello"}]}


# --- the input channel -----------------------------------------------------------------------

TRICKY = [
    '{{ env "X" }}',
    "${NOMAD_ALLOC_ID}",
    "${attr.unique.hostname} {{ key \"secret/x\" }} {{- file \"/etc/passwd\" -}}",
    'quotes " and \' and \\ backslash',
    "line one\nline two\r\n\ttabbed",
    "unicode: é 答え \U0001f600 ​ \u0000 end",
    "{{" * 50 + "${" * 50,
]


@pytest.mark.parametrize("text", TRICKY)
def test_input_round_trips_through_base64(text):
    body = spec(query=text, chunks=[chunk("c1", text)])
    assert decoded_input(body) == {"query": text, "chunks": [{"id": "c1", "text": text}]}


@pytest.mark.parametrize("text", TRICKY)
def test_the_template_text_cannot_be_evaluated_by_nomad(text):
    (template,) = parts(spec(query=text, chunks=[chunk("c1", text)]))[2]["Templates"]
    embedded = template["EmbeddedTmpl"]
    assert re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", embedded)
    assert "{{" not in embedded and "${" not in embedded
    assert template["DestPath"] == "local/input.json"
    assert set(template) == {"EmbeddedTmpl", "DestPath"}


def test_a_large_payload_round_trips():
    text = ("word {{ x }} ${y} é " * 10_000).strip()
    body = spec(query=text, chunks=[chunk("c1", text)], s=settings(RAG_CONTAINER_MAX_INPUT_BYTES="1048576"))
    assert decoded_input(body)["query"] == text


def test_input_over_the_cap_is_refused_before_a_body_exists():
    s = settings(RAG_CONTAINER_MAX_INPUT_BYTES="1024")
    spec(query="q", s=s)
    with pytest.raises(ContainerError) as excinfo:
        spec(query="x" * 2000, s=s)
    assert str(excinfo.value) == "container input too large"


def test_the_cap_counts_bytes_not_characters():
    s = settings(RAG_CONTAINER_MAX_INPUT_BYTES="1024")
    with pytest.raises(ContainerError):
        spec(query="é" * 600, chunks=[chunk("c1", "t")], s=s)


def test_refusal_does_not_carry_the_input():
    # Nothing local here holds the query: the frames a locals-capturing reporter walks are the
    # builder's own.
    with pytest.raises(ContainerError) as excinfo:
        build_job(CAPPED, job_id(), QUERY_CANARY + "x" * 2000, [chunk()])
    err = excinfo.value
    assert QUERY_CANARY not in f"{err} {err!r}"
    assert err.__cause__ is None and err.__context__ is None
    trace = traceback.TracebackException.from_exception(err, capture_locals=True)
    assert QUERY_CANARY not in "".join(trace.format())


def test_two_builds_for_different_ids_differ_only_in_the_id():
    a, b = spec(jid=job_id()), spec(jid=job_id())
    assert a["Job"]["ID"] != b["Job"]["ID"]
    a["Job"]["ID"] = a["Job"]["Name"] = b["Job"]["ID"] = b["Job"]["Name"] = "same"
    assert a == b


def test_the_task_kill_timeout_is_explicit_and_shorter_than_the_stop_wait():
    _, _, task = parts(spec())
    assert task["KillTimeout"] == KILL_TIMEOUT_S * 1_000_000_000
    assert (_PURGE_POLLS - 1) * _POLL_S > KILL_TIMEOUT_S  # the last read still lands after the kill
