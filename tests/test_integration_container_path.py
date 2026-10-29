"""Integration lane for the container path: a real Nomad agent and the real built image.

One query goes through `ContainerPath` on a seeded Qdrant (real retrieval and pipeline, the stub
generator running in a hardened container). Needs NOMAD_ADDR and RAG_CONTAINER_IMAGE (a
digest-pinned image in a registry the agent can pull from) besides QDRANT_URL; CI provides all
three, locally the lane skips. NOMAD_TOKEN, NOMAD_NAMESPACE and NOMAD_CACERT are passed through
when set. The dev agent has only the `default` namespace and no ACLs, so token and namespace
behaviour is covered by the unit lane only.
"""

import json
import os
import time
import uuid

import httpx
import pytest
from qdrant_client import QdrantClient

from config import load_settings
from container_generator import NomadGenerator
from container_session import ContainerSession
from nomad_client import NomadClient, build_job
from normalizer import ContainerError, ContextChunk, ProviderError
from rag_container_path import ContainerPath
from retriever import FastEmbedder, QdrantRetriever
from tests.contract import assert_container_answer_derived, assert_valid_rag_response
from tests.seed_qdrant import drop, seed, unique_test_collection

pytestmark = [pytest.mark.integration, pytest.mark.nomad, pytest.mark.enable_socket]

QUERY = "Why is the sky blue?"
TERMINAL = {"complete", "failed", "lost"}

# Runs in place of the image's entrypoint, in a job the test registers itself, so no probe code
# ships in the production image. It reports what the hardened task can see.
PROBE = """
import json, os, socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2).close()
    network = "reachable"
except OSError as exc:
    network = type(exc).__name__
try:
    open("/app/probe", "w").close()
    rootfs = "writable"
except OSError as exc:
    rootfs = type(exc).__name__
print(json.dumps({"uid": os.getuid(), "network": network, "rootfs": rootfs}))
"""


def container_env(collection):
    env = {
        "ANTHROPIC_API_KEY": "test-not-a-key",
        "QDRANT_URL": os.environ["QDRANT_URL"],
        "QDRANT_COLLECTION": collection,
        "NOMAD_ADDR": os.environ["NOMAD_ADDR"],
        "RAG_CONTAINER_IMAGE": os.environ["RAG_CONTAINER_IMAGE"],
        "RAG_CONTAINER_TIMEOUT_S": "120",  # the first run may pull the image
    }
    for name in ("QDRANT_API_KEY", "RAG_EMBED_CACHE_DIR", "NOMAD_TOKEN", "NOMAD_NAMESPACE", "NOMAD_CACERT"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def settings_for(collection, **overrides):
    return load_settings({**container_env(collection), **overrides})


def nomad_get(path, **params):
    """Raw read of the agent, independent of the code under test."""
    headers = {"X-Nomad-Token": os.environ["NOMAD_TOKEN"]} if os.environ.get("NOMAD_TOKEN") else {}
    if os.environ.get("NOMAD_NAMESPACE"):
        params["namespace"] = os.environ["NOMAD_NAMESPACE"]
    return httpx.get(
        os.environ["NOMAD_ADDR"] + path, params=params, headers=headers, timeout=30, trust_env=False
    )


def session_jobs():
    response = nomad_get("/v1/jobs", prefix="rag-session-")
    assert response.status_code == 200
    return [job["ID"] for job in response.json()]


class SpyClient(NomadClient):
    """Notes whether the registered job existed on the agent while the run was in flight."""

    def __init__(self, settings):
        super().__init__(settings)
        self.existed_during_run = None
        self.registered = []
        self.live_at_purge = None

    def read_stdout(self, alloc_id, *args, **kwargs):
        job_id = next(iter(self.registered), None)
        self.existed_during_run = job_id is not None and nomad_get(f"/v1/job/{job_id}").status_code == 200
        return super().read_stdout(alloc_id, *args, **kwargs)

    def register(self, body):
        self.registered = [body["Job"]["ID"]]
        super().register(body)

    def deregister(self, job_id):
        # After the purge the job's allocation list is always empty, so look just before it, once
        # the stop has been waited on: any non-terminal allocation here is a container still running.
        allocs = nomad_get(f"/v1/job/{job_id}/allocations").json()
        self.live_at_purge = [a["ID"] for a in allocs if a["ClientStatus"] not in TERMINAL]
        super().deregister(job_id)


class SleepingClient(SpyClient):
    """Registers a job whose task never answers, so the run ends on the deadline mid-run."""

    def register(self, body):
        config = body["Job"]["TaskGroups"][0]["Tasks"][0]["Config"]
        config["entrypoint"] = ["python", "-I", "-c", "import time; time.sleep(600)"]
        super().register(body)


@pytest.fixture(autouse=True)
def no_session_job_survives():
    """Sweep: every `rag-session-*` job is gone after each test, or the test fails."""
    yield
    leaked = session_jobs()
    if leaked:
        cleaner = NomadClient(settings_for("unused"))
        for job_id in leaked:  # clean up before failing, so one leak does not cascade
            cleaner.deregister(job_id)
        pytest.fail(f"{len(leaked)} session job(s) survived: {leaked}")


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedder(settings_for("unused"))


@pytest.fixture(scope="module")
def collection(embedder):
    cfg = settings_for("unused")
    key = cfg.QDRANT_API_KEY.get_secret_value() if cfg.QDRANT_API_KEY else None
    client = QdrantClient(url=cfg.QDRANT_URL, api_key=key, check_compatibility=False)
    name = unique_test_collection()
    try:
        seed(client, name, embedder.embed_documents)
        yield name
    finally:
        drop(client, name)


def assert_gone(job_id):
    """The job is purged. Running containers are checked by the spy just before the purge: this
    allocation list is `[]` once the job is gone, so on its own it cannot fail."""
    assert nomad_get(f"/v1/job/{job_id}").status_code == 404
    assert nomad_get(f"/v1/job/{job_id}/allocations").json() == []


def test_one_query_runs_in_a_container_and_the_job_is_purged(collection, embedder):
    settings = settings_for(collection)
    spies = []

    def factory(config):
        spy = SpyClient(config)
        spies.append(spy)
        return ContainerSession(config, client=spy)

    path = ContainerPath(
        settings, retriever=QdrantRetriever(settings, embedder=embedder), session_factory=factory
    )
    response = path.answer(QUERY)
    assert_valid_rag_response(response)
    assert response.chunks and response.chunks[0].source_id == "weather"
    assert_container_answer_derived(response, QUERY)
    (spy,) = spies
    (job_id,) = spy.registered
    assert spy.existed_during_run is True
    assert spy.live_at_purge == []
    assert_gone(job_id)


def test_a_container_that_cannot_start_still_purges_its_job(collection, embedder):
    name = os.environ["RAG_CONTAINER_IMAGE"].split("@")[0]
    settings = settings_for(collection, RAG_CONTAINER_IMAGE=f"{name}@sha256:{'0' * 64}")
    spy = SpyClient(settings)
    session = ContainerSession(settings, client=spy)
    chunk = ContextChunk(id="1", text="t", score=0.5, source_id="s")
    with pytest.raises(ContainerError) as excinfo:
        with session:
            NomadGenerator(session).generate(QUERY, [chunk])
    assert str(excinfo.value) == "container failed to start"
    assert spy.live_at_purge == []
    assert_gone(session.job_id)


def test_a_failure_on_the_path_is_the_fixed_public_error(collection, embedder):
    name = os.environ["RAG_CONTAINER_IMAGE"].split("@")[0]
    settings = settings_for(collection, RAG_CONTAINER_IMAGE=f"{name}@sha256:{'0' * 64}")
    path = ContainerPath(settings, retriever=QdrantRetriever(settings, embedder=embedder))
    with pytest.raises(ProviderError) as excinfo:
        path.answer(QUERY)
    assert str(excinfo.value) == "provider_failed: generation failed unexpectedly"


def test_the_running_task_has_no_network_and_a_read_only_root(collection):
    settings = settings_for(collection)
    chunk = ContextChunk(id="1", text="t", score=0.5, source_id="s")
    job_id = f"rag-session-{uuid.uuid4()}"
    body = build_job(settings, job_id, QUERY, [chunk])
    config = body["Job"]["TaskGroups"][0]["Tasks"][0]["Config"]
    config["entrypoint"] = ["python", "-I", "-c", PROBE]  # the spec is the one under test
    client = NomadClient(settings)
    try:
        client.register(body)
        deadline = time.monotonic() + 120
        allocs = []
        while time.monotonic() < deadline:
            allocs = client.allocations(job_id)
            if allocs and allocs[0]["ClientStatus"] in TERMINAL:
                break
            time.sleep(1)
        assert allocs and allocs[0]["ClientStatus"] == "complete"
        report = json.loads(client.read_stdout(allocs[0]["ID"], 4096))
    finally:
        client.deregister(job_id)
    assert report == {"uid": 65532, "network": "OSError", "rootfs": "OSError"}
    assert_gone(job_id)


def test_a_run_that_times_out_mid_flight_is_stopped_before_it_is_purged(collection):
    settings = settings_for(collection, RAG_CONTAINER_TIMEOUT_S="20")
    spy = SleepingClient(settings)
    session = ContainerSession(settings, client=spy)
    chunk = ContextChunk(id="1", text="t", score=0.5, source_id="s")
    with pytest.raises(ContainerError) as excinfo:
        with session:
            NomadGenerator(session).generate(QUERY, [chunk])
    assert str(excinfo.value) == "container run timed out"
    assert spy.live_at_purge == []  # the stop was waited on while the task was still running
    assert session._alloc_ids  # the task was placed, so the stop had a running container to end
    for alloc_id in session._alloc_ids:  # the session's GC may already have removed the record
        response = nomad_get(f"/v1/allocation/{alloc_id}")
        assert response.status_code == 404 or response.json()["ClientStatus"] in TERMINAL
    assert_gone(session.job_id)
