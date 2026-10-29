"""Nomad's JSON job shape (PascalCase) and the HTTP client that speaks it.

Nomad's request format lives here only (rule 5). A job runs the stub generator once: batch, no
restart, no reschedule, no network, read-only root, no capabilities. CPU, memory and the image
come from Settings (rule 4); the query and chunks ride as base64 so Nomad's template engine and
interpolation never see user text. Nothing from Settings other than limits, namespace and image
goes into the spec: no key, token or address.

The client ignores the process environment (proxies, CA variables), never follows a redirect (the
token must not leave the configured host) and never switches TLS verification off. Every failure
is a `ContainerError` with fixed text, raised outside the `except` that caught the transport's:
no Nomad body, URL, header or token reaches a message, a cause or a traceback.
"""

import base64
import json
import re
import ssl
import uuid
from collections.abc import Callable, Sequence

import httpx

from config import Settings
from normalizer import ContainerError, ContextChunk

# Marks the jobs of this process so a sweep can tell its own sessions from another's.
_INSTANCE = uuid.uuid4().hex

_JOB_ID = re.compile(r"rag-session-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_ALLOC_ID = re.compile(_UUID)
_READ_CHUNK_BYTES = 8192

# Fixed hardening shape (not tunable): the task is a one-shot stub, it needs none of this room.
_USER = "65532"
_PIDS_LIMIT = 64
_EPHEMERAL_DISK_MB = 50
_TASK = "gen"
_INPUT_PATH = "local/input.json"


def _input(query: str, chunks: Sequence[ContextChunk], max_bytes: int) -> str | None:
    """base64 of the JSON the entrypoint reads, or None if it is over `max_bytes` bytes."""
    payload = {"query": query, "chunks": [{"id": c.id, "text": c.text} for c in chunks]}
    raw = json.dumps(payload).encode("ascii")  # json.dumps escapes non-ASCII
    return base64.b64encode(raw).decode("ascii") if len(raw) <= max_bytes else None


def build_job(
    settings: Settings, job_id: str, query: str, chunks: Sequence[ContextChunk]
) -> dict:
    """The request body that registers one session job for `query` and its `chunks`."""
    if not _JOB_ID.fullmatch(job_id):
        raise ContainerError("container job id is not a session id")
    if settings.RAG_CONTAINER_IMAGE is None:
        raise ContainerError("container image is not configured")
    embedded = _input(query, chunks, settings.RAG_CONTAINER_MAX_INPUT_BYTES)
    if embedded is None:
        # Error trackers capture locals: the query and the context stay out of them.
        query = chunks = None
        raise ContainerError("container input too large")
    job = {
        "ID": job_id,
        "Name": job_id,
        "Type": "batch",
        "Datacenters": ["*"],
        "Meta": {"rag": "session", "rag_instance": _INSTANCE},
        "TaskGroups": [
            {
                "Name": "session",
                "Count": 1,
                "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
                "ReschedulePolicy": {"Attempts": 0, "Unlimited": False},
                "EphemeralDisk": {"SizeMB": _EPHEMERAL_DISK_MB},
                "Tasks": [
                    {
                        "Name": _TASK,
                        "Driver": "docker",
                        "User": _USER,
                        "Config": {
                            "image": settings.RAG_CONTAINER_IMAGE,
                            # No group-level Networks block: next to this it fails every
                            # allocation (recorded in tests/fixtures/nomad/README.md).
                            "network_mode": "none",
                            "readonly_rootfs": True,
                            "cap_drop": ["ALL"],
                            "privileged": False,
                            "security_opt": ["no-new-privileges"],
                            "pids_limit": _PIDS_LIMIT,
                            "cpu_hard_limit": True,
                        },
                        # MemoryMaxMB is never set: memory is a hard limit, not a floor.
                        "Resources": {
                            "CPU": settings.RAG_CONTAINER_CPU_MHZ,
                            "MemoryMB": settings.RAG_CONTAINER_MEMORY_MB,
                        },
                        "LogConfig": {"MaxFiles": 1, "MaxFileSizeMB": 1},
                        "Templates": [{"EmbeddedTmpl": embedded, "DestPath": _INPUT_PATH}],
                    }
                ],
            }
        ],
    }
    if settings.NOMAD_NAMESPACE is not None:
        job["Namespace"] = settings.NOMAD_NAMESPACE
    return {"Job": job}


class NomadClient:
    """The few Nomad calls a session needs. Ids are checked before they become part of a path."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        if settings.NOMAD_ADDR is None:
            raise ContainerError("nomad address is not configured")
        verify = _verify(settings)
        headers = {}
        if settings.NOMAD_TOKEN is not None:
            headers["X-Nomad-Token"] = settings.NOMAD_TOKEN.get_secret_value()
        self._base = settings.NOMAD_ADDR.rstrip("/")
        self._namespace = settings.NOMAD_NAMESPACE
        self._http = httpx.Client(
            headers=headers,
            timeout=settings.RAG_CONTAINER_TIMEOUT_S,
            verify=verify,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    def __repr__(self) -> str:
        return "NomadClient()"

    def close(self) -> None:
        self._http.close()

    def register(self, body: dict) -> None:
        """Register the job built by `build_job`."""
        try:
            self._json(self._send("PUT", "/v1/jobs", json=body), dict)
        finally:
            body = None  # the body holds the query: keep it out of a captured traceback

    def evaluations(self, job_id: str) -> list[dict]:
        return self._json(self._send("GET", f"/v1/job/{_job(job_id)}/evaluations"), list)

    def allocations(self, job_id: str) -> list[dict]:
        return self._json(self._send("GET", f"/v1/job/{_job(job_id)}/allocations"), list)

    def stop(self, job_id: str) -> None:
        """Stop the job but keep it registered, so its allocations stay readable until it is purged."""
        self._json(self._send("DELETE", f"/v1/job/{_job(job_id)}"), dict)

    def deregister(self, job_id: str) -> None:
        """Stop and purge the job (`purge=true`: no record is left for Nomad's garbage collector)."""
        self._json(self._send("DELETE", f"/v1/job/{_job(job_id)}", params={"purge": "true"}), dict)

    def gc_allocation(self, alloc_id: str) -> None:
        """Best effort: ask the client to collect the allocation's directory; failure is ignored."""
        try:
            self._send("PUT", f"/v1/client/allocation/{_alloc(alloc_id)}/gc")
        except ContainerError:
            pass

    def read_stdout(
        self,
        alloc_id: str,
        max_bytes: int,
        *,
        expired: Callable[[], bool] = lambda: False,
        timeout: float | None = None,
    ) -> bytes:
        """The task's stdout, read no further than one byte past `max_bytes`.

        The extra byte lets the parser refuse an oversize output without the host ever holding
        more than the cap plus one read.
        """
        params = {"task": _TASK, "type": "stdout", "plain": "true"}
        response = self._send(
            "GET",
            f"/v1/client/fs/logs/{_alloc(alloc_id)}",
            params=params,
            stream=True,
            timeout=timeout,
            failure="container output could not be read",
        )
        data, problem = bytearray(), None
        try:
            if response.status_code != 200:
                problem = "container output could not be read"
            else:
                for chunk in response.iter_bytes(_READ_CHUNK_BYTES):
                    data += chunk
                    if len(data) > max_bytes:
                        break
                    if expired():
                        problem = "container run timed out"
                        break
        except Exception:
            problem = "container output could not be read"
        finally:
            response.close()
        if problem is not None:
            raise ContainerError(problem)
        return bytes(data[: max_bytes + 1])

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json: dict | None = None,
        stream: bool = False,
        timeout: float | None = None,
        failure: str = "nomad request failed",
    ) -> httpx.Response:
        query = dict(params or {})
        if self._namespace is not None:
            query["namespace"] = self._namespace
        request = response = None
        try:
            request = self._http.build_request(
                method,
                self._base + path,
                params=query or None,
                json=json,
                **({} if timeout is None else {"timeout": timeout}),
            )
            response = self._http.send(request, stream=stream)
        except Exception:
            pass
        if response is not None and response.status_code != 200:
            response.close()
            response = None
        if response is None:
            json = request = None
            raise ContainerError(failure)
        return response

    @staticmethod
    def _json(response: httpx.Response, kind: type):
        try:
            data = response.json()
        except Exception:
            data = None
        if not isinstance(data, kind) or (kind is dict and not data):
            data = None
        elif kind is list and not all(isinstance(item, dict) for item in data):
            data = None
        if data is None:
            raise ContainerError("nomad returned an unexpected result")
        return data


def _verify(settings: Settings):
    """The default trust store, or the configured CA file; never "no verification"."""
    if settings.NOMAD_CACERT is None:
        return True
    context = None
    try:
        context = ssl.create_default_context(cafile=str(settings.NOMAD_CACERT))
    except Exception:
        pass
    if context is None:
        raise ContainerError("nomad CA certificate could not be loaded")
    return context


def _job(job_id) -> str:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ContainerError("container job id is not a session id")
    return job_id


def _alloc(alloc_id) -> str:
    if not isinstance(alloc_id, str) or not _ALLOC_ID.fullmatch(alloc_id):
        raise ContainerError("nomad returned an unexpected result")
    return alloc_id
