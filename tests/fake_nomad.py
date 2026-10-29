"""A fake Nomad agent that replays the responses recorded from a real one (tests/fixtures/nomad).

Shapes, status codes and field names come from the recordings (see the README next to them);
only what a scenario must vary (job id, stdout bytes, allocation status, injected faults) is
changed. It is an `httpx.MockTransport` handler that logs every request, and a fake clock so a
deadline fires without sleeping.
"""

import copy
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from config import Settings, load_settings

FIXTURES = Path(__file__).parent / "fixtures" / "nomad"

IMAGE = "registry.example.com:5000/rag/session@sha256:" + "c" * 64
NAMESPACE = "rag-sessions"
TOKEN = "test-not-a-key-LEAK-nomad"
OK_STDOUT = b'{"v":1,"answer":"Stub answer from the container.","input_tokens":12,"output_tokens":5}\n'

_UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_ROUTES = {
    "register": ("PUT", re.compile(r"/v1/jobs")),
    "evaluations": ("GET", re.compile(rf"/v1/job/(?P<job>rag-session-{_UUID})/evaluations")),
    "allocations": ("GET", re.compile(rf"/v1/job/(?P<job>rag-session-{_UUID})/allocations")),
    "logs": ("GET", re.compile(rf"/v1/client/fs/logs/(?P<alloc>{_UUID})")),
    "deregister": ("DELETE", re.compile(rf"/v1/job/(?P<job>rag-session-{_UUID})")),
    "gc": ("PUT", re.compile(rf"/v1/client/allocation/(?P<alloc>{_UUID})/gc")),
}


def settings(**overrides) -> Settings:
    env = {
        "ANTHROPIC_API_KEY": "test-not-a-key-LEAK-anthropic",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "docs",
        "NOMAD_ADDR": "https://nomad.example.com:4646",
        "NOMAD_TOKEN": TOKEN,
        "NOMAD_NAMESPACE": NAMESPACE,
        "RAG_CONTAINER_IMAGE": IMAGE,
        "RAG_CONTAINER_TIMEOUT_S": "5",
    }
    env.update(overrides)
    return load_settings({k: v for k, v in env.items() if v is not None})


def recorded(name: str):
    """The parsed body of a recorded response, e.g. recorded('exit3-allocations')."""
    return json.loads((FIXTURES / f"{name}.json").read_text())["body"]


def recorded_status(name: str) -> int:
    return json.loads((FIXTURES / f"{name}.json").read_text())["status"]


class FakeClock:
    """`now` and `sleep` for the session: sleeping advances time, so deadlines cost nothing."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = 0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += seconds


# Mutations of a recorded allocation list, for states the recording run did not reach.
def pending(allocs: list[dict]) -> list[dict]:
    alloc = copy.deepcopy(allocs[0])
    alloc["ClientStatus"] = "pending"
    task = alloc["TaskStates"]["gen"]
    task["State"] = "pending"
    task["Events"] = task["Events"][:2]
    return [alloc]


def lost(allocs: list[dict]) -> list[dict]:
    alloc = copy.deepcopy(allocs[0])
    alloc["ClientStatus"] = "lost"
    return [alloc]


def two(allocs: list[dict]) -> list[dict]:
    other = copy.deepcopy(allocs[0])
    other["ID"] = "11111111-2222-3333-4444-555555555555"
    return [allocs[0], other]


def none(allocs: list[dict]) -> list[dict]:
    return []


def exit_code(code: int) -> Callable[[list[dict]], list[dict]]:
    def edit(allocs):
        alloc = copy.deepcopy(allocs[0])
        for event in alloc["TaskStates"]["gen"]["Events"]:
            if event["Type"] == "Terminated":
                event["ExitCode"] = code
        return [alloc]

    return edit


class _Stream(httpx.SyncByteStream):
    def __init__(self, chunks: Iterator[bytes], on_chunk: Callable[[], None], fail: bool):
        self._chunks, self._on_chunk, self._fail = chunks, on_chunk, fail
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            self._on_chunk()
            yield chunk
        if self._fail:
            raise httpx.ReadError("connection reset")

    def close(self):
        self.closed = True


class FakeNomad:
    """Use as `httpx.MockTransport(FakeNomad("ok"))`; `scenario` names the recording replayed."""

    def __init__(
        self,
        scenario: str = "ok",
        *,
        stdout: bytes = OK_STDOUT,
        edit_allocations: Callable[[list[dict]], list[dict]] | None = None,
        register_status: int | None = None,
        deregister_status: int | None = None,
        logs_status: int | None = None,
        logs_fail_midway: bool = False,
        redirect_register: bool = False,
    ):
        self.scenario = scenario
        self.stdout = stdout
        self.edit_allocations = edit_allocations
        self.register_status = register_status
        self.deregister_status = deregister_status
        self.logs_status = logs_status
        self.logs_fail_midway = logs_fail_midway
        self.redirect_register = redirect_register
        self.requests: list[httpx.Request] = []
        self.registered: list[str] = []
        self.specs: list[dict] = []
        self.purged = False
        self.streamed_chunks = 0

    # --- bookkeeping ------------------------------------------------------------------

    def calls(self, name: str) -> list[httpx.Request]:
        method, pattern = _ROUTES[name]
        return [
            r for r in self.requests if r.method == method and pattern.fullmatch(r.url.path)
        ]

    @property
    def job_id(self) -> str:
        (job_id,) = self.registered
        return job_id

    def _allocation_list(self) -> list[dict]:
        allocs = recorded(f"{self.scenario}-allocations")
        return self.edit_allocations(allocs) if self.edit_allocations else allocs

    def _reply(self, name: str, status: int | None = None, rewrite: bool = True) -> httpx.Response:
        body = recorded(name)
        code = recorded_status(name) if status is None else status
        if isinstance(body, dict) and "_raw_text" in body:
            return httpx.Response(code, text=body["_raw_text"])
        if rewrite and self.registered:
            body = _swap_job_id(body, self.registered[-1])
        return httpx.Response(code, json=body)

    # --- the handler ----------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for name, (method, pattern) in _ROUTES.items():
            match = pattern.fullmatch(request.url.path)
            if request.method == method and match:
                return getattr(self, f"_{name}")(request, **match.groupdict())
        return httpx.Response(404, text="unknown route")

    def _register(self, request):
        spec = json.loads(request.content)
        self.specs.append(spec)
        self.registered.append(spec["Job"]["ID"])
        if self.redirect_register:
            return httpx.Response(307, headers={"Location": "https://elsewhere.example.com/v1/jobs"})
        if self.register_status is not None:
            return httpx.Response(self.register_status, text="Internal error LEAK-register-body")
        return self._reply(f"{self.scenario}-register")

    def _evaluations(self, request, job):
        # The recorded 'ok' evaluation stands in for the scenarios that never record one.
        name = self.scenario if (FIXTURES / f"{self.scenario}-evaluations.json").exists() else "ok"
        return self._reply(f"{name}-evaluations")

    def _allocations(self, request, job):
        if self.purged:
            return self._reply(f"{self.scenario}-allocations-after-purge")
        body = _swap_job_id(self._allocation_list(), self.registered[-1])
        return httpx.Response(recorded_status(f"{self.scenario}-allocations"), json=body)

    def _logs(self, request, alloc):
        if self.logs_status is not None:
            return httpx.Response(self.logs_status, text="LEAK-logs-body")
        if self.scenario == "badimage":
            return self._reply("badimage-logs-stdout-plain")
        count = lambda: setattr(self, "streamed_chunks", self.streamed_chunks + 1)  # noqa: E731
        chunks = (self.stdout[i : i + 4096] for i in range(0, len(self.stdout), 4096))
        return httpx.Response(200, stream=_Stream(chunks, count, self.logs_fail_midway))

    def _deregister(self, request, job):
        self.purged = True
        if self.deregister_status is not None:
            return httpx.Response(self.deregister_status, text="LEAK-deregister-body")
        return self._reply(f"{self.scenario}-deregister")

    def _gc(self, request, alloc):
        name = self.scenario if (FIXTURES / f"{self.scenario}-alloc-gc.json").exists() else "ok"
        return self._reply(f"{name}-alloc-gc")


def _swap_job_id(body, new: str):
    """Recorded bodies name the spike's job id; replace it with the one the test registered."""
    text = json.dumps(body)
    return json.loads(re.sub(r"rag-spike-[a-z0-9]+-[0-9a-f]{8}", new, text))


def assert_purged_once(fake: FakeNomad, namespace: str | None = NAMESPACE) -> None:
    """Exactly one `DELETE /v1/job/<registered id>?purge=true` (with namespace), after register."""
    deletes = [r for r in fake.requests if r.method == "DELETE"]
    assert len(deletes) == 1, f"expected one purge, saw {len(deletes)}"
    (delete,) = deletes
    assert delete.url.path == f"/v1/job/{fake.job_id}"
    expected = {"purge": "true"} | ({"namespace": namespace} if namespace else {})
    assert dict(delete.url.params) == expected
    assert fake.requests.index(delete) > fake.requests.index(fake.calls("register")[0])
