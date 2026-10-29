"""One session container's lifecycle (rules 2 and 3): register, wait, read, purge.

`ContainerSession` registers a one-shot batch job, polls it to a terminal state under a hard
deadline and returns its stdout. It never waits for capacity: an evaluation that could not place
the job fails the run at once (rule 3's fallback signal is the `ContainerError`). Exit status and
failure are read from the allocation's task events, not from the logs. The job is purged on
success and on every failure path, the timeout included; a purge that fails is swallowed so it can
never replace the error that caused it. Every error is fixed text.
"""

import time
import uuid
from collections.abc import Callable, Sequence

from config import Settings
from nomad_client import NomadClient, build_job
from normalizer import ContainerError, ContextChunk

# How often the allocation is read, and how many reads the purge waits for it to stop.
_POLL_S = 0.5
_PURGE_POLLS = 10

_TASK = "gen"
_RUNNING = frozenset({"pending", "running"})


def _terminated_with(alloc: dict) -> int | None:
    """Exit code of the task's last `Terminated` event, or None if it never terminated."""
    try:
        events = alloc["TaskStates"][_TASK]["Events"]
        codes = [e["ExitCode"] for e in events if e["Type"] == "Terminated"]
    except (KeyError, TypeError):
        return None
    code = codes[-1] if codes else None
    return code if type(code) is int else None


class ContainerSession:
    def __init__(
        self,
        settings: Settings,
        *,
        client: NomadClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._timeout = settings.RAG_CONTAINER_TIMEOUT_S
        self._settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else NomadClient(settings)
        self._clock, self._sleep = clock, sleep
        self._job_id: str | None = None
        self._alloc_ids: set[str] = set()
        self._used = self._closed = False

    def __repr__(self) -> str:
        return "ContainerSession()"

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def max_output_bytes(self) -> int:
        return self._settings.RAG_CONTAINER_MAX_OUTPUT_BYTES

    def __enter__(self) -> "ContainerSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run(self, query: str, chunks: Sequence[ContextChunk]) -> bytes:
        """Run the job for `query` and `chunks`; return its stdout. A session serves one run."""
        if self._used or self._closed:
            raise ContainerError("container session already used")
        self._used = True
        body = build_job(self._settings, f"rag-session-{uuid.uuid4()}", query, chunks)
        query = chunks = None
        # Recorded before the call: a register whose answer is lost may still have created it.
        self._job_id = body["Job"]["ID"]
        try:
            self._client.register(body)
            body = None
            return self._await_stdout(self._clock() + self._timeout)
        finally:
            self.close()

    def _await_stdout(self, deadline: float) -> bytes:
        job_id = self._job_id
        while True:
            alloc = self._current_allocation(job_id)
            if alloc is None and self._unplaceable(job_id):
                raise ContainerError("container could not be placed")
            if alloc is not None:
                self._alloc_ids.add(alloc["ID"])
                status = alloc.get("ClientStatus")
                if status == "complete":
                    if _terminated_with(alloc) != 0:
                        raise ContainerError("container exited with a failure")
                    return self._client.read_stdout(
                        alloc["ID"],
                        self.max_output_bytes,
                        expired=lambda: self._clock() >= deadline,
                        timeout=max(deadline - self._clock(), 0.001),
                    )
                if status == "lost":
                    raise ContainerError("container was lost")
                if status == "failed":
                    if _terminated_with(alloc) is None:
                        raise ContainerError("container failed to start")
                    raise ContainerError("container exited with a failure")
            if self._clock() >= deadline:
                raise ContainerError("container run timed out")
            self._sleep(_POLL_S)

    def _current_allocation(self, job_id: str) -> dict | None:
        allocs = self._client.allocations(job_id)
        if len(allocs) > 1:
            raise ContainerError("container returned an unexpected allocation count")
        return allocs[0] if allocs else None

    def _unplaceable(self, job_id: str) -> bool:
        """True if an evaluation of the job failed to place it (and no allocation exists)."""
        return any(e.get("FailedTGAllocs") for e in self._client.evaluations(job_id))

    def close(self) -> None:
        """Purge the job (once); idempotent and never raises, so it cannot mask another error."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._job_id is not None:
                self._purge(self._job_id)
        except Exception:
            pass
        finally:
            if self._owns_client:
                try:
                    self._client.close()
                except Exception:
                    pass

    def _purge(self, job_id: str) -> None:
        try:
            self._client.deregister(job_id)
        except ContainerError:
            return  # best effort: the original failure, if any, is the one to report
        try:
            for _ in range(_PURGE_POLLS):
                allocs = self._client.allocations(job_id)
                if not any(a.get("ClientStatus") in _RUNNING for a in allocs):
                    break
                self._sleep(_POLL_S)
        except ContainerError:
            pass
        for alloc_id in sorted(self._alloc_ids):
            self._client.gc_allocation(alloc_id)
