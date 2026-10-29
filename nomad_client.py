"""Nomad's JSON job shape (PascalCase) for one session container.

Nomad's request format lives here only (rule 5). A job runs the stub generator once: batch, no
restart, no reschedule, no network, read-only root, no capabilities. CPU, memory and the image
come from Settings (rule 4); the query and chunks ride as base64 so Nomad's template engine and
interpolation never see user text. Nothing from Settings other than limits, namespace and image
goes into the spec: no key, token or address.
"""

import base64
import json
import re
import uuid
from collections.abc import Sequence

from config import Settings
from normalizer import ContainerError, ContextChunk

# Marks the jobs of this process so a sweep can tell its own sessions from another's.
_INSTANCE = uuid.uuid4().hex

_JOB_ID = re.compile(r"rag-session-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

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
