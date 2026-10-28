"""Spike recorder: runs scenarios against a Nomad dev agent and saves the raw responses."""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

ADDR = os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646")
IMAGE = os.environ["SPIKE_IMAGE"]
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def call(method, path, body=None, name=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(ADDR + path, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {"_raw_text": raw.decode("utf-8", "replace")}
    if name:
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump({"method": method, "path": path, "status": status, "body": parsed}, f, indent=1, sort_keys=True)
    return status, parsed


def job_spec(job_id, image, mode="ok", memory=128):
    payload = base64.b64encode(json.dumps({"query": "q {{ env \"X\" }} ${NOMAD_ALLOC_ID} é"}).encode()).decode()
    return {"Job": {
        "ID": job_id, "Name": job_id, "Type": "batch", "Datacenters": ["*"],
        "Meta": {"rag": "session"},
        "TaskGroups": [{
            "Name": "g", "Count": 1,
            "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
            "ReschedulePolicy": {"Attempts": 0, "Unlimited": False},
            "EphemeralDisk": {"SizeMB": 50},
            "Tasks": [{
                "Name": "gen", "Driver": "docker", "User": "65532",
                "Config": {
                    "image": image, "args": [mode], "network_mode": "none", "readonly_rootfs": True,
                    "cap_drop": ["ALL"], "privileged": False, "security_opt": ["no-new-privileges"],
                    "pids_limit": 64, "cpu_hard_limit": True,
                },
                "Resources": {"CPU": 100, "MemoryMB": memory},
                "LogConfig": {"MaxFiles": 1, "MaxFileSizeMB": 1},
                "Templates": [{"EmbeddedTmpl": payload, "DestPath": "local/input.json"}],
            }],
        }],
    }}


def wait_terminal(job_id, seconds=90):
    deadline = time.time() + seconds
    allocs = []
    while time.time() < deadline:
        _, allocs = call("GET", f"/v1/job/{job_id}/allocations")
        if allocs and all(a["ClientStatus"] in ("complete", "failed", "lost") for a in allocs):
            return allocs
        time.sleep(1)
    return allocs


def scenario(name, image, mode="ok", memory=128, wait_eval=False):
    job_id = f"rag-spike-{name}-{uuid.uuid4().hex[:8]}"
    call("PUT", "/v1/jobs", job_spec(job_id, image, mode, memory), f"{name}-register")
    if wait_eval:
        time.sleep(8)
        call("GET", f"/v1/job/{job_id}/evaluations", name=f"{name}-evaluations")
        call("GET", f"/v1/job/{job_id}/allocations", name=f"{name}-allocations")
    else:
        allocs = wait_terminal(job_id)
        call("GET", f"/v1/job/{job_id}/evaluations", name=f"{name}-evaluations")
        call("GET", f"/v1/job/{job_id}/allocations", name=f"{name}-allocations")
        for alloc in allocs[:1]:
            aid = alloc["ID"]
            call("GET", f"/v1/allocation/{aid}", name=f"{name}-allocation")
            for stream in ("stdout", "stderr"):
                call("GET", f"/v1/client/fs/logs/{aid}?task=gen&type={stream}&plain=true", name=f"{name}-logs-{stream}-plain")
                call("GET", f"/v1/client/fs/logs/{aid}?task=gen&type={stream}", name=f"{name}-logs-{stream}-framed")
            call("PUT", f"/v1/client/allocation/{aid}/gc", name=f"{name}-alloc-gc")
    call("DELETE", f"/v1/job/{job_id}?purge=true", name=f"{name}-deregister")
    time.sleep(2)
    call("GET", f"/v1/job/{job_id}", name=f"{name}-job-after-purge")
    call("GET", f"/v1/job/{job_id}/allocations", name=f"{name}-allocations-after-purge")


call("GET", "/v1/status/leader", name="leader")
call("GET", "/v1/nodes", name="nodes")
scenario("ok", IMAGE)
scenario("exit3", IMAGE, mode="fail")
scenario("badimage", "127.0.0.1:5000/spike-missing@sha256:" + "0" * 64)
scenario("unplaceable", IMAGE, memory=900000, wait_eval=True)
