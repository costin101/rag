"""Spike stub: probes what a hardened Nomad task can see, then prints one JSON line."""
import base64
import json
import os
import socket
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
path = "/local/input.json"
with open(path, "rb") as f:
    raw = f.read()
payload = json.loads(base64.b64decode(raw))
os.unlink(path)

try:
    socket.create_connection(("1.1.1.1", 53), timeout=2).close()
    network = "reachable"
except OSError as exc:
    network = type(exc).__name__

try:
    with open("/app/probe", "w"):
        pass
    rootfs = "writable"
except OSError as exc:
    rootfs = type(exc).__name__

sys.stderr.write("spike stderr line\n")
print(json.dumps({"v": 1, "answer": "echo:" + payload["query"], "input_tokens": 1, "output_tokens": 1,
                  "uid": os.getuid(), "network": network, "rootfs": rootfs}))
sys.exit(3 if mode == "fail" else 0)
