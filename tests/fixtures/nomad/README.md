# Recorded Nomad responses

Raw responses of a real `nomad agent -dev` (2.0.7, linux_amd64, docker driver) recorded by a
throwaway CI spike (issue #8, PR 0; GitHub Actions run 40790930992). `tests/fake_nomad.py`
replays these instead of guessing the API. File names are `<scenario>-<call>.json`; each file
holds `method`, `path`, `status` and the parsed `body` (`_raw_text` when the body is not JSON).

Scenarios: `ok` (exit 0), `exit3` (valid JSON, exit 3), `badimage` (pull fails), `unplaceable`
(memory no node can offer).

## Findings that shape the design

- A group-level `Networks` block with `Mode: none` next to docker `network_mode="none"` fails
  every allocation (`Setup Failure: pre-run hook "network" failed`). The job spec sets only the
  docker `network_mode="none"`; the container still has no network (probe: `OSError`).
- The hardened task (`readonly_rootfs`, `cap_drop=["ALL"]`, numeric `User` 65532,
  `no-new-privileges`, `cpu_hard_limit`) runs; the embedded template lands at `/local/input.json`
  and the entrypoint can read and unlink it; the rootfs is read-only.
- `{{ env "X" }}` and `${NOMAD_ALLOC_ID}` survive base64 in `EmbeddedTmpl` literally.
- Logs: `?plain=true` returns the raw stream (200); without it a frame
  `{"Data": <base64>, "File", "Offset"}`. A task that never started gives 404
  `task "gen" not started yet. No logs available`.
- Failures live in `TaskStates.gen.Events`: `Terminated` with `ExitCode` (exit 3 gives
  `ClientStatus: failed`), `Driver Failure` for a bad image. A non-zero exit still has readable stdout.
- Unplaceable: no allocation is created. The job-register eval is `complete` with
  `FailedTGAllocs` and a `BlockedEval` id; a second eval with status `blocked` appears.
- Purge: `DELETE /v1/job/<id>?purge=true` gives 200; then `GET /v1/job/<id>` is 404 and the
  job's allocations are `[]`. `PUT /v1/client/allocation/<id>/gc` gives 200 with an empty body.
- The dev agent has no ACLs and only the `default` namespace, so token and namespace behaviour
  is not covered by these recordings.

No response was recorded for `DELETE /v1/job/<id>` without `purge` (the stop); `tests/fake_nomad.py` answers it with the recorded `*-deregister` response, which has the same shape.
