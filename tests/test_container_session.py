import json

import httpx
import pytest

from container_session import ContainerSession
from nomad_client import NomadClient
from normalizer import ContainerError, ContextChunk
from tests.fake_nomad import (
    NAMESPACE,
    TOKEN,
    FakeClock,
    FakeNomad,
    assert_purged_once,
    exit_code,
    lost,
    none,
    pending,
    settings,
    two,
)

CHUNKS = [ContextChunk(id="c1", text="the sky is blue", score=0.5, source_id="s1")]
QUERY = "why is the sky blue?"


def make(fake=None, **overrides):
    fake = fake or FakeNomad()
    cfg = settings(**overrides)
    clock = FakeClock()
    client = NomadClient(cfg, transport=httpx.MockTransport(fake))
    return ContainerSession(cfg, client=client, clock=clock.now, sleep=clock.sleep), fake, clock


def run_failing(fake, message, **overrides):
    session, fake, clock = make(fake, **overrides)
    with pytest.raises(ContainerError) as excinfo:
        with session:
            session.run(QUERY, CHUNKS)
    assert str(excinfo.value) == message
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None
    assert_purged_once(fake)
    return fake, clock


# --- success ------------------------------------------------------------------------------------


def test_success_returns_stdout_and_purges_once():
    session, fake, _ = make()
    with session:
        stdout = session.run(QUERY, CHUNKS)
    assert stdout == fake.stdout
    assert_purged_once(fake)


def test_the_registered_spec_carries_the_query_and_chunks():
    session, fake, _ = make()
    with session:
        session.run(QUERY, CHUNKS)
    (spec,) = fake.specs
    assert spec["Job"]["ID"] == session.job_id == fake.job_id
    assert spec["Job"]["Namespace"] == NAMESPACE


def test_the_purge_comes_after_the_read_and_then_the_allocation_is_collected():
    session, fake, _ = make()
    with session:
        session.run(QUERY, CHUNKS)
    steps = [(r.method, r.url.path.split("/")[1:3]) for r in fake.requests]
    kinds = [m for m, _ in steps]
    assert kinds[0] == "PUT" and kinds.index("DELETE") > max(
        i for i, r in enumerate(fake.requests) if "logs" in r.url.path
    )
    (gc,) = fake.calls("gc")
    assert fake.requests.index(gc) > fake.requests.index(fake.calls("deregister")[0])


def test_close_is_idempotent():
    session, fake, _ = make()
    session.run(QUERY, CHUNKS)
    session.close()
    session.close()
    assert_purged_once(fake)


def test_close_without_a_run_sends_nothing():
    session, fake, _ = make()
    session.close()
    with session:
        pass
    assert fake.requests == []


def test_a_session_serves_exactly_one_run():
    session, fake, _ = make()
    with session:
        session.run(QUERY, CHUNKS)
        with pytest.raises(ContainerError) as excinfo:
            session.run(QUERY, CHUNKS)
    assert str(excinfo.value) == "container session already used"
    assert len(fake.calls("register")) == 1
    assert_purged_once(fake)


def test_a_closed_session_cannot_run():
    session, fake, _ = make()
    session.close()
    with pytest.raises(ContainerError):
        session.run(QUERY, CHUNKS)
    assert fake.requests == []


def test_two_sessions_get_distinct_job_ids():
    (a, fa, _), (b, fb, _) = make(), make()
    with a, b:
        a.run(QUERY, CHUNKS)
        b.run(QUERY, CHUNKS)
    assert fa.job_id != fb.job_id


def test_an_input_over_the_cap_registers_nothing():
    session, fake, _ = make(RAG_CONTAINER_MAX_INPUT_BYTES="1024")
    with pytest.raises(ContainerError) as excinfo:
        with session:
            session.run("x" * 5000, CHUNKS)
    assert str(excinfo.value) == "container input too large"
    assert fake.requests == []


# --- every failure raises a fixed error and still purges once --------------------------------------


def test_register_5xx():
    fake, _ = run_failing(FakeNomad(register_status=500), "nomad request failed")
    assert len(fake.calls("allocations")) <= 1  # only the purge-time check


def test_register_redirect_is_not_followed():
    fake, _ = run_failing(FakeNomad(redirect_register=True), "nomad request failed")
    assert {r.url.host for r in fake.requests} == {"nomad.example.com"}


def test_unplaceable_eval_fails_fast_without_waiting():
    fake, clock = run_failing(FakeNomad("unplaceable"), "container could not be placed")
    assert clock.t == 0 and clock.sleeps == 0


def test_alloc_pending_forever_hits_the_deadline():
    fake, clock = run_failing(
        FakeNomad(edit_allocations=pending), "container run timed out", RAG_CONTAINER_TIMEOUT_S="5"
    )
    assert clock.t >= 5
    assert clock.sleeps <= 40


def test_the_deadline_is_the_configured_one():
    _, short = run_failing(FakeNomad(edit_allocations=pending), "container run timed out", RAG_CONTAINER_TIMEOUT_S="2")
    _, long = run_failing(FakeNomad(edit_allocations=pending), "container run timed out", RAG_CONTAINER_TIMEOUT_S="9")
    assert short.t < long.t


def test_alloc_lost():
    run_failing(FakeNomad(edit_allocations=lost), "container was lost")


def test_two_allocations():
    run_failing(FakeNomad(edit_allocations=two), "container returned an unexpected allocation count")


def test_no_allocation_and_no_failed_placement_times_out():
    run_failing(FakeNomad(edit_allocations=none), "container run timed out")


def test_non_zero_exit_with_valid_json_is_a_failure():
    run_failing(FakeNomad("exit3", stdout=b'{"v":1,"answer":"a","input_tokens":1,"output_tokens":1}\n'),
                "container exited with a failure")


def test_exit_137():
    run_failing(FakeNomad("exit3", edit_allocations=exit_code(137)), "container exited with a failure")


def test_a_complete_allocation_with_a_non_zero_terminated_event_is_a_failure():
    run_failing(FakeNomad("ok", edit_allocations=exit_code(1)), "container exited with a failure")


def test_bad_image_is_a_start_failure_and_logs_are_never_read():
    fake, _ = run_failing(FakeNomad("badimage"), "container failed to start")
    assert fake.calls("logs") == []


def test_logs_non_200():
    run_failing(FakeNomad(logs_status=500), "container output could not be read")


def test_logs_cut_off_midway():
    run_failing(FakeNomad(logs_fail_midway=True), "container output could not be read")


# --- the purge ------------------------------------------------------------------------------------------


def test_a_failing_purge_does_not_mask_the_original_error():
    fake, _ = run_failing(FakeNomad("exit3", deregister_status=500), "container exited with a failure")
    assert len(fake.calls("deregister")) == 2  # the stop, then the failing purge


def test_a_failing_purge_after_success_does_not_raise():
    session, fake, _ = make(FakeNomad(deregister_status=500))
    with session:
        assert session.run(QUERY, CHUNKS) == fake.stdout
    assert_purged_once(fake)


class Lingering(FakeNomad):
    """After the stop, the allocation still reads `running` for `polls` more reads."""

    def __init__(self, polls):
        super().__init__()
        self.polls = polls
        self.reads_after_stop = 0

    def _allocation_list(self):
        allocs = super()._allocation_list()
        if self.stopped:
            self.reads_after_stop += 1
            if self.reads_after_stop <= self.polls:
                allocs[0]["ClientStatus"] = "running"
        return allocs


def test_the_job_is_stopped_then_waited_on_then_purged():
    fake = Lingering(polls=2)
    session, _, _ = make(fake)
    with session:
        session.run(QUERY, CHUNKS)
    stop, purge = [r for r in fake.requests if r.method == "DELETE"]
    between = [r for r in fake.requests[fake.requests.index(stop) : fake.requests.index(purge)] if r.method == "GET"]
    assert fake.reads_after_stop == 3  # two running reads, then the terminal one
    assert len(between) == 3
    assert_purged_once(fake)


def test_the_wait_is_bounded():
    fake = Lingering(polls=10**6)
    session, _, clock = make(fake)
    with session:
        session.run(QUERY, CHUNKS)
    assert fake.reads_after_stop < 30
    assert_purged_once(fake)


def test_a_failing_stop_still_purges_and_does_not_raise():
    session, fake, _ = make(FakeNomad(stop_status=500))
    with session:
        assert session.run(QUERY, CHUNKS) == fake.stdout
    assert_purged_once(fake)


def test_purge_runs_when_the_body_exits_with_an_unrelated_exception():
    session, fake, _ = make()
    with pytest.raises(KeyboardInterrupt):
        with session:
            session.run(QUERY, CHUNKS)
            raise KeyboardInterrupt
    assert_purged_once(fake)


# --- no leakage ---------------------------------------------------------------------------------------------


def test_nothing_secret_is_in_any_error_or_repr():
    session, fake, _ = make(FakeNomad("exit3"))
    with pytest.raises(ContainerError) as excinfo:
        with session:
            session.run(QUERY, CHUNKS)
    err = excinfo.value
    assert TOKEN not in f"{err} {err!r} {session!r}"
    assert "LEAK" not in f"{err} {err!r} {session!r}"
