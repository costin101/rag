import json
import traceback
from pathlib import Path

import httpx
import pytest

import nomad_client
from nomad_client import NomadClient, build_job
from normalizer import ContainerError, ContextChunk
from tests.fake_nomad import NAMESPACE, TOKEN, FakeNomad, settings

JOB = "rag-session-9b1f3c52-8d1e-4c7a-9a55-0f3c2d1b7e10"
ALLOC = "38f571c2-36f3-c1de-b4a4-c479a4456745"


def client_for(fake, **overrides):
    return NomadClient(settings(**overrides), transport=httpx.MockTransport(fake))


def body():
    return build_job(settings(), JOB, "q", [ContextChunk(id="c1", text="t", score=0.5, source_id="s1")])


def registered(fake):
    client = client_for(fake)
    client.register(body())
    return client


# --- the request shape ----------------------------------------------------------------


def test_register_puts_the_job_body_with_the_namespace():
    fake = FakeNomad()
    client_for(fake).register(body())
    (request,) = fake.requests
    assert (request.method, request.url.path) == ("PUT", "/v1/jobs")
    assert dict(request.url.params) == {"namespace": NAMESPACE}
    assert json.loads(request.content)["Job"]["ID"] == JOB


def test_token_header_is_sent_only_when_a_token_is_set():
    with_token, without = FakeNomad(), FakeNomad()
    client_for(with_token).register(body())
    client_for(without, NOMAD_TOKEN=None).register(body())
    assert with_token.requests[0].headers["X-Nomad-Token"] == TOKEN
    assert "X-Nomad-Token" not in without.requests[0].headers


def test_no_namespace_means_no_namespace_parameter():
    fake = FakeNomad()
    client_for(fake, NOMAD_NAMESPACE=None).register(body())
    assert dict(fake.requests[0].url.params) == {}


def test_deregister_is_a_purge_with_the_namespace():
    fake = FakeNomad()
    client_for(fake).deregister(JOB)
    request = fake.requests[-1]
    assert (request.method, request.url.path) == ("DELETE", f"/v1/job/{JOB}")
    assert dict(request.url.params) == {"purge": "true", "namespace": NAMESPACE}


def test_stop_is_a_delete_without_purge_with_the_namespace():
    fake = FakeNomad()
    client_for(fake).stop(JOB)
    request = fake.requests[-1]
    assert (request.method, request.url.path) == ("DELETE", f"/v1/job/{JOB}")
    assert dict(request.url.params) == {"namespace": NAMESPACE}


def test_status_reads_use_the_job_paths():
    fake = FakeNomad()
    client = registered(fake)
    client.evaluations(JOB)
    allocs = client.allocations(JOB)
    assert [r.url.path for r in fake.requests[1:]] == [
        f"/v1/job/{JOB}/evaluations",
        f"/v1/job/{JOB}/allocations",
    ]
    assert allocs[0]["ClientStatus"] == "complete"


def test_gc_puts_the_allocation_gc_path():
    fake = FakeNomad()
    client_for(fake).gc_allocation(ALLOC)
    (request,) = fake.requests
    assert (request.method, request.url.path) == ("PUT", f"/v1/client/allocation/{ALLOC}/gc")


def test_gc_is_best_effort():
    class Down(FakeNomad):
        def __call__(self, request):
            raise httpx.ConnectError("down")

    client_for(Down()).gc_allocation(ALLOC)  # does not raise


def test_stdout_is_read_plain_for_the_task():
    fake = FakeNomad()
    out = client_for(fake).read_stdout(ALLOC, 4096)
    (request,) = fake.requests
    assert request.url.path == f"/v1/client/fs/logs/{ALLOC}"
    assert dict(request.url.params) == {
        "task": "gen",
        "type": "stdout",
        "plain": "true",
        "namespace": NAMESPACE,
    }
    assert out == fake.stdout


def test_stdout_read_stops_at_the_byte_cap():
    fake = FakeNomad(stdout=b"x" * 5_000_000)
    out = client_for(fake).read_stdout(ALLOC, 1024)
    assert len(out) == 1025  # one byte past the cap is enough for the parser to refuse it
    assert fake.streamed_chunks <= 2


# --- ids never reach a path unchecked ------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "../x", "38f571c2", ALLOC.upper(), ALLOC + "/gc", ALLOC[:-1] + "?", "a/../" + ALLOC, 5, None]
)
def test_an_allocation_id_must_be_a_uuid(bad):
    fake = FakeNomad()
    client = client_for(fake)
    with pytest.raises(ContainerError):
        client.read_stdout(bad, 1024)
    client.gc_allocation(bad)  # best effort, but still never sent
    assert fake.requests == []


@pytest.mark.parametrize("bad", ["", "rag-session-x", "../v1/jobs", JOB + "/x", JOB + "?purge=false", None])
def test_a_job_id_must_be_a_session_id(bad):
    fake = FakeNomad()
    client = client_for(fake)
    for call in (client.deregister, client.stop, client.evaluations, client.allocations):
        with pytest.raises(ContainerError):
            call(bad)
    assert fake.requests == []


# --- failures are fixed text -------------------------------------------------------------


def test_a_5xx_is_a_fixed_error_without_the_body():
    err = pytest.raises(ContainerError, client_for(FakeNomad(register_status=500)).register, body())
    assert str(err.value) == "nomad request failed"
    assert "LEAK" not in f"{err.value!r}"


def test_logs_non_200_is_a_fixed_error():
    with pytest.raises(ContainerError) as excinfo:
        client_for(FakeNomad(logs_status=404)).read_stdout(ALLOC, 1024)
    assert str(excinfo.value) == "container output could not be read"


def test_a_connection_failure_midway_is_a_fixed_error():
    with pytest.raises(ContainerError) as excinfo:
        client_for(FakeNomad(logs_fail_midway=True)).read_stdout(ALLOC, 1024)
    assert str(excinfo.value) == "container output could not be read"
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def test_a_redirect_is_not_followed():
    fake = FakeNomad(redirect_register=True)
    with pytest.raises(ContainerError):
        client_for(fake).register(body())
    assert len(fake.requests) == 1


@pytest.mark.parametrize("payload", ["not json", "[1]", "{}", "null"])
def test_a_malformed_status_body_is_a_fixed_error(payload):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    client = NomadClient(settings(), transport=transport)
    with pytest.raises(ContainerError) as excinfo:
        client.allocations(JOB)
    assert str(excinfo.value) == "nomad returned an unexpected result"


def test_the_token_is_in_no_error_repr_or_traceback_local():
    fake = FakeNomad(register_status=500)
    client = client_for(fake)
    with pytest.raises(ContainerError) as excinfo:
        client.register(body())
    err = excinfo.value
    trace = traceback.TracebackException.from_exception(err, capture_locals=True)
    assert TOKEN not in f"{err} {err!r} {client!r} " + "".join(trace.format())
    assert err.__cause__ is None and err.__context__ is None


# --- transport hardening ---------------------------------------------------------------------


def test_the_http_client_ignores_the_environment_and_never_follows_redirects():
    http = client_for(FakeNomad())._http
    assert http.trust_env is False
    assert http.follow_redirects is False
    assert http.timeout.read == 5.0 and http.timeout.connect == 5.0


def test_a_missing_address_is_a_fixed_error():
    with pytest.raises(ContainerError) as excinfo:
        NomadClient(settings(NOMAD_ADDR=None, NOMAD_TOKEN=None))
    assert str(excinfo.value) == "nomad address is not configured"


def test_an_unreadable_ca_file_is_a_fixed_error(tmp_path):
    with pytest.raises(ContainerError) as excinfo:
        NomadClient(settings(NOMAD_CACERT=str(tmp_path / "absent.pem")))
    assert str(excinfo.value) == "nomad CA certificate could not be loaded"


def test_tls_verification_is_never_switched_off():
    assert "verify=False" not in Path(nomad_client.__file__).read_text().replace(" ", "")
