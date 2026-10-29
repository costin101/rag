import traceback

import httpx
import pytest

import pipeline
from config import Settings
from container_generator import NomadGenerator
from container_session import ContainerSession
from nomad_client import NomadClient
from normalizer import ConfigError, ContainerError, Generation, InvalidQueryError, ProviderError
from rag_container_path import ContainerPath
from tests.contract import assert_valid_rag_response
from tests.fake_nomad import FakeClock, FakeNomad, assert_purged_once, settings
from tests.test_pipeline import StubGenerator, StubRetriever


def path_for(fake=None, retriever=None, **overrides):
    fake = fake or FakeNomad()
    cfg = settings(**overrides)

    def factory(config: Settings):
        clock = FakeClock()
        client = NomadClient(config, transport=httpx.MockTransport(fake))
        return ContainerSession(config, client=client, clock=clock.now, sleep=clock.sleep)

    return ContainerPath(cfg, retriever=retriever or StubRetriever(), session_factory=factory), fake


# --- the response contract ------------------------------------------------------------------------------


def test_the_container_path_returns_a_contract_valid_response():
    path, fake = path_for()
    r = path.answer("why is the sky blue?")
    assert_valid_rag_response(r)
    assert r.answer == "Stub answer from the container."
    assert (r.usage.input_tokens, r.usage.output_tokens) == (12, 5)
    assert_purged_once(fake)


def test_same_shape_chunks_and_sources_as_the_api_provider_stub():
    path, _ = path_for()
    container = path.answer("why?")
    api = pipeline.answer(
        "why?",
        StubRetriever(),
        StubGenerator(Generation(text="Stub answer from the container.", input_tokens=12, output_tokens=5)),
    )
    assert_valid_rag_response(api)
    assert container.chunks == api.chunks
    assert container.sources == api.sources
    assert container.usage == api.usage
    assert container.answer == api.answer
    assert container.model_dump(exclude={"metrics"}) == api.model_dump(exclude={"metrics"})


def test_the_retrieved_chunks_are_what_the_container_is_given():
    path, fake = path_for()
    path.answer("why?")
    import base64
    import json

    (spec,) = fake.specs
    template = spec["Job"]["TaskGroups"][0]["Tasks"][0]["Templates"][0]["EmbeddedTmpl"]
    sent = json.loads(base64.b64decode(template))
    assert sent["query"] == "why?"
    assert [c["id"] for c in sent["chunks"]] == ["c1", "c2"]


def test_top_k_comes_from_settings():
    retriever = StubRetriever()
    path, _ = path_for(retriever=retriever, RAG_TOP_K="3")
    path.answer("q")
    assert retriever.calls == [("q", 3)]


# --- failures stay inside the public error set --------------------------------------------------------------


def test_a_container_failure_becomes_the_fixed_provider_error_and_purges():
    path, fake = path_for(FakeNomad("exit3"))
    with pytest.raises(ProviderError) as excinfo:
        path.answer("why?")
    err = excinfo.value
    assert str(err) == "provider_failed: generation failed unexpectedly"
    assert err.__cause__ is None and err.__context__ is None
    assert_purged_once(fake)


def test_every_container_failure_is_the_same_public_error():
    messages = set()
    for fake in (FakeNomad("exit3"), FakeNomad("unplaceable"), FakeNomad("badimage"), FakeNomad(stdout=b"junk")):
        path, _ = path_for(fake)
        with pytest.raises(ProviderError) as excinfo:
            path.answer("why?")
        messages.add(str(excinfo.value))
    assert messages == {"provider_failed: generation failed unexpectedly"}


def test_pipeline_maps_a_raising_nomad_generator_without_changing_pipeline():
    class Raising:
        def generate(self, query, chunks):
            raise ContainerError("container exited with a failure")

    with pytest.raises(ProviderError) as excinfo:
        pipeline.answer("why?", StubRetriever(), Raising())
    assert str(excinfo.value) == "provider_failed: generation failed unexpectedly"
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def test_a_blank_query_registers_nothing():
    path, fake = path_for()
    with pytest.raises(InvalidQueryError):
        path.answer("  ")
    assert fake.requests == []


def test_nothing_secret_reaches_a_failure_traceback():
    path, _ = path_for(FakeNomad("exit3"))
    with pytest.raises(ProviderError) as excinfo:
        path.answer("why?")
    trace = traceback.TracebackException.from_exception(excinfo.value, capture_locals=True)
    assert "LEAK" not in "".join(trace.format())


# --- construction ------------------------------------------------------------------------------------------------


def test_missing_nomad_settings_are_all_named_at_construction():
    with pytest.raises(ConfigError) as excinfo:
        ContainerPath(settings(NOMAD_ADDR=None, NOMAD_TOKEN=None, RAG_CONTAINER_IMAGE=None), retriever=StubRetriever())
    text = str(excinfo.value)
    assert "NOMAD_ADDR: required but not set" in text
    assert "RAG_CONTAINER_IMAGE: required but not set" in text


@pytest.mark.parametrize("missing", ["NOMAD_ADDR", "RAG_CONTAINER_IMAGE"])
def test_each_missing_variable_is_named_alone(missing):
    overrides = {missing: None}
    if missing == "NOMAD_ADDR":
        overrides["NOMAD_TOKEN"] = None
    other = "RAG_CONTAINER_IMAGE" if missing == "NOMAD_ADDR" else "NOMAD_ADDR"
    with pytest.raises(ConfigError) as excinfo:
        ContainerPath(settings(**overrides), retriever=StubRetriever())
    assert missing in str(excinfo.value) and other not in str(excinfo.value)


def test_default_wiring_builds_a_real_session_from_settings(monkeypatch):
    path = ContainerPath(settings(), retriever=StubRetriever())
    session = path._session_factory(settings())
    assert isinstance(session, ContainerSession)
    session.close()


def test_the_generator_is_a_provider_for_the_session():
    gen = NomadGenerator(ContainerSession(settings(), client=NomadClient(settings())))
    assert callable(gen.generate)
