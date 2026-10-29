import httpx
import pytest

from container_generator import NomadGenerator
from container_session import ContainerSession
from nomad_client import NomadClient
from normalizer import ContainerError, ContextChunk, Generation
from tests.fake_nomad import FakeClock, FakeNomad, assert_purged_once, settings

CHUNKS = [ContextChunk(id="c1", text="the sky is blue", score=0.5, source_id="s1")]
QUERY = "why is the sky blue?"


def generator(fake=None, **overrides):
    fake = fake or FakeNomad()
    cfg = settings(**overrides)
    clock = FakeClock()
    client = NomadClient(cfg, transport=httpx.MockTransport(fake))
    session = ContainerSession(cfg, client=client, clock=clock.now, sleep=clock.sleep)
    return NomadGenerator(session), session, fake


def test_a_valid_container_line_becomes_a_generation():
    gen, session, fake = generator()
    with session:
        result = gen.generate(QUERY, CHUNKS)
    assert result == Generation(text="Stub answer from the container.", input_tokens=12, output_tokens=5)
    assert_purged_once(fake)


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "​ㅤ"])
def test_a_blank_query_registers_nothing(query):
    gen, session, fake = generator()
    with pytest.raises(ContainerError):
        gen.generate(query, CHUNKS)
    session.close()
    assert fake.requests == []


@pytest.mark.parametrize("chunks", [[], ()])
def test_no_chunks_registers_nothing(chunks):
    gen, session, fake = generator()
    with pytest.raises(ContainerError):
        gen.generate(QUERY, chunks)
    session.close()
    assert fake.requests == []


BAD_STDOUT = {
    "garbage": b"Traceback (most recent call last):\n",
    "empty": b"",
    "truncated": b'{"v":1,"answer":"a","input_t',
    "extra key": b'{"v":1,"answer":"a","input_tokens":1,"output_tokens":1,"x":1}\n',
    "wrong version": b'{"v":2,"answer":"a","input_tokens":1,"output_tokens":1}\n',
    "two lines": b'{"v":1,"answer":"a","input_tokens":1,"output_tokens":1}\n' * 2,
}


@pytest.mark.parametrize("stdout", BAD_STDOUT.values(), ids=BAD_STDOUT.keys())
def test_exit_zero_with_unusable_stdout_is_a_fixed_error_and_still_purges(stdout):
    gen, session, fake = generator(FakeNomad(stdout=stdout))
    with pytest.raises(ContainerError) as excinfo:
        with session:
            gen.generate(QUERY, CHUNKS)
    assert str(excinfo.value) == "container output is not a valid v1 result"
    assert_purged_once(fake)


def test_stdout_over_the_configured_cap_is_refused():
    gen, session, fake = generator(
        FakeNomad(stdout=b"x" * 5_000_000), RAG_CONTAINER_MAX_OUTPUT_BYTES="1024"
    )
    with pytest.raises(ContainerError) as excinfo:
        with session:
            gen.generate(QUERY, CHUNKS)
    assert str(excinfo.value) == "container output too large"
    assert fake.streamed_chunks <= 2
    assert_purged_once(fake)


def test_a_container_failure_passes_through_with_its_fixed_text():
    gen, session, fake = generator(FakeNomad("exit3"))
    with pytest.raises(ContainerError) as excinfo:
        with session:
            gen.generate(QUERY, CHUNKS)
    assert str(excinfo.value) == "container exited with a failure"
