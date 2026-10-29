"""The container path's `Provider` (rule 5): a `NomadGenerator` is what `pipeline.answer` calls.

It sends the query and the retrieved chunks to the session's container and parses the stdout line
into the same `Generation` every provider returns, so `from_generation` produces the response
unchanged. A blank query or no chunks is refused before anything is registered.
"""

from collections.abc import Sequence

from container_session import ContainerSession
from normalizer import ContainerError, ContextChunk, Generation, is_blank, parse_container_output


class NomadGenerator:
    def __init__(self, session: ContainerSession):
        self._session = session

    def generate(self, query: str, chunks: Sequence[ContextChunk]) -> Generation:
        if not isinstance(query, str) or is_blank(query) or not chunks:
            raise ContainerError("container generation needs a query and context")
        stdout = self._session.run(query, chunks)
        return parse_container_output(stdout, max_bytes=self._session.max_output_bytes)
