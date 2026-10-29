"""Single-query CLI: `python main.py [--path api|container] QUERY`; the default path is the API.

stdout is one JSON document (a `RAGResponse`). A failure ends with a line `<code>: <message>` on
stderr (the embedding hub client may print progress or warnings before it) and exits 1; a usage error exits 2. No traceback, cause or SDK repr ever reaches the
output: every exception is converted here, and a non-`RAGError` prints a fixed line only.
"""

import os
import sys
from collections.abc import Callable, Mapping
from typing import TextIO

from config import Settings, load_settings
from normalizer import InvalidQueryError, RAGError, is_blank
from rag_api_fallback import ApiPath
from rag_container_path import ContainerPath

USAGE = "usage: main.py [--path api|container] QUERY"
_PATHS = ("api", "container")
_UNEXPECTED = "unexpected_error: the request failed unexpectedly"


def _parse(argv: list[str]) -> tuple[str, str] | None:
    """(path, query) from the arguments, or None when they do not fit the usage."""
    if len(argv) == 1:
        return "api", argv[0]
    if len(argv) == 3 and argv[0] == "--path" and argv[1] in _PATHS:
        return argv[1], argv[2]
    return None


def run(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    api_path: Callable[[Settings], object] | None = None,
    container_path: Callable[[Settings], object] | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parsed = _parse(argv)
    if parsed is None:
        err.write(USAGE + "\n")
        return 2
    chosen, query = parsed
    failure = None
    try:
        if is_blank(query):
            raise InvalidQueryError("the query must not be blank")
        settings = load_settings(os.environ if env is None else env)
        build = (api_path or ApiPath) if chosen == "api" else (container_path or ContainerPath)
        out.write(build(settings).answer(query).model_dump_json() + "\n")
        return 0
    except RAGError as exc:
        failure = f"{exc.code.value}: {exc.message}"
    except Exception:
        failure = _UNEXPECTED
    err.write(failure + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
