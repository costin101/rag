"""Single-query CLI for the API path: `python main.py QUERY`.

stdout is one JSON document (a `RAGResponse`). A failure prints exactly one line, `<code>: <message>`,
to stderr and exits 1; a usage error exits 2. No traceback, cause or SDK repr ever reaches the
output: every exception is converted here, and a non-`RAGError` prints a fixed line only.
"""

import os
import sys
from collections.abc import Callable, Mapping
from typing import TextIO

from config import Settings, load_settings
from normalizer import InvalidQueryError, RAGError, is_blank
from rag_api_fallback import ApiPath

USAGE = "usage: main.py QUERY"
_UNEXPECTED = "unexpected_error: the request failed unexpectedly"


def run(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    api_path: Callable[[Settings], object] | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    if len(argv) != 1:
        err.write(USAGE + "\n")
        return 2
    failure = None
    try:
        if is_blank(argv[0]):
            raise InvalidQueryError("the query must not be blank")
        settings = load_settings(os.environ if env is None else env)
        path = (api_path or ApiPath)(settings)
        out.write(path.answer(argv[0]).model_dump_json() + "\n")
        return 0
    except RAGError as exc:
        failure = f"{exc.code.value}: {exc.message}"
    except Exception:
        failure = _UNEXPECTED
    err.write(failure + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
