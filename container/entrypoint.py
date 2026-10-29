"""In-image stub generator (rule 7): stdlib only, imports nothing from the project.

Reads the job input the Nomad template wrote (base64 of JSON `{"query": str, "chunks":
[{"id": str, "text": str}, ...]}`), deletes it, and prints one JSON line
`{"v":1,"answer":str,"input_tokens":int,"output_tokens":int}`. Deterministic: the answer is
derived from the query and the chunk ids, token counts are word counts with a floor of one.
Swapping in a real model is an image change only.
"""

import base64
import json
import os
import sys

# The only place the image looks; nothing (argument, variable) can redirect it.
INPUT_PATH = "/local/input.json"

# The answer quotes only the start of the query so its size stays bounded.
_QUOTED_QUERY_CHARS = 80


def _words(*texts):
    # The floor is on the total: a count of zero would read as "nothing was processed".
    return max(1, sum(len(text.split()) for text in texts))


def _read(path):
    """The (query, chunks) in `path`, or None if missing or malformed. The file is removed."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        os.unlink(path)
        data = json.loads(base64.b64decode(raw, validate=True))
        query, chunks = data["query"], data["chunks"]
        valid = (
            type(query) is str
            and type(chunks) is list
            and chunks
            and all(
                type(c) is dict and type(c.get("id")) is str and type(c.get("text")) is str
                for c in chunks
            )
        )
        return (query, chunks) if valid else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(path):
    loaded = _read(path)
    if loaded is None:
        sys.stderr.write("entrypoint: invalid input\n")
        return 2
    query, chunks = loaded
    ids = ", ".join(c["id"] for c in chunks)
    answer = f'Stub answer to "{query[:_QUOTED_QUERY_CHARS]}" from {len(chunks)} chunks: {ids}'
    result = {
        "v": 1,
        "answer": answer,
        "input_tokens": _words(query, *(c["text"] for c in chunks)),
        "output_tokens": _words(answer),
    }
    # ASCII-only JSON: the line is the same bytes whatever the container's locale.
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(INPUT_PATH))
