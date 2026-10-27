"""RAGResponse is the only caller-visible shape (rule 1): any change to it must be deliberate.

To accept an intentional contract change, regenerate the committed snapshot and review the diff:

    python -m tests.test_schema_snapshot
"""

import json
from pathlib import Path

from normalizer import RAGResponse

SNAPSHOT = Path(__file__).parent / "snapshots" / "rag_response.schema.json"


def current_schema_text() -> str:
    return json.dumps(RAGResponse.model_json_schema(), indent=2, sort_keys=True) + "\n"


def test_rag_response_schema_matches_snapshot():
    assert SNAPSHOT.exists(), "snapshot missing; run: python -m tests.test_schema_snapshot"
    assert json.loads(SNAPSHOT.read_text()) == json.loads(current_schema_text()), (
        "RAGResponse schema drifted from tests/snapshots/rag_response.schema.json; "
        "if intentional, run: python -m tests.test_schema_snapshot"
    )


def test_snapshot_top_level_properties_are_the_contract():
    props = set(json.loads(SNAPSHOT.read_text())["properties"])
    assert props == {"answer", "chunks", "sources", "usage", "metrics"}


if __name__ == "__main__":
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(current_schema_text())
    print(f"wrote {SNAPSHOT}")
