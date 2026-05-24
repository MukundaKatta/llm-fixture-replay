"""llm-fixture-replay - VCR-style record and replay for LLM calls.

Wrap an LLM-like callable. In RECORD mode, hit the real API and write the
request and response to a JSONL fixture file. In REPLAY mode, match the
request against fixtures and return the saved response without hitting the
API. AUTO mode replays when a fixture exists, records when it does not.

    from llm_fixture_replay import FixtureBox, Mode

    box = FixtureBox(path="tests/fixtures/llm.jsonl", mode=Mode.AUTO)

    def real_anthropic(prompt: str) -> str:
        return anthropic_sdk.complete(prompt)

    wrapped = box.wrap(real_anthropic)
    response = wrapped("Hello, world")

Hash-based matching keeps fixtures stable across runs. Pass a custom
``hasher`` to ignore request-specific fields like timestamps or request ids.

Sibling to ``agentsnap``, ``llm-cache-mem``, and ``agenttap``.
"""

from llm_fixture_replay.box import (
    FixtureBox,
    FixtureMissError,
    Mode,
)

__version__ = "0.1.0"

__all__ = [
    "FixtureBox",
    "FixtureMissError",
    "Mode",
    "__version__",
]
