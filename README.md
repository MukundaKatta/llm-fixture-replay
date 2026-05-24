# llm-fixture-replay

[![PyPI](https://img.shields.io/pypi/v/llm-fixture-replay.svg)](https://pypi.org/project/llm-fixture-replay/)
[![Python](https://img.shields.io/pypi/pyversions/llm-fixture-replay.svg)](https://pypi.org/project/llm-fixture-replay/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**VCR-style record and replay for LLM calls.**

Hit a real API once, save the request and response to a JSON Lines fixture,
then replay the same response forever in your test suite. Hash-based matching
keeps fixtures stable across runs. Sync and async wrapping. Zero runtime deps.

## Install

```bash
pip install llm-fixture-replay
```

## Record once, replay forever

```python
from llm_fixture_replay import FixtureBox, Mode

box = FixtureBox(path="tests/fixtures/llm.jsonl", mode=Mode.AUTO)

def real_anthropic(prompt: str) -> str:
    # talks to the real API
    return anthropic_sdk.complete(prompt)

wrapped = box.wrap(real_anthropic)

# first run: hits the real API and records the response
# every later run: looks up the saved response by hash and returns it
response = wrapped("Hello, world")
```

`Mode.AUTO` is the common case: REPLAY when a matching fixture exists,
otherwise RECORD on the spot. Useful when you want to add new test cases
without changing the test code.

The other modes:

```python
Mode.RECORD    # always hit the real fn and append to the fixture file
Mode.REPLAY    # never hit the real fn; raise FixtureMissError on miss
Mode.DISABLED  # passthrough; never touch the fixture file
```

`Mode.REPLAY` is what you want in CI. A missing fixture raises
`FixtureMissError(hash, args_repr)` so the failure points straight at the
exact call that needs a recording.

## Async wrap

```python
import asyncio

box = FixtureBox(path="tests/fixtures/llm.jsonl", mode=Mode.AUTO)

async def real_openai(prompt: str) -> dict:
    return await openai_async.responses.create(input=prompt)

wrapped = box.wrap_async(real_openai)

response = asyncio.run(wrapped("Hello"))
```

`wrap_async` mirrors `wrap` but returns a coroutine function. The fixture
file format is the same.

## Custom hasher

The default hasher takes `sha256(repr((args, sorted kwargs)))`. That works
when every argument is part of the cache key. When some part of the request
should be ignored (a timestamp, a request id, a session uuid), pass your
own:

```python
import hashlib

def hash_prompt_only(*args, **kw) -> str:
    # ignore request_id, timestamp, etc; only the prompt drives matching
    return hashlib.sha256(kw["prompt"].encode()).hexdigest()

box = FixtureBox(
    path="tests/fixtures/llm.jsonl",
    mode=Mode.AUTO,
    hasher=hash_prompt_only,
)
```

## pytest fixture pattern

A convenient way to wire `Mode.AUTO` into a test suite:

```python
import pytest
from llm_fixture_replay import FixtureBox, Mode

@pytest.fixture
def llm(request):
    fixture_file = request.node.path.with_suffix(".llm.jsonl")
    box = FixtureBox(path=str(fixture_file), mode=Mode.AUTO)
    yield box.wrap(real_anthropic)

def test_summarize(llm):
    out = llm("summarize this paragraph: ...")
    assert "summary" in out.lower()
```

The first run records `test_summarize.llm.jsonl` next to the test file.
Subsequent runs replay from it. Commit the JSONL file to git.

## Stats and reset

```python
box.stats()    # {"hits": 5, "misses": 1, "records": 1}
box.clear()    # delete the fixture file
```

## What it does NOT do

- No HTTP. The library wraps any callable, including SDK methods, REST
  clients, or your own router. It does not parse provider responses.
- No automatic redaction. If your real response includes secrets and you do
  not want them in the fixture, redact in your wrapper before returning.
- No partial-match scoring. Matching is exact: same hash or
  `FixtureMissError`. Use a custom hasher when you need looser semantics.

## Siblings

Other libraries in the same family:

- [`agentsnap`](https://pypi.org/project/agentsnap/): snapshot equality on
  full agent traces (Jest-style snapshots, `AGENTSNAP_UPDATE=1` refreshes).
- [`llm-cache-mem`](https://pypi.org/project/llm-cache-mem/): in-process
  cache for production LLM calls (this lib is the test-time cousin).
- [`agenttap`](https://pypi.org/project/agenttap/): wire-level capture of
  the exact JSON sent over httpx, with credential redaction.

## License

MIT
