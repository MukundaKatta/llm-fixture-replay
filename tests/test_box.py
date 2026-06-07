import asyncio
import json
import os

import pytest

from llm_fixture_replay import FixtureBox, FixtureMissError, Mode

# ---------- basic record + replay ----------


def test_record_then_replay_roundtrip(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(prompt):
        calls["n"] += 1
        return f"echo: {prompt}"

    record_box = FixtureBox(path=str(path), mode=Mode.RECORD)
    wrapped = record_box.wrap(real)
    assert wrapped("hello") == "echo: hello"
    assert calls["n"] == 1
    assert path.exists()

    # new box, REPLAY mode, same path: should not call real
    replay_box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = replay_box.wrap(real)
    assert wrapped("hello") == "echo: hello"
    assert calls["n"] == 1  # not incremented


def test_replay_miss_raises_fixture_miss_error(tmp_path):
    path = tmp_path / "fix.jsonl"
    path.write_text("")  # empty file exists

    box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = box.wrap(lambda x: x)
    with pytest.raises(FixtureMissError) as exc:
        wrapped("missing")
    assert isinstance(exc.value.hash, str) and len(exc.value.hash) == 64
    assert "missing" in exc.value.args_repr


def test_replay_miss_when_file_does_not_exist(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = box.wrap(lambda x: x)
    with pytest.raises(FixtureMissError):
        wrapped("a")


def test_fixture_miss_error_payload_useful(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = box.wrap(lambda x: x)
    try:
        wrapped("the-prompt")
    except FixtureMissError as e:
        assert e.hash and len(e.hash) == 64
        assert "the-prompt" in e.args_repr
        # error message has a hash prefix and args excerpt
        msg = str(e)
        assert "no fixture matches" in msg
        assert e.hash[:12] in msg
    else:
        pytest.fail("expected FixtureMissError")


# ---------- AUTO ----------


def test_auto_records_on_miss_then_replays(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(prompt):
        calls["n"] += 1
        return {"data": prompt.upper()}

    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap(real)

    # first call: miss -> record
    assert wrapped("hi") == {"data": "HI"}
    assert calls["n"] == 1

    # second call: hit -> replay (no real call)
    assert wrapped("hi") == {"data": "HI"}
    assert calls["n"] == 1

    stats = box.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["records"] == 1


def test_auto_records_distinct_hashes(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap(lambda x: x + "!")
    assert wrapped("a") == "a!"
    assert wrapped("b") == "b!"
    # both saved
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


# ---------- DISABLED ----------


def test_disabled_always_calls_real_and_writes_nothing(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(x):
        calls["n"] += 1
        return x

    box = FixtureBox(path=str(path), mode=Mode.DISABLED)
    wrapped = box.wrap(real)
    wrapped("a")
    wrapped("b")
    wrapped("a")
    assert calls["n"] == 3
    assert not path.exists()
    assert box.stats() == {"hits": 0, "misses": 0, "records": 0}


# ---------- custom hasher ----------


def test_custom_hasher_used_for_matching(tmp_path):
    """Only the `prompt` kwarg should drive matching; request_id ignored."""
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(prompt, request_id):
        calls["n"] += 1
        return f"resp:{prompt}:{calls['n']}"

    def hash_prompt_only(*args, **kw):
        import hashlib

        return hashlib.sha256(kw["prompt"].encode()).hexdigest()

    box = FixtureBox(
        path=str(path),
        mode=Mode.AUTO,
        hasher=hash_prompt_only,
    )
    wrapped = box.wrap(real)

    first = wrapped(prompt="hello", request_id="req-1")
    assert calls["n"] == 1
    # different request_id, same prompt -> should hit, not call real again
    second = wrapped(prompt="hello", request_id="req-2")
    assert calls["n"] == 1
    assert first == second


def test_default_hasher_distinguishes_different_args(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(x):
        calls["n"] += 1
        return calls["n"]

    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap(real)
    assert wrapped("a") == 1
    assert wrapped("b") == 2
    assert wrapped("a") == 1  # replay


def test_kwarg_order_does_not_affect_hash(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    calls = {"n": 0}

    def real(**kw):
        calls["n"] += 1
        return calls["n"]

    wrapped = box.wrap(real)
    assert wrapped(a=1, b=2) == 1
    # different declaration order, same kwargs -> hit
    assert wrapped(b=2, a=1) == 1
    assert calls["n"] == 1


# ---------- duplicate hashes: newest wins ----------


def test_record_duplicates_keep_newest_on_lookup(tmp_path):
    path = tmp_path / "fix.jsonl"
    n = {"i": 0}

    def real(x):
        n["i"] += 1
        return f"v{n['i']}"

    record_box = FixtureBox(path=str(path), mode=Mode.RECORD)
    wrapped = record_box.wrap(real)
    assert wrapped("same") == "v1"
    assert wrapped("same") == "v2"  # RECORD always hits real

    # two lines on disk
    assert len(path.read_text().strip().splitlines()) == 2

    # REPLAY returns the newest
    replay_box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped_r = replay_box.wrap(real)
    assert wrapped_r("same") == "v2"


# ---------- fixture file format ----------


def test_fixture_file_is_valid_jsonl(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.RECORD)
    wrapped = box.wrap(lambda x: {"got": x})
    wrapped("alpha")
    wrapped("beta")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert set(entry) == {
            "hash",
            "args_repr",
            "kwargs_repr",
            "response_json",
            "recorded_at",
        }
        assert isinstance(entry["hash"], str)
        assert len(entry["hash"]) == 64  # sha256 hex


def test_record_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "fix.jsonl"
    box = FixtureBox(path=str(nested), mode=Mode.RECORD)
    wrapped = box.wrap(lambda x: x)
    wrapped("ok")
    assert nested.exists()


def test_record_handles_non_serializable_response_via_repr(tmp_path):
    """Unknown types fall back to repr rather than blowing up."""
    path = tmp_path / "fix.jsonl"

    class Weird:
        def __repr__(self):
            return "Weird()"

    box = FixtureBox(path=str(path), mode=Mode.RECORD)
    wrapped = box.wrap(lambda: Weird())
    wrapped()
    entry = json.loads(path.read_text().strip())
    assert entry["response_json"] == "Weird()"


# ---------- stats ----------


def test_stats_counts_correctly(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap(lambda x: x)

    wrapped("a")  # miss + record
    wrapped("a")  # hit
    wrapped("b")  # miss + record
    wrapped("a")  # hit

    stats = box.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    assert stats["records"] == 2


def test_stats_in_replay_only_mode(tmp_path):
    path = tmp_path / "fix.jsonl"
    # seed a fixture
    rec = FixtureBox(path=str(path), mode=Mode.RECORD)
    rec.wrap(lambda x: x)("a")

    box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = box.wrap(lambda x: x)
    wrapped("a")  # hit
    with pytest.raises(FixtureMissError):
        wrapped("z")  # miss
    stats = box.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["records"] == 0


# ---------- clear ----------


def test_clear_deletes_fixture_file(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.RECORD)
    box.wrap(lambda x: x)("a")
    assert path.exists()
    box.clear()
    assert not path.exists()


def test_clear_on_missing_file_is_safe(tmp_path):
    path = tmp_path / "never.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.RECORD)
    # should not raise
    box.clear()
    assert not path.exists()


# ---------- async ----------


async def test_async_wrap_records_then_replays(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    async def real(prompt):
        calls["n"] += 1
        await asyncio.sleep(0)
        return f"async:{prompt}"

    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap_async(real)

    assert await wrapped("hi") == "async:hi"
    assert calls["n"] == 1
    assert await wrapped("hi") == "async:hi"
    assert calls["n"] == 1  # replay, no real call


async def test_async_replay_miss_raises(tmp_path):
    path = tmp_path / "fix.jsonl"
    box = FixtureBox(path=str(path), mode=Mode.REPLAY)

    async def real(x):
        return x

    wrapped = box.wrap_async(real)
    with pytest.raises(FixtureMissError):
        await wrapped("missing")


async def test_async_wrap_accepts_plain_callable(tmp_path):
    """wrap_async also accepts a sync callable that returns a value."""
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    def real(prompt):
        calls["n"] += 1
        return f"sync:{prompt}"

    box = FixtureBox(path=str(path), mode=Mode.AUTO)
    wrapped = box.wrap_async(real)

    assert await wrapped("hi") == "sync:hi"
    assert calls["n"] == 1
    # second call replays without invoking real again
    assert await wrapped("hi") == "sync:hi"
    assert calls["n"] == 1


async def test_async_disabled_passthrough(tmp_path):
    path = tmp_path / "fix.jsonl"
    calls = {"n": 0}

    async def real(x):
        calls["n"] += 1
        return x

    box = FixtureBox(path=str(path), mode=Mode.DISABLED)
    wrapped = box.wrap_async(real)
    await wrapped("a")
    await wrapped("a")
    assert calls["n"] == 2
    assert not path.exists()


# ---------- corrupt lines tolerated ----------


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "fix.jsonl"
    # write a valid record then a corrupt line then a valid one
    box = FixtureBox(path=str(path), mode=Mode.RECORD)
    wrapped = box.wrap(lambda x: x)
    wrapped("good1")
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    wrapped("good2")

    rb = FixtureBox(path=str(path), mode=Mode.REPLAY)
    rb_wrapped = rb.wrap(lambda x: x)
    assert rb_wrapped("good1") == "good1"
    assert rb_wrapped("good2") == "good2"


# ---------- properties ----------


def test_path_and_mode_exposed(tmp_path):
    p = str(tmp_path / "fix.jsonl")
    box = FixtureBox(path=p, mode=Mode.REPLAY)
    assert box.path == p
    assert box.mode is Mode.REPLAY


def test_record_mode_appends_not_truncates(tmp_path):
    path = tmp_path / "fix.jsonl"
    box1 = FixtureBox(path=str(path), mode=Mode.RECORD)
    box1.wrap(lambda x: x)("a")
    # second box, same path
    box2 = FixtureBox(path=str(path), mode=Mode.RECORD)
    box2.wrap(lambda x: x)("b")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_empty_fixture_file_is_treated_as_no_match(tmp_path):
    path = tmp_path / "fix.jsonl"
    path.write_text("")
    assert os.path.exists(path)
    box = FixtureBox(path=str(path), mode=Mode.REPLAY)
    wrapped = box.wrap(lambda x: x)
    with pytest.raises(FixtureMissError):
        wrapped("anything")
