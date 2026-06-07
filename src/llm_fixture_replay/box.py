"""Core FixtureBox implementation."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any


class Mode(str, Enum):
    """Behavior of a FixtureBox when its wrapped callable is invoked.

    * ``RECORD`` always calls the real function and appends the result to
      the fixture file. Existing entries with the same hash are superseded
      by the new one on lookup (most-recent wins).
    * ``REPLAY`` never calls the real function. Returns the saved response
      on a hash match; raises :class:`FixtureMissError` on miss.
    * ``AUTO`` tries REPLAY first, falling back to RECORD on miss. Useful
      when you want to add new cases without changing the test code.
    * ``DISABLED`` is a passthrough. The fixture file is never read or
      written. Useful as a flag flip in tests that want a real-network run.
    """

    RECORD = "RECORD"
    REPLAY = "REPLAY"
    AUTO = "AUTO"
    DISABLED = "DISABLED"


class FixtureMissError(LookupError):
    """Raised in REPLAY mode when no fixture matches the request hash.

    The error message includes the offending hash and a short ``args_repr``
    excerpt so the failure points directly at the call that needs to be
    re-recorded. The full payload is also available as attributes.
    """

    def __init__(self, hash_: str, args_repr: str):
        self.hash = hash_
        self.args_repr = args_repr
        excerpt = args_repr if len(args_repr) <= 120 else args_repr[:117] + "..."
        super().__init__(f"no fixture matches hash {hash_[:12]}... for call {excerpt}")


# A hasher receives the wrapped function's args and kwargs and returns a
# stable string used as the fixture lookup key.
Hasher = Callable[..., str]


def _default_hasher(*args: Any, **kwargs: Any) -> str:
    """SHA-256 of ``repr((args, sorted kwargs))``.

    Stable across runs as long as the inputs have deterministic ``repr``s.
    Sort kwargs so order on the call site does not affect the hash.
    """
    sorted_kwargs = dict(sorted(kwargs.items()))
    payload = repr((args, sorted_kwargs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FixtureBox:
    """VCR-style record and replay box wrapped around an LLM callable.

    Parameters
    ----------
    path:
        Path to the JSONL fixture file. Each line is one recording.
    mode:
        :class:`Mode` value controlling read/write behavior. Defaults to
        ``Mode.AUTO``.
    hasher:
        Optional custom hasher ``(*args, **kw) -> str``. The default hashes
        ``repr((args, sorted kwargs))``.
    """

    def __init__(
        self,
        path: str,
        mode: Mode = Mode.AUTO,
        hasher: Hasher | None = None,
    ) -> None:
        self._path = path
        self._mode = mode
        self._hasher: Hasher = hasher if hasher is not None else _default_hasher
        self._hits = 0
        self._misses = 0
        self._records = 0
        # serializes appends and re-reads against the fixture file
        self._lock = threading.Lock()

    # ---- public ----

    @property
    def path(self) -> str:
        return self._path

    @property
    def mode(self) -> Mode:
        return self._mode

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a sync wrapper around ``fn`` honoring this box's mode."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self._mode is Mode.DISABLED:
                return fn(*args, **kwargs)

            key = self._hasher(*args, **kwargs)
            args_repr = repr(args)
            kwargs_repr = repr(dict(sorted(kwargs.items())))

            if self._mode is Mode.REPLAY:
                hit = self._lookup(key)
                if hit is None:
                    self._misses += 1
                    raise FixtureMissError(key, args_repr)
                self._hits += 1
                return hit

            if self._mode is Mode.AUTO:
                hit = self._lookup(key)
                if hit is not None:
                    self._hits += 1
                    return hit
                # fall through to record
                self._misses += 1

            # RECORD path (or AUTO miss)
            result = fn(*args, **kwargs)
            self._record(key, args_repr, kwargs_repr, result)
            return result

        return wrapper

    def wrap_async(self, fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Return an async wrapper around an async ``fn`` honoring this
        box's mode. Mirrors :meth:`wrap` for coroutine functions.

        ``fn`` is normally a coroutine function, but a plain callable that
        returns an awaitable is also accepted; its return value is awaited
        only when it is actually awaitable."""

        async def _call(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self._mode is Mode.DISABLED:
                return await _call(*args, **kwargs)

            key = self._hasher(*args, **kwargs)
            args_repr = repr(args)
            kwargs_repr = repr(dict(sorted(kwargs.items())))

            if self._mode is Mode.REPLAY:
                hit = self._lookup(key)
                if hit is None:
                    self._misses += 1
                    raise FixtureMissError(key, args_repr)
                self._hits += 1
                return hit

            if self._mode is Mode.AUTO:
                hit = self._lookup(key)
                if hit is not None:
                    self._hits += 1
                    return hit
                self._misses += 1

            result = await _call(*args, **kwargs)
            self._record(key, args_repr, kwargs_repr, result)
            return result

        return wrapper

    def stats(self) -> dict[str, int]:
        """Return cumulative counters for this box.

        Keys: ``hits``, ``misses``, ``records``.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "records": self._records,
        }

    def clear(self) -> None:
        """Delete the fixture file if it exists. Stats counters are kept."""
        with self._lock, contextlib.suppress(FileNotFoundError):
            os.remove(self._path)

    # ---- internal ----

    def _lookup(self, key: str) -> Any | None:
        """Return the most recent recorded response for ``key``, or None."""
        if not os.path.exists(self._path):
            return None
        last_hit: Any | None = None
        found = False
        with self._lock, open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # skip corrupt lines rather than failing the test
                    continue
                if entry.get("hash") == key:
                    last_hit = entry.get("response_json")
                    found = True
        return last_hit if found else None

    def _record(
        self,
        key: str,
        args_repr: str,
        kwargs_repr: str,
        result: Any,
    ) -> None:
        """Append a fixture entry to the JSONL file."""
        entry = {
            "hash": key,
            "args_repr": args_repr,
            "kwargs_repr": kwargs_repr,
            "response_json": result,
            "recorded_at": _now_iso(),
        }
        # json.dumps with default=repr so unknown types fall back to repr
        # rather than blowing up the test run
        serialized = json.dumps(entry, default=repr)
        with self._lock:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(serialized + "\n")
            self._records += 1
