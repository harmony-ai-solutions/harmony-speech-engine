"""Typed engine errors that are safe to surface as per-request failures.

Phase 5-1 (honest batching): the executor used to assert batch alignment with a
bare ``assert``/``AssertionError``, which either crashed the engine's background
loop or was indistinguishable from a programming bug. :class:`EngineBatchError`
is the "the engine built an inconsistent batch" error; the model runner
converts it into *per-request* error results, so a malformed batch degrades to
HTTP errors for the affected requests instead of taking the worker down.

It subclasses :class:`AssertionError` for backwards compatibility with the
Phase 0-3 handlers/tests that asserted alignment with ``pytest.raises(
AssertionError)``: the invariant is the same, only the type is now nameable and
catchable by the engine.
"""


class EngineError(Exception):
    """Base class for engine-level failures that map to request errors."""


class EngineBatchError(EngineError, AssertionError):
    """A batch invariant was violated (input/request misalignment, duplicates)."""
