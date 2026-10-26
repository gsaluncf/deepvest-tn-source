"""Measurement-only wrappers around the application runtime.

Times are local process observations, not signed remote attestations.
Exclusive times subtract instrumented descendants; inclusive times overlap.
"""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter_ns

_phase = ContextVar("pvldb_measurement_phase", default="setup")


@contextmanager
def stage(name):
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


class Profiler:
    """Single-thread experiment instrumentation; removes all wrappers on exit."""

    def __init__(self):
        self._stack = []
        self._originals = []
        self._data = defaultdict(lambda: {"calls": 0, "failures": 0,
                                         "inclusive_ns": 0, "exclusive_ns": 0})

    def _wrap(self, cls, name, label):
        original = getattr(cls, name)

        @wraps(original)
        def measured(*args, **kwargs):
            current = [perf_counter_ns(), 0]
            self._stack.append(current)
            key = (_phase.get(), label)
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = perf_counter_ns() - current[0]
                self._stack.pop()
                if self._stack:
                    self._stack[-1][1] += elapsed
                item = self._data[key]
                item["calls"] += 1
                item["failures"] += int(failed)
                item["inclusive_ns"] += elapsed
                item["exclusive_ns"] += elapsed - current[1]

        self._originals.append((cls, name, original))
        setattr(cls, name, measured)

    def __enter__(self):
        from tn_governed.values.primitives import Context
        from tn_governed.values.evidence import EvidenceRecorder
        from tn_governed.publications import PublicationStore
        from tn_governed.sqlite_publications import SQLitePublicationStore
        from tn_polars.tn_boundary import PolicySet
        targets = [
            (Context, "_metadata", "primitive_metadata"),
            (Context, "_decision", "decision_and_obligations"),
            (Context, "_open_values", "native_admission_and_decryption"),
            (Context, "_publish", "value_publication"),
            (EvidenceRecorder, "append", "signed_record"),
            (PublicationStore, "put", "publication_persistence"),
            (PublicationStore, "put_record", "record_persistence"),
            (SQLitePublicationStore, "put", "publication_persistence"),
            (SQLitePublicationStore, "put_record", "record_persistence"),
            (PolicySet, "decide", "dataframe_policy_evaluation"),
        ]
        try:
            for target in targets:
                self._wrap(*target)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for cls, name, original in reversed(self._originals):
            setattr(cls, name, original)
        self._originals.clear()

    def export(self):
        return {
            "clock": "time.perf_counter_ns",
            "scope": "single-thread Python wall time; existing native calls included",
            "inclusive_times_overlap": True,
            "exclusive_times_subtract_instrumented_descendants": True,
            "components": [{"phase": phase, "component": component, **values}
                           for (phase, component), values in sorted(self._data.items())],
        }
