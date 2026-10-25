"""Developer-facing governed string interface; native execution lives behind Context."""

from __future__ import annotations

import operator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote, urlencode

from deepvest_tn.publications import object_digest

if TYPE_CHECKING:
    from .runtime import Context

_active: ContextVar[Context | None] = ContextVar("tn_string_context", default=None)
_exposure: ContextVar[tuple[str, str] | None] = ContextVar("tn_string_exposure", default=None)
_STRING_METHODS = frozenset(
    name for name, method in vars(str).items()
    if callable(method) and not name.startswith("_") and name != "maketrans"
)


class OperationDenied(PermissionError):
    """No completed permit exists for this protected step."""


class MetadataUnavailable(LookupError):
    """Declared facts cannot answer; the metadata provider must not decrypt."""


@dataclass(frozen=True)
class ValueReference:
    object_id: str
    field: str
    group: str = "strings"

    def __post_init__(self):
        object_digest(self.object_id)
        if not isinstance(self.field, str) or not self.field or self.group not in {"strings", "values"}:
            raise ValueError("invalid string field reference")

    def to_dict(self):
        return {"object_id": self.object_id, "group": self.group, "field": self.field}


class StringMetadataProvider(Protocol):
    """Interface for ordinary, ungoverned observations of logical string content.

    Providers resolve verified value references, not the str subclass's transport
    buffer. The live provider reads declared facts from native signed evidence.
    These methods return ordinary types from declared facts without opening the
    payload. If that is impossible, raise MetadataUnavailable. Never decrypt or
    call reveal here, and never measure/compare the locator. Verification failures
    must propagate as their own errors, not as missing metadata.
    """

    def length(self, value: TNString) -> int:
        """Python character count, including zero for an empty logical string."""
        ...

    def equals(self, left: TNString, right: str) -> bool:
        """Compare text, including with a plain str; provenance does not affect equality."""
        ...

    def compare(self, left: TNString, right: str) -> int:
        """Return -1, 0, or 1 in Python lexicographic order; a hash cannot supply this."""
        ...

    def python_hash(self, value: TNString) -> int:
        """Equal hash to the same plaintext str in THIS Python process, not a wire digest."""
        ...

    def fingerprint(self, value: TNString) -> str:
        """Content comparison fingerprint under the provider's common scheme, not identity."""
        ...


class TNString(str):
    """String interface over a native TN field, with sealed-reference characters.

    Ordinary operations refer to logical text. The builtin str buffer contains
    only a locator to retained TN material, including the evidence citations;
    it is neither the payload nor an independently authoritative TN envelope.
    Explicit base-class calls and foreign consumers may still read that locator.
    """

    reference: ValueReference
    decision_id: str | None
    result_id: str | None
    metadata_provider: StringMetadataProvider | None

    def __new__(cls, reference, decision_id=None, result_id=None, *, metadata_provider=None):
        if not isinstance(reference, ValueReference):
            raise TypeError("TNString requires a native publication field reference")
        locator = "tn-ref:" + "/".join(
            quote(part, safe="")
            for part in (reference.object_id, reference.group, reference.field)
        )
        citations = {key: value for key, value in (
            ("decision", decision_id), ("result", result_id),
        ) if value is not None}
        if citations:
            locator += "?" + urlencode(citations)
        obj = str.__new__(cls, locator)
        object.__setattr__(obj, "reference", reference)
        object.__setattr__(obj, "decision_id", decision_id)
        object.__setattr__(obj, "result_id", result_id)
        object.__setattr__(obj, "metadata_provider", metadata_provider)
        return obj

    def __setattr__(self, name, value):
        raise AttributeError("TNString bindings are read-only; operations create new incarnations")

    def __delattr__(self, name):
        raise AttributeError("TNString bindings are read-only")

    def __getattribute__(self, name):
        # Unsupported ordinary methods must not quietly transform the locator.
        if name in _STRING_METHODS and getattr(type(self), name) is getattr(str, name):
            raise NotImplementedError(f"governed str.{name} is not implemented yet")
        return super().__getattribute__(name)

    @property
    def identity(self):
        """Explicit incarnation identity; never the default meaning of equality."""
        return self.reference

    def same_incarnation(self, other) -> bool:
        return isinstance(other, TNString) and self.reference == other.reference

    def _metadata(self) -> StringMetadataProvider:
        context = _active.get()
        provider = context.string_metadata if context is not None else None
        if provider is None:
            provider = self.metadata_provider
        if provider is None:
            raise MetadataUnavailable("logical string metadata provider is not connected yet")
        return provider

    @property
    def content_fingerprint(self) -> str:
        return self._metadata().fingerprint(self)

    def __len__(self) -> int:
        return self._metadata().length(self)

    def __bool__(self) -> bool:
        return len(self) != 0

    def __eq__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        if self is other:
            return True
        return self._compare(other, "eq")

    def _compare(self, other, operation):
        try:
            provider = self._metadata()
            if operation == "eq":
                return provider.equals(self, other)
            order = provider.compare(self, other)
            relation = {"lt": operator.lt, "le": operator.le,
                        "gt": operator.gt, "ge": operator.ge}[operation]
            return relation(order, 0)
        except MetadataUnavailable:
            context = _active.get()
            if context is None:
                raise OperationDenied(
                    "comparison needs declared metadata or an active governed context"
                ) from None
            # This is a protected operation, never a metadata-provider decrypt.
            return context.compare(self, other, operation=operation)

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __lt__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return self._compare(other, "lt")

    def __le__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return self._compare(other, "le")

    def __gt__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return self._compare(other, "gt")

    def __ge__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return self._compare(other, "ge")

    def __hash__(self) -> int:
        return self._metadata().python_hash(self)

    def __repr__(self):
        return f"TNString({self.reference.object_id}, field={self.reference.field!r}, sealed=True)"

    def _run(self, operation, *others, parameters=()):
        context = _active.get()
        if context is None:
            raise OperationDenied("a governed execution context is required")
        return context.perform(operation, (self, *others), parameters=parameters)

    def copy(self):
        """Publish a new incarnation of the same text under the current context."""
        return self._run("copy")

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, protocol):
        raise TypeError("handoff requires native publications and evidence references, not pickle")

    @staticmethod
    def _ordinary_index(value):
        from .numeric import TNNumeric
        if isinstance(value, TNNumeric):
            raise NotImplementedError("governed index parameters need an explicit contributing-input binding")
        return operator.index(value)

    def as_int(self, base=10):
        """Parse text into a governed integer, preserving its contributing governance."""
        return self._run("to_int", parameters=(self._ordinary_index(base),))

    def strip(self, chars=None):
        if chars is not None and type(chars) is not str:
            raise TypeError("strip characters must be an ordinary string or None")
        return self._run("strip", parameters=(chars,))

    def replace(self, old, new, count=-1):
        if type(old) is not str or type(new) is not str:
            raise TypeError("replacement parameters must be ordinary strings")
        return self._run("replace", parameters=(old, new, self._ordinary_index(count)))

    def upper(self):
        return self._run("upper")

    def lower(self):
        return self._run("lower")

    def slice(self, start=None, end=None, step=None):
        bounds = tuple(None if value is None else self._ordinary_index(value)
                       for value in (start, end, step))
        return self._run("slice", parameters=bounds)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(key.start, key.stop, key.step)
        return self._run("index", parameters=(self._ordinary_index(key),))

    def __iter__(self):
        if _active.get() is None:
            raise OperationDenied("string iteration requires a governed execution context")
        return (self[index] for index in range(len(self)))

    def __reversed__(self):
        if _active.get() is None:
            raise OperationDenied("string iteration requires a governed execution context")
        return (self[index] for index in range(len(self) - 1, -1, -1))

    def concat(self, other):
        result = self.__add__(other)
        if result is NotImplemented:
            raise TypeError("concat requires a string")
        return result

    def __add__(self, other):
        if isinstance(other, TNString):
            return self._run("concat", other)
        if isinstance(other, str):
            return self._run("concat", parameters=(str.__str__(other), "right"))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, TNString):
            return other.__add__(self)
        if isinstance(other, str):
            return self._run("concat", parameters=(str.__str__(other), "left"))
        return NotImplemented

    def __mul__(self, count):
        from .numeric import TNNumeric
        if isinstance(count, TNNumeric):
            return self._run("repeat", count)
        return self._run("repeat", parameters=(operator.index(count),))

    __rmul__ = __mul__

    def __contains__(self, item):
        raise NotImplementedError("logical string membership is not implemented yet")

    def __mod__(self, other):
        raise NotImplementedError("governed percent formatting is not implemented yet")

    def __rmod__(self, other):
        raise NotImplementedError("governed percent formatting is not implemented yet")

    def attest(self, statement, context):
        if type(statement) is not str or not statement.strip():
            raise ValueError("attestation requires an ordinary nonempty statement")
        return context.perform("attest", (self,), parameters=(statement,))

    def reveal(self, context, destination):
        return context.perform("reveal", (self,), destination=destination)

    def __str__(self):
        context = _active.get()
        exposure = _exposure.get()
        if context is None:
            raise OperationDenied("str requires a governed execution context")
        destination = exposure[1] if exposure and exposure[0] == context.recorder.context_id else None
        # A scope supplies a destination; every conversion still obtains a fresh decision.
        return self.reveal(context, destination)

    def __format__(self, spec):
        # An ordinary formatted string is content exposure, like str(self).
        return format(str(self), spec)
