"""Governed numeric operations over typed, encrypted native publication fields."""

from __future__ import annotations

import operator
from decimal import Decimal

from .numeric_codec import NUMERIC_TYPES, KINDS

from .string import MetadataUnavailable, OperationDenied, ValueReference, _active, _exposure


class TNNumeric:
    """A typed numeric handle; plaintext exists only inside authorized execution.

    The underlying Python type determines arithmetic, promotion and exceptions.
    Ordinary scalar conversions remain explicit governed exposure.
    """

    __slots__ = ("reference", "decision_id", "result_id", "metadata_provider")

    def __init__(self, reference, decision_id=None, result_id=None, *, metadata_provider=None):
        if hasattr(self, "reference"):
            raise AttributeError("TNNumeric bindings are already initialized")
        if not isinstance(reference, ValueReference):
            raise TypeError("TNNumeric requires a native publication field reference")
        for name, value in (("reference", reference), ("decision_id", decision_id),
                            ("result_id", result_id), ("metadata_provider", metadata_provider)):
            object.__setattr__(self, name, value)

    @property
    def dtype(self):
        return KINDS[self._metadata().facts(self)["kind"]]

    def __setattr__(self, name, value):
        raise AttributeError("TNNumeric bindings are read-only")

    def __delattr__(self, name):
        raise AttributeError("TNNumeric bindings are read-only")

    @property
    def identity(self):
        return self.reference

    def same_incarnation(self, other):
        return isinstance(other, TNNumeric) and self.reference == other.reference

    def __repr__(self):
        return f"TNNumeric({self.reference.object_id}, field={self.reference.field!r}, sealed=True)"

    def _run(self, operation, *others, parameters=()):
        context = _active.get()
        if context is None:
            raise OperationDenied("a governed execution context is required")
        return context.perform(operation, (self, *others), parameters=parameters)

    def _metadata(self):
        context = _active.get()
        provider = context.string_metadata if context is not None else None
        if provider is None:
            provider = self.metadata_provider
        if provider is None:
            raise MetadataUnavailable("numeric metadata is unavailable")
        return provider

    @property
    def content_fingerprint(self):
        return self._metadata().fingerprint(self)

    def __hash__(self):
        return self._metadata().python_hash(self)

    def _compare(self, other, operation):
        if not isinstance(other, TNNumeric) and type(other) not in (*NUMERIC_TYPES, bool):
            return NotImplemented
        try:
            provider = self._metadata()
            if operation == "eq":
                return provider.equals(self, other)
            if self.dtype is complex or type(other) is complex or (isinstance(other, TNNumeric) and other.dtype is complex):
                raise MetadataUnavailable("complex ordering must use native numeric semantics")
            return {"lt": operator.lt, "le": operator.le,
                    "gt": operator.gt, "ge": operator.ge}[operation](
                        provider.compare(self, other), 0)
        except MetadataUnavailable:
            context = _active.get()
            if context is None:
                raise OperationDenied("comparison needs declared facts or a governed context") from None
            return context.compare(self, other, operation=operation)

    def __eq__(self, other):
        return self._compare(other, "eq")

    def __ne__(self, other):
        answer = self.__eq__(other)
        return NotImplemented if answer is NotImplemented else not answer

    def __lt__(self, other): return self._compare(other, "lt")
    def __le__(self, other): return self._compare(other, "le")
    def __gt__(self, other): return self._compare(other, "gt")
    def __ge__(self, other): return self._compare(other, "ge")
    def __bool__(self):
        try:
            return not self._metadata().equals(self, 0)
        except MetadataUnavailable:
            return self._run("truth")

    def _binary(self, operation, other, reverse=False):
        if isinstance(other, TNNumeric):
            context = _active.get()
            if context is None:
                raise OperationDenied("numeric arithmetic requires a governed context")
            values = (other, self) if reverse else (self, other)
            return context.perform(operation, values)
        if type(other) in (*NUMERIC_TYPES, bool):
            return self._run(operation, parameters=(other, "left" if reverse else "right"))
        return NotImplemented

    def __add__(self, other): return self._binary("add", other)
    def __radd__(self, other): return self._binary("add", other, True)
    def __sub__(self, other): return self._binary("subtract", other)
    def __rsub__(self, other): return self._binary("subtract", other, True)
    def __mul__(self, other):
        from .string import TNString
        if isinstance(other, TNString):
            return other.__mul__(self)
        return self._binary("multiply", other)
    def __rmul__(self, other): return self._binary("multiply", other, True)
    def __floordiv__(self, other): return self._binary("floor_divide", other)
    def __rfloordiv__(self, other): return self._binary("floor_divide", other, True)
    def __mod__(self, other): return self._binary("modulo", other)
    def __rmod__(self, other): return self._binary("modulo", other, True)
    def __and__(self, other): return self._binary("bit_and", other)
    def __rand__(self, other): return self._binary("bit_and", other, True)
    def __or__(self, other): return self._binary("bit_or", other)
    def __ror__(self, other): return self._binary("bit_or", other, True)
    def __xor__(self, other): return self._binary("bit_xor", other)
    def __rxor__(self, other): return self._binary("bit_xor", other, True)
    def __lshift__(self, other): return self._binary("left_shift", other)
    def __rlshift__(self, other): return self._binary("left_shift", other, True)
    def __rshift__(self, other): return self._binary("right_shift", other)
    def __rrshift__(self, other): return self._binary("right_shift", other, True)
    def __neg__(self): return self._run("negate")
    def __pos__(self): return self._run("positive")
    def __abs__(self): return self._run("absolute")
    def __invert__(self): return self._run("invert")

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            values, arguments = [self], []
            for value in (other, modulo):
                if isinstance(value, TNNumeric):
                    arguments.append({"input": len(values)})
                    values.append(value)
                elif type(value) in (*NUMERIC_TYPES, bool):
                    arguments.append({"literal": value})
                else:
                    return NotImplemented
            return self._run("power_mod", *values[1:], parameters=tuple(arguments))
        return self._binary("power", other)

    def __rpow__(self, other): return self._binary("power", other, True)

    def __truediv__(self, other):
        return self._binary("true_divide", other)
    def __rtruediv__(self, other): return self._binary("true_divide", other, True)

    def __divmod__(self, other):
        return self._binary("divmod", other)
    def __rdivmod__(self, other): return self._binary("divmod", other, True)

    def _integer_fact(self, name):
        if self.dtype is not int:
            raise AttributeError(f"{self.dtype.__name__} has no {name}")
        return self._metadata().facts(self)[name]

    def bit_length(self): return self._integer_fact("bit_length")
    def bit_count(self): return self._integer_fact("bit_count")
    def is_integer(self): return self._run("is_integer")
    def copy(self): return self._run("copy")
    def __copy__(self): return self.copy()
    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result
    def __reduce_ex__(self, protocol):
        raise TypeError("handoff requires native publications and evidence references, not pickle")
    def as_string(self): return self._run("to_string")
    def attest(self, statement, context):
        if type(statement) is not str or not statement.strip():
            raise ValueError("attestation requires an ordinary nonempty string")
        return context.perform("attest", (self,), parameters=(statement,))
    def reveal(self, context, destination):
        return context.perform("reveal", (self,), destination=destination)

    def _expose(self):
        context = _active.get()
        scope = _exposure.get()
        if context is None:
            raise OperationDenied("numeric conversion requires a governed context")
        destination = scope[1] if scope and scope[0] == context.recorder.context_id else None
        return self.reveal(context, destination)

    def __int__(self): return int(self._expose())
    def __index__(self): return operator.index(self._expose())
    def __float__(self): return float(self._expose())
    def __complex__(self): return complex(self._expose())
    def __str__(self): return str(self._expose())
    def __format__(self, spec): return format(self._expose(), spec)
    def __round__(self, ndigits=None):
        if isinstance(ndigits, TNNumeric):
            raise NotImplementedError("governed rounding precision needs a contributing-input binding")
        return self._run("round", parameters=() if ndigits is None else (operator.index(ndigits),))
    def __trunc__(self): return self._run("trunc")
    def __floor__(self): return self._run("floor")
    def __ceil__(self): return self._run("ceil")
    def conjugate(self): return self._run("conjugate")
    @property
    def real(self): return self._run("real")
    @property
    def imag(self): return self._run("imag")
    @property
    def numerator(self): return self._run("numerator")
    @property
    def denominator(self): return self._run("denominator")
    def as_integer_ratio(self): return self._run("as_integer_ratio")
    def to_bytes(self, *args, **kwargs):
        raise NotImplementedError("byte export requires an explicit governed disclosure contract")

    def quantize(self, exponent, rounding=None):
        if isinstance(exponent, TNNumeric):
            return self._run("quantize", exponent, parameters=(rounding,))
        if type(exponent) is not Decimal:
            raise TypeError("quantize requires a Decimal or governed numeric exponent")
        return self._run("quantize", parameters=(rounding, exponent))

    def sqrt(self): return self._run("sqrt")
    def normalize(self): return self._run("normalize")
    def exp(self): return self._run("exp")
    def ln(self): return self._run("ln")
    def log10(self): return self._run("log10")
