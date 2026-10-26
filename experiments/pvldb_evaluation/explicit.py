"""Explicit input propagation for the PVLDB comparison.

This experiment adapter uses the production Context for admission, decisions,
monitoring, native decryption/inclusion, publication, and signed records. Only
the developer surface differs: callers name every contributing handle and
compute with ordinary str/int arguments inside a trusted, pure callback.

The callback must use only its supplied arguments and the declared literals.
It must not perform I/O, open another value, or return plaintext to a caller.
Its result is immediately passed to the same encrypted publication path used
by primitive operators. This is an explicit-metadata computation interface,
not a sandbox for untrusted callbacks or another cryptographic implementation.

For example, inside ``with context``::

    ops = ExplicitOperations(context)
    upper = ops.execute("upper", (name,), lambda text: text.upper())
    total = ops.execute("add", (left, right), lambda a, b: a + b)
    label = ops.execute("to_string", (total,), str)

Literal operands need the same parameters and descriptors as Context.perform::

    total = ops.execute(
        "add", (amount,), lambda number: number + 250,
        parameters=(250, "right"),
        literals=({"parameter_index": 0, "position": "right", "claims": []},),
    )

Returned handles serve as explicit metadata references for subsequent calls;
the adapter never invokes their overloaded arithmetic or string operations.
SQL setup, attestation, final reveal, and the model/report workflow are shared
unchanged by both comparison modes.
"""

from uuid import uuid4

from experiments.governed_strings.numeric import TNNumeric
from experiments.governed_strings.numeric_codec import NUMERIC_TYPES
from experiments.governed_strings.string import OperationDenied, TNString, _active


class ExplicitOperations:
    """Run ordinary scalar computation with explicitly supplied contributors."""

    _ARITIES = {
        "upper": (1,),
        "concat": (1, 2),
        "add": (1, 2),
        "to_string": (1,),
        "slice": (1,),
    }

    def __init__(self, context):
        self.context = context

    def execute(self, operation, values, compute, *, parameters=(), literals=()):
        """Publish ``compute(*opened_values)`` under the Context checks.

        ``parameters`` are the same operation parameters recorded by the
        primitive path. ``literals`` describe ordinary operand positions; they
        never introduce source claims. For slice parameters, omitted descriptors
        receive Context.perform's ordinary operation-parameter representation.
        """
        if operation not in self._ARITIES:
            raise NotImplementedError(f"unsupported explicit operation: {operation}")
        if _active.get() is not self.context:
            raise OperationDenied("content operations require this active context")
        values, parameters = tuple(values), tuple(parameters)
        if len(values) not in self._ARITIES[operation]:
            raise ValueError(f"invalid contributing-input count for {operation}")
        if any(not isinstance(value, (TNString, TNNumeric)) for value in values):
            raise TypeError("contributors must be governed primitive handles")
        if not callable(compute):
            raise TypeError("compute must be a pure scalar callback")
        literal_inputs = [dict(item) for item in literals]
        if any(item.get("claims") != [] for item in literal_inputs):
            raise ValueError("ordinary literals have no source claims")
        if parameters and not literal_inputs:
            literal_inputs = [
                {"parameter_index": index, "role": "operation-parameter", "claims": []}
                for index in range(len(parameters))
            ]

        context = self.context
        call_id = uuid4().hex
        request, metadata, decision = context._prepare(
            operation, values, None, parameters, call_id,
        )
        try:
            claims = list(dict.fromkeys(claim.id for claim in request.claims))
            origins = []
            for _, binding, _ in metadata:
                for origin in binding["origins"]:
                    if origin not in origins:
                        origins.append(origin)

            opened = context._open_values(request, values)
            data = context._merge_opened(values, opened)
            content = compute(*(plain for _, plain in opened))
            if type(content) not in (str, *NUMERIC_TYPES):
                raise NotImplementedError("explicit computation must return an ordinary string or supported number")
            return context._publish(
                data, content=content, declared=None, claims=claims, origins=origins,
                values=values, request=request, decision_id=decision,
                call_id=call_id, literals=literal_inputs,
            )
        except Exception as exc:
            context._append("Failure", request, call_id, {
                "decision_id": decision, "error_type": type(exc).__name__,
            })
            raise
