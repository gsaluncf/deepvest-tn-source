"""Private lazy row provenance, materialized together with released business data.

Source row ordinals survive joins as lists. Aggregation records distinct support
for each source and preserves earlier group obligations through further grouping.
These columns are never exposed as business data or accepted from callers.
"""

from dataclasses import dataclass, replace
import json
from uuid import uuid4

import polars as pl


def _name():
    return "__tn_support_" + uuid4().hex


def _flat(name):
    return pl.col(name).list.explode(keep_nulls=False, empty_as_null=False).drop_nulls()


SCALAR, ROW, REDUCED, MIXED, UNKNOWN = "scalar", "row", "reduced", "mixed", "unknown"


def expression_shape(expression):
    """Prove scalar, row-valued, or reduced output for a closed expression set.

    Unknown expression nodes (windows, UDFs, slices, filtered reductions, etc.)
    cannot establish group support. They remain usable with explicit empty rules.
    MIXED retains a proven row/reducer composition so an outer reducer can make
    it scalar; its presence alone never proves a scalar aggregate result.
    """
    def combine(parts):
        shapes = set(parts)
        if UNKNOWN in shapes:
            return UNKNOWN
        if MIXED in shapes or {ROW, REDUCED} <= shapes:
            return MIXED
        if ROW in shapes:
            return ROW
        if REDUCED in shapes:
            return REDUCED
        return SCALAR

    def visit(node):
        if node == "Len":
            return REDUCED
        if not isinstance(node, dict) or len(node) != 1:
            return UNKNOWN
        kind, value = next(iter(node.items()))
        if kind == "Column":
            return ROW
        if kind == "Literal":
            # Polars serializes Series and container literals too. Only these
            # primitive scalar encodings prove one value without a source root.
            scalar_types = {"String", "Boolean", "Null", "Int8", "Int16", "Int32", "Int64", "Int128",
                            "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "Date", "Time"}
            if isinstance(value, dict) and len(value) == 1:
                variant, literal = next(iter(value.items()))
                if isinstance(literal, dict) and len(literal) == 1:
                    literal_type = next(iter(literal))
                    if variant == "Dyn" and literal_type in {"Int", "Float", "Str"}:
                        return SCALAR
                    if variant == "Scalar" and literal_type in scalar_types:
                        return SCALAR
            return UNKNOWN
        if kind == "Alias":
            return visit(value[0])
        if kind == "Cast":
            dtype = value["dtype"].get("Literal") if isinstance(value["dtype"], dict) else value["dtype"]
            if not isinstance(dtype, str) or dtype not in {"String", "Boolean", "Null", "Int8", "Int16", "Int32", "Int64", "Int128",
                                      "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "Date", "Time"}:
                return UNKNOWN
            return visit(value["expr"])
        if kind == "BinaryExpr":
            return combine([visit(value["left"]), visit(value["right"])])
        if kind == "Ternary":
            return combine([visit(value[key]) for key in ("predicate", "truthy", "falsy")])
        if kind == "Agg":
            operation, argument = next(iter(value.items()))
            if operation not in {"Sum", "Mean", "Median", "Min", "Max", "First", "Last", "Count", "NUnique", "Std", "Var", "Quantile"}:
                return UNKNOWN
            if isinstance(argument, dict) and "input" in argument:
                argument = argument["input"]
            elif operation in {"Std", "Var"}:
                argument = argument[0]
            elif operation == "Quantile":
                if visit(argument["quantile"]) not in {SCALAR, REDUCED}:
                    return UNKNOWN
                argument = argument["expr"]
            return REDUCED if visit(argument) != UNKNOWN else UNKNOWN
        if kind == "Function":
            function = value["function"]
            name = next(iter(function)) if isinstance(function, dict) else function
            if name == "Quantile":
                # Polars 1.44 represents quantile as a function rather than
                # Agg.Quantile. Prove both the source and scalar parameter.
                arguments = value["input"]
                if (len(arguments) != 2 or visit(arguments[0]) == UNKNOWN
                        or visit(arguments[1]) not in {SCALAR, REDUCED}):
                    return UNKNOWN
                return REDUCED
            if name not in {"Round", "RoundSF", "Abs", "Negate", "Clip", "Floor", "Ceil", "Sqrt", "Pow", "FillNull", "Boolean"}:
                return UNKNOWN
            if name == "Boolean" and function["Boolean"] not in {"IsNull", "IsNotNull", "IsFinite", "IsInfinite", "IsNan", "IsNotNan", "Not"}:
                return UNKNOWN
            return combine([visit(item) for item in value["input"]])
        return UNKNOWN

    try:
        return visit(json.loads(expression.meta.serialize(format="json")))
    except Exception:
        return UNKNOWN


def expression_kind(expression):
    """Adapter compatibility view: contains a reducer, and proven output shape."""
    shape = expression_shape(expression)
    return shape in {REDUCED, MIXED}, shape in {SCALAR, ROW, REDUCED}


@dataclass(frozen=True)
class Support:
    frame: pl.LazyFrame
    ids: dict
    groups: dict
    policy_sources: dict
    aggregated: bool = False
    known: bool = True
    shape_known: bool = True

    @classmethod
    def source(cls, frame, source, policies):
        name = _name()
        tracked = frame.with_row_index(name).with_columns(
            pl.concat_list(pl.concat_str(pl.lit(source + "::"), pl.col(name))).alias(name))
        return cls(tracked, {name: source}, {}, {policy: frozenset({source}) for policy in policies})

    @property
    def hidden(self):
        return list(self.ids) + list(self.groups)

    def apply(self, frame):
        return replace(self, frame=frame)

    def unknown(self, frame):
        return Support(frame, {}, {}, self.policy_sources, self.aggregated, False, False)

    def project(self, expressions):
        if all(expression_shape(expr) == SCALAR for expr in expressions):
            # Polars returns one row for a literal-only projection. Collapse
            # provenance alongside it instead of broadcasting the literal to
            # the original row count; prior aggregate obligations still apply.
            metadata = [_flat(name).implode().alias(name) for name in self.hidden]
            return self.apply(self.frame.select(*expressions, *metadata))
        return self.apply(self.frame.select(*expressions, *self.hidden))

    def aggregate(self, keys, expressions):
        known = self.known and all(expression_shape(key) in {SCALAR, ROW} for key in keys)
        # A non-reduction in GroupBy.agg creates a list of raw values, not an
        # aggregate eligible for a group threshold.
        known = known and all(expression_shape(expr) in {SCALAR, REDUCED} for expr in expressions)
        metadata = [_flat(name).unique().implode().alias(name) for name in self.ids]
        metadata += [_flat(name).implode().alias(name) for name in self.groups]
        groups = dict(self.groups)
        for policy, sources in self.policy_sources.items():
            count_name = _name()
            names = [name for name, source in self.ids.items() if source in sources]
            if not names:
                known = False
                continue
            contributors = pl.concat_list([pl.col(name).fill_null(pl.lit([], dtype=pl.List(pl.String))) for name in names])
            metadata.append(contributors.list.explode(keep_nulls=False, empty_as_null=False)
                            .drop_nulls().n_unique().cast(pl.UInt64).implode().alias(count_name))
            groups[count_name] = policy
        if keys:
            frame = self.frame.group_by(*keys).agg(*expressions, *metadata)
        else:
            frame = self.frame.select(*expressions, *metadata)
        return Support(frame, self.ids, groups, self.policy_sources, True, known, self.shape_known)

    def joined(self, other, keys, how):
        # Independent copies of the same frame may share hidden names. Rename
        # the right metadata while retaining source identity for deduplication.
        mapping = {name: _name() for name in other.hidden}
        right = other.frame.rename(mapping)
        ids = self.ids | {mapping[name]: source for name, source in other.ids.items()}
        groups = self.groups | {mapping[name]: source for name, source in other.groups.items()}
        policies = dict(self.policy_sources)
        for policy, sources in other.policy_sources.items():
            policies[policy] = policies.get(policy, frozenset()) | sources
        joined = self.frame.join(right, on=keys, how=how)
        if how in {"semi", "anti"}:
            # Right rows determine membership but Polars does not expose their
            # support in these joins. Required aggregate facts fail closed.
            return Support(joined, self.ids, self.groups, policies,
                           self.aggregated or other.aggregated, False,
                           self.shape_known and other.shape_known)
        # Multiple branches of one source must be unioned before counting.
        for source in set(ids.values()):
            names = [name for name, value in ids.items() if value == source]
            if len(names) > 1:
                target = _name()
                joined = joined.with_columns(pl.concat_list([
                    pl.col(name).fill_null(pl.lit([], dtype=pl.List(pl.String))) for name in names
                ]).list.drop_nulls().list.unique().alias(target)).drop(names)
                ids = {name: value for name, value in ids.items() if name not in names} | {target: source}
        return Support(joined, ids, groups, policies, self.aggregated or other.aggregated,
                       self.known and other.known, self.shape_known and other.shape_known)

    def facts(self, materialized, policy_ids=None):
        sizes = []
        for name, source in self.groups.items():
            if policy_ids is None or source in policy_ids:
                sizes.extend(size for group in materialized[name].to_list() if group is not None for size in group)
        return {"known": self.known or (not self.aggregated and self.shape_known), "aggregated": self.aggregated,
                "group_sizes": sizes, "output_rows": materialized.height}
