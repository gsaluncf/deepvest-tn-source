"""Declared primitive facts, bound inside native signed publications.

No payload opening happens here. Process-local Python hashes are cached only
where authorized source/compute code already held the content; never exported.
"""

import hashlib

from .string import MetadataUnavailable, TNString
from .numeric import TNNumeric
from .numeric_codec import NUMERIC_TYPES, encode, equality_key
import json

_PROCESS_HASHES = {}


def fingerprint(content):
    if type(content) is str:
        encoded = b"str\x00" + content.encode("utf-8", "surrogatepass")
    elif type(content) is int:
        encoded = b"int\x00" + str(content).encode("ascii")
    elif type(content) in NUMERIC_TYPES:
        encoded = type(content).__name__.encode() + b"\x00" + encode(content).encode("ascii")
    else:
        raise TypeError("unsupported primitive facts")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def numeric_equality_fingerprint(content):
    key = equality_key(content)
    if key is None:
        return None
    if key[0] == "ratio" and key[2] == "1":
        return fingerprint(int(key[1]))
    return "sha256:" + hashlib.sha256(json.dumps(key).encode("ascii")).hexdigest()


def describe(content):
    identity = fingerprint(content)
    if type(content) is str or equality_key(content) is not None:
        _PROCESS_HASHES[identity] = hash(content)
    facts = {"kind": type(content).__name__, "fingerprint": identity,
             "fingerprint_scheme": "tn.primitive.content.v1"}
    if type(content) is str:
        facts["length"] = len(content)
    elif type(content) is int:
        facts.update(bit_length=content.bit_length(), bit_count=content.bit_count())
    else:
        facts["equality_fingerprint"] = numeric_equality_fingerprint(content)
    return facts


class DeclaredMetadata:
    def __init__(self, context):
        self.context = context

    def facts(self, value):
        _, binding, _ = self.context._metadata(value)
        facts = binding.get("declared")
        if not isinstance(facts, dict):
            raise MetadataUnavailable("publication has no declared primitive facts")
        return facts

    def length(self, value):
        return self.facts(value)["length"]

    def fingerprint(self, value):
        return self.facts(value)["fingerprint"]

    def python_hash(self, value):
        digest = self.fingerprint(value)
        if digest not in _PROCESS_HASHES:
            raise MetadataUnavailable("no plaintext-equivalent hash has been computed in this process")
        return _PROCESS_HASHES[digest]

    def equals(self, left, right):
        facts = self.facts(left)
        other = self.facts(right) if isinstance(right, (TNString, TNNumeric)) else None
        if facts["kind"] == "str":
            if other is not None:
                return other["kind"] == "str" and facts["fingerprint"] == other["fingerprint"]
            return type(right) is str and facts["fingerprint"] == fingerprint(right)
        right_kind = other["kind"] if other else type(right).__name__
        if right_kind == "str":
            return False
        if {facts["kind"], right_kind} == {"Decimal", "float"}:
            raise MetadataUnavailable("Decimal/float comparison uses the active decimal context")
        left_key = facts["fingerprint"] if facts["kind"] == "int" else facts.get("equality_fingerprint")
        if other:
            right_key = other["fingerprint"] if other["kind"] == "int" else other.get("equality_fingerprint")
        else:
            if type(right) is bool:
                right = int(right)
            if type(right) not in NUMERIC_TYPES:
                return False
            right_key = numeric_equality_fingerprint(right)
        if left_key is None or right_key is None:
            raise MetadataUnavailable("non-reflexive numbers require protected comparison")
        return left_key == right_key

    def compare(self, left, right):
        if self.equals(left, right):
            return 0
        raise MetadataUnavailable("no declared sort key; comparison needs the protected runtime")
