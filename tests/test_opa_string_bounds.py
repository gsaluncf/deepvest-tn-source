"""Read OPA's NUL-terminated UTF-8 output without copying unused Wasm memory."""

import pytest

from tn_polars.opa import OpaWasmEvaluator


CHUNK = 64 * 1024


class Memory:
    def __init__(self, raw):
        self.raw = raw
        self.reads = []

    def data_len(self, _store):
        return len(self.raw)

    def read(self, _store, start, end):
        self.reads.append((start, end))
        return self.raw[start:end]


def evaluator(raw):
    result = OpaWasmEvaluator.__new__(OpaWasmEvaluator)
    result.store = object()
    result.memory = Memory(raw)
    return result


@pytest.mark.parametrize("address", [-1, 5, True, 0.0, "0", None])
def test_invalid_opa_string_address_is_rejected_before_read(address):
    reader = evaluator(b"ok\0x")
    with pytest.raises(ValueError, match="outside Wasm memory"):
        reader._read_c_string(address)
    assert reader.memory.reads == []


@pytest.mark.parametrize("length", [CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK])
def test_terminator_at_chunk_boundary_stops_further_reads(length):
    reader = evaluator(b"x" * length + b"\0" + b"unused" * 30_000)
    assert reader._read_c_string(0) == "x" * length
    assert len(reader.memory.reads) == length // CHUNK + 1
    assert all(end - start <= CHUNK for start, end in reader.memory.reads)


def test_utf8_character_split_between_chunks_is_decoded_after_joining():
    expected = "x" * (CHUNK - 1) + "€done"
    reader = evaluator(expected.encode("utf-8") + b"\0" + b"unused" * 30_000)
    assert reader._read_c_string(0) == expected
    assert reader.memory.reads == [(0, CHUNK), (CHUNK, 2 * CHUNK)]


def test_empty_string_and_offset_address():
    reader = evaluator(b"prefix\0answer\0unused")
    assert reader._read_c_string(6) == ""
    assert reader._read_c_string(7) == "answer"


@pytest.mark.parametrize("address", [0, 2 * CHUNK + 3])
def test_missing_terminator_including_end_of_memory_is_rejected(address):
    reader = evaluator(b"x" * (2 * CHUNK + 3))
    with pytest.raises(ValueError, match="unterminated"):
        reader._read_c_string(address)


def test_invalid_utf8_is_not_replaced_or_accepted():
    with pytest.raises(UnicodeDecodeError):
        evaluator(b"\xff\0")._read_c_string(0)
