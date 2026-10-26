import hashlib
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from experiments.governed_strings.sql import GovernedSQL, source_identity


def test_source_identity_streams_the_snapshot(tmp_path, monkeypatch):
    database = tmp_path / "source.sqlite"
    payload = b"sqlite snapshot\x00" * 100_000
    database.write_bytes(payload)

    def no_whole_file_read(*_args, **_kwargs):
        raise AssertionError("source hashing must not load the whole database")

    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    identity = source_identity(database)
    assert identity == {
        "kind": "sqlite",
        "location": database.resolve().as_uri(),
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_sql_assignment_requires_the_exact_runtime_type(tmp_path, monkeypatch):
    class_name_collision = type("int", (), {})
    monkeypatch.setitem(sqlite3.converters, "FAKE_INT", lambda _raw: class_name_collision())
    database = tmp_path / "source.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value FAKE_INT)")
        connection.execute("INSERT INTO items VALUES (1)")

    resolution = SimpleNamespace(
        columns={"value": {"kind": "int"}},
        policies={"value": object()},
        assignment_id="assignment",
        packet=None,
    )
    gateway = SimpleNamespace(
        resolver=SimpleNamespace(resolve=lambda *_args, **_kwargs: resolution),
        actor="analyst",
        application="analytics",
        purpose="internal",
    )
    gateway._decision = lambda *_args: "decision"
    gateway._append = lambda *_args, **_kwargs: None

    def no_publication(*_args, **_kwargs):
        raise AssertionError("a value with the wrong runtime type must not be published")

    gateway.originate_batch = no_publication
    consumer = SimpleNamespace(
        resolver=SimpleNamespace(resolve=lambda *_args, **_kwargs: None),
        actor="analyst",
        application="analytics",
        purpose="internal",
    )
    driver = GovernedSQL(
        database,
        gateway=gateway,
        consumer=consumer,
        edition_publisher=SimpleNamespace(),
        directory=tmp_path / "editions",
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    with pytest.raises(TypeError, match="SQL type differs"):
        driver.execute("SELECT value FROM items").fetchall()
