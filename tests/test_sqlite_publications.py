import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import tn

from deepvest_tn.publications import open_publication_store
from deepvest_tn.sqlite_publications import SQLitePublicationStore
from test_publications import POLICY, publication


@contextmanager
def database_connection(path):
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=PERSIST")
        with connection:
            yield connection
    finally:
        connection.close()


def pair():
    with tn.Session(POLICY, groups=["finance"]) as session:
        data = session.create({"amount": 42}, session.policy("portfolio.feed.data.v1"),
                              object_type="portfolio.feed.data.v1", group="finance")
        return data.snapshot, session.release(
            data, use=tn.UseContext("source", "analysis", "publish"),
            to="analytics", decide=lambda ctx: True)


@pytest.mark.parametrize("backend", ["files", "sqlite"])
def test_backend_preserves_exact_bytes_after_restart_and_rejects_reformatting(tmp_path, backend):
    source = publication()
    with open_publication_store(tmp_path, backend=backend) as store:
        assert store.put(source) == source.id
        assert store.put(source) == source.id
        altered = tn.GovernedObject.parse(json.dumps(source.envelope, indent=2))
        with pytest.raises(ValueError, match="immutable"):
            store.put(altered)
    with open_publication_store(tmp_path) as reopened:
        assert reopened.get(source.id) == source.forward()
        assert reopened.read(source.id).id == source.id


def test_sqlite_reads_detect_tampering_without_relying_on_cached_validation(tmp_path):
    source = publication()
    with SQLitePublicationStore(tmp_path) as store:
        store.put(source)
        assert store.get(source.id) == source.forward()
        with database_connection(store.database_path) as external:
            external.execute("UPDATE publications SET wire=? WHERE object_id=?",
                             (source.forward() + b" ", source.id))
        with pytest.raises(ValueError, match="digest"):
            store.get(source.id)
        changed = dict(source.envelope)
        changed["timestamp"] = "2026-01-01T00:00:00Z"
        wire = json.dumps(changed).encode()
        with database_connection(store.database_path) as external:
            external.execute("UPDATE publications SET wire=?, wire_sha256=? WHERE object_id=?",
                             (wire, hashlib.sha256(wire).hexdigest(), source.id))
        with pytest.raises(tn.governed.VerificationError):
            store.read(source.id)


def test_record_batch_is_atomic_and_immutable(tmp_path):
    first, second = pair(), pair()
    with SQLitePublicationStore(tmp_path) as store:
        store.put_record(*first, context_id="context", sequence=1)
        store.put_record(*first, context_id="context", sequence=1)
        with pytest.raises(ValueError, match="immutable"):
            store.put_record(*second, context_id="context", sequence=1)
        for obj in first:
            assert store.get(obj.id) == obj.forward()
        for obj in second:
            with pytest.raises(FileNotFoundError):
                store.get(obj.id)
        assert store.record_ids("context") == (first[1].id,)
        with database_connection(store.database_path) as observer:
            assert observer.execute("SELECT count(*) FROM publications").fetchone()[0] == 2
            assert observer.execute("SELECT count(*) FROM record_references").fetchone()[0] == 1


def test_failed_record_commit_does_not_advance_recorder_or_allow_computation(tmp_path, monkeypatch):
    from experiments.governed_strings import Assessment, Context, EvidenceRecorder, Request

    with SQLitePublicationStore(tmp_path) as store, tn.Session(POLICY, groups=["values", "evidence"]) as session:
        policy = session.policy("portfolio.feed.data.v1")
        recorder = EvidenceRecorder(session, store, policy)
        context = Context(session, recorder, lambda r: Assessment(True, "internal"),
                          application="test", purpose="analysis", actor="worker",
                          metadata_accept=lambda c: c.writer == session.did)
        request = Request("test", "analysis", "select", (), (), (policy,), ())
        decision = context._decision(request, "source")
        value = context.originate(5, policy, source={"column": "amount"}, request=request,
                                  decision_id=decision, call_id="source")
        before = recorder.head
        opened = []
        real_open = context._open_values

        def check_durable(*args):
            with SQLitePublicationStore(tmp_path) as observer:
                head = observer.record_ids(recorder.context_id)[-1]
                assert head == recorder.head
                assert recorder.read(head).body["kind"] == "Decision"
            opened.append(True)
            return real_open(*args)

        monkeypatch.setattr(context, "_open_values", check_durable)
        with context:
            assert (value + 1).reveal(context, "internal") == 6
        assert opened
        opened.clear()
        before = recorder.head
        with database_connection(store.database_path) as external:
            external.execute("CREATE TRIGGER fail_ref BEFORE INSERT ON record_references "
                             "BEGIN SELECT RAISE(ABORT, 'test write failure'); END")
        with context, pytest.raises(sqlite3.IntegrityError, match="test write failure"):
            value + 1
        assert not opened and recorder.head == before
        assert store.record_ids(recorder.context_id)[-1] == before


@pytest.mark.parametrize("commit", [False, True])
def test_process_crash_recovers_only_committed_records(tmp_path, commit):
    snapshot, released = pair()
    for name, obj in (("snapshot", snapshot), ("released", released)):
        (tmp_path / name).write_bytes(obj.forward())
    database = tmp_path / "store"
    with SQLitePublicationStore(database):
        pass
    code = '''
import os,sys,tn
from pathlib import Path
from deepvest_tn.sqlite_publications import SQLitePublicationStore
root=Path(sys.argv[1])
class CrashStore(SQLitePublicationStore):
    def _insert_reference(self, *args):
        super()._insert_reference(*args)
        if sys.argv[2] == 'False': os._exit(23)
store=CrashStore(root/'store')
store.put_record(tn.GovernedObject.read(root/'snapshot'), tn.GovernedObject.read(root/'released'),
                 context_id='crash', sequence=1)
os._exit(23)
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(commit)],
                            capture_output=True, text=True, env=os.environ.copy(), timeout=20)
    assert result.returncode == 23, result.stderr
    with SQLitePublicationStore(database) as recovered:
        assert recovered.record_ids("crash") == ((released.id,) if commit else ())
        for obj in (snapshot, released):
            if commit:
                assert recovered.get(obj.id) == obj.forward()
            else:
                with pytest.raises(FileNotFoundError):
                    recovered.get(obj.id)


def test_independent_connections_can_write_and_read_without_losing_records(tmp_path):
    objects = [publication() for _ in range(8)]
    with SQLitePublicationStore(tmp_path) as first, SQLitePublicationStore(tmp_path) as second:
        def save(store):
            for obj in objects:
                store.put(obj)
                assert store.get(obj.id) == obj.forward()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(save, store) for store in (first, second)]
            for job in jobs:
                job.result(timeout=20)
        with database_connection(first.database_path) as observer:
            assert observer.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert observer.execute("SELECT count(*) FROM publications").fetchone()[0] == len(objects)


def test_backend_selection_and_durability_settings_are_explicit(tmp_path):
    with SQLitePublicationStore(tmp_path) as store:
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "persist"
        assert store._connection.execute("PRAGMA synchronous").fetchone()[0] == 3
        with pytest.raises(ValueError, match="backend"):
            open_publication_store(tmp_path, backend="files")
    with pytest.raises(ValueError, match="backend"):
        open_publication_store(tmp_path, backend="unknown")


@pytest.mark.parametrize("backend", ["files", "sqlite"])
def test_independent_verifier_can_open_storage_read_only(tmp_path, backend):
    source = publication()
    with open_publication_store(tmp_path, backend=backend) as store:
        store.put(source)
    with open_publication_store(tmp_path, read_only=True) as verifier:
        assert verifier.get(source.id) == source.forward()
        with pytest.raises(PermissionError):
            verifier.put(source)
    assert not (tmp_path / "missing").exists()
    with pytest.raises(sqlite3.OperationalError):
        open_publication_store(tmp_path / "missing", backend="sqlite", read_only=True)
    assert not (tmp_path / "missing").exists()
