"""Transactional storage of exact TN publications and signed-record references."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import sqlite3
import threading

import tn

from .publications import PublicationStore, object_digest, validate_record_position


class SQLitePublicationStore(PublicationStore):
    """Keep encrypted wire bytes in SQLite; each read still verifies native integrity."""

    def __init__(self, root, *, read_only=False):
        super().__init__(root, read_only=read_only)
        if (self.root / "objects").exists() or (self.root / "index").exists():
            raise ValueError("requested backend differs from existing publication storage")
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "publications.sqlite3"
        self._lock = threading.RLock()
        location = self.database_path.resolve().as_uri() + "?mode=ro" if read_only else self.database_path
        self._connection = sqlite3.connect(location, uri=read_only, isolation_level=None,
                                           check_same_thread=False, timeout=5)
        try:
            if read_only:
                self._connection.execute("PRAGMA query_only=ON")
                if self._connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("unsupported publication database version")
                return
            # Retain the rollback journal instead of creating and deleting it on every commit.
            mode = self._connection.execute("PRAGMA journal_mode=PERSIST").fetchone()[0]
            self._connection.execute("PRAGMA synchronous=EXTRA")
            self._connection.execute("PRAGMA foreign_keys=ON")
            if mode != "persist" or self._connection.execute("PRAGMA synchronous").fetchone()[0] != 3:
                raise RuntimeError("durable SQLite settings were not applied")
            with self._transaction():
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    self._connection.execute("CREATE TABLE publications ("
                        "object_id TEXT PRIMARY KEY, wire BLOB NOT NULL, wire_sha256 TEXT NOT NULL) WITHOUT ROWID")
                    self._connection.execute("CREATE TABLE record_references ("
                        "context_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence > 0), "
                        "object_id TEXT NOT NULL REFERENCES publications(object_id), "
                        "PRIMARY KEY(context_id, sequence)) WITHOUT ROWID")
                    self._connection.execute("PRAGMA user_version=1")
                elif version != 1:
                    raise ValueError("unsupported publication database version")
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @contextmanager
    def _transaction(self):
        if self.read_only:
            raise PermissionError("publication store is read-only")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _encoded(publication):
        if not isinstance(publication, tn.GovernedObject):
            raise TypeError("publication storage requires a native GovernedObject")
        wire = publication.forward()
        verified = tn.GovernedObject.read(io.BytesIO(wire))
        object_digest(verified.id)
        return verified.id, wire, hashlib.sha256(wire).hexdigest()

    def _insert(self, encoded):
        identity, wire, digest = encoded
        self._connection.execute(
            "INSERT OR IGNORE INTO publications(object_id, wire, wire_sha256) VALUES (?, ?, ?)",
            encoded)
        retained = self._connection.execute(
            "SELECT wire, wire_sha256 FROM publications WHERE object_id=?", (identity,)).fetchone()
        if retained != (wire, digest):
            raise ValueError("immutable publication representation differs")

    def put(self, publication):
        encoded = self._encoded(publication)
        with self._transaction():
            self._insert(encoded)
        return encoded[0]

    def _insert_reference(self, context_id, sequence, identity):
        self._connection.execute("INSERT OR IGNORE INTO record_references VALUES (?, ?, ?)",
                                 (context_id, sequence, identity))
        stored = self._connection.execute(
            "SELECT object_id FROM record_references WHERE context_id=? AND sequence=?",
            (context_id, sequence)).fetchone()
        if stored != (identity,):
            raise ValueError("immutable record reference differs")

    def put_record(self, snapshot, publication, *, context_id, sequence):
        validate_record_position(context_id, sequence)
        before, after = self._encoded(snapshot), self._encoded(publication)
        with self._transaction():
            self._insert(before)
            self._insert(after)
            self._insert_reference(context_id, sequence, after[0])

    def record_ids(self, context_id):
        validate_record_position(context_id, 1)
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, object_id FROM record_references WHERE context_id=? ORDER BY sequence",
                (context_id,)).fetchall()
        for expected, (sequence, identity) in enumerate(rows, 1):
            if sequence != expected:
                raise ValueError("record sequence has a gap")
            object_digest(identity)
        return tuple(identity for _, identity in rows)

    def read(self, object_id):
        object_digest(object_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT wire, wire_sha256 FROM publications WHERE object_id=?", (object_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"publication is not stored: {object_id}")
        wire, digest = row
        if hashlib.sha256(wire).hexdigest() != digest:
            raise ValueError("retained publication wire digest differs from its immutable index")
        publication = tn.GovernedObject.read(io.BytesIO(wire))
        if publication.id != object_id or publication.forward() != wire:
            raise ValueError("retained publication has the wrong native identity")
        return publication

    def publication_sizes(self):
        with self._lock:
            return dict(self._connection.execute("SELECT object_id, length(wire) FROM publications"))
