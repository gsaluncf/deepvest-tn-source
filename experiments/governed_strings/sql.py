"""Read-only SQL gateway: resolve governance, authorize, query, publish primitives.

The initial grammar deliberately supports direct-column SELECT ... FROM ... LIMIT
with an optional bound limit. Expressions, joins, and predicates need their own
control-dependency mapping and are rejected rather than assigned guessed lineage.
"""

from dataclasses import dataclass
from contextlib import closing
import hashlib
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .runtime import Request
from .string import OperationDenied

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SELECT = re.compile(r"^\s*select\s+([A-Za-z0-9_,\s]+)\s+from\s+([A-Za-z_][A-Za-z0-9_]*)"
                    r"(?:\s+limit\s+(\?|[0-9]+))?\s*;?\s*$", re.I)


def source_identity(path):
    path = Path(path).resolve()
    return {"kind": "sqlite", "location": path.as_uri(),
            "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass
class QueryResponse:
    data: list[dict]


class TableQuery:
    def __init__(self, driver, table):
        if not IDENTIFIER.fullmatch(table):
            raise ValueError("invalid table identifier")
        self.driver, self.table_name = driver, table
        self.columns, self.row_limit = None, 1

    def select(self, columns):
        names = [name.strip() for name in columns.split(",")]
        if not names or any(not IDENTIFIER.fullmatch(name) for name in names):
            raise ValueError("select requires direct column names")
        self.columns = names
        return self

    def limit(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("limit requires a nonnegative ordinary integer")
        self.row_limit = count
        return self

    def execute(self):
        if self.columns is None:
            raise ValueError("select columns before executing")
        cursor = self.driver.execute(
            f"SELECT {', '.join(self.columns)} FROM {self.table_name} LIMIT ?", (self.row_limit,))
        return QueryResponse([dict(zip(self.columns, row, strict=True)) for row in cursor.fetchall()])


class GovernedCursor:
    def __init__(self, columns, rows):
        self.description = tuple((name, None, None, None, None, None, None) for name in columns)
        self._rows, self._index = rows, 0

    def fetchone(self):
        if self._index == len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self):
        rows = self._rows[self._index:]
        self._index = len(self._rows)
        return rows


class GovernedSQL:
    def __init__(self, database, *, gateway, consumer, edition_publisher, directory, detect_types=0,
                 batch_size=64):
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError("SQL publication batch size must be between 1 and 64")
        self.database = Path(database).resolve()
        self.batch_size = batch_size
        self.detect_types = detect_types
        self.gateway, self.consumer, self.edition_publisher = gateway, consumer, edition_publisher
        self.directory = Path(directory)
        self.executions, self.editions, self.published_values = [], [], []

    def table(self, name):
        return TableQuery(self, name)

    def execute(self, sql, parameters=()):
        match = SELECT.fullmatch(sql)
        if match is None:
            raise NotImplementedError("this driver supports direct-column SELECT FROM LIMIT only")
        columns, table, limit = match.groups()
        columns = [item.strip() for item in columns.split(",")]
        if len(set(columns)) != len(columns) or any(not IDENTIFIER.fullmatch(item) for item in columns):
            raise ValueError("invalid or duplicated projection columns")
        if limit == "?":
            if len(parameters) != 1 or type(parameters[0]) is not int or parameters[0] < 0:
                raise ValueError("LIMIT ? requires one nonnegative integer")
        elif parameters:
            raise ValueError("no parameters are accepted without LIMIT ?")
        identity = source_identity(self.database)
        gateway = self.gateway
        resolution = gateway.resolver.resolve(table, columns, actor=gateway.actor,
            application=gateway.application, purpose=gateway.purpose, expected_source=identity)
        query_id = uuid4().hex
        request = Request(gateway.application, gateway.purpose, "select", (), (),
                          tuple(resolution.policies.values()), (), parameters=(sql, list(parameters), resolution.assignment_id))
        decision = gateway._decision(request, query_id)
        try:
            # The first SQL connection/statement happens after a durable, completed permit.
            with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, detect_types=self.detect_types)) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.execute("BEGIN")
                raw_rows = connection.execute(sql, parameters).fetchall()
                if source_identity(self.database) != identity:
                    raise OperationDenied("SQL source changed since its signed governance assignment")
            gateway._append("SQLRead", request, query_id, {"decision_id": decision,
                "query_id": query_id, "row_count": len(raw_rows), "source": identity,
                "assignment_id": resolution.assignment_id})
            self.executions.append({"query_id": query_id, "rows": len(raw_rows),
                                    "assignment_id": resolution.assignment_id})
            self.consumer.resolver.resolve(table, columns, actor=self.consumer.actor,
                application=self.consumer.application, purpose=self.consumer.purpose, expected_source=identity,
                packet=resolution.packet)
            rows = [[None] * len(columns) for _ in raw_rows]
            publications, handles, positions = [], [], []
            for start in range(0, len(raw_rows), self.batch_size):
                for column_index, column in enumerate(columns):
                    cells = []
                    for index in range(start, min(start + self.batch_size, len(raw_rows))):
                        content = raw_rows[index][column_index]
                        if type(content).__name__ != resolution.columns[column]["kind"]:
                            raise TypeError("SQL type differs from its signed column assignment")
                        source = {"table": table, "column": column, "row": index,
                              "query_id": query_id, "assignment_id": resolution.assignment_id,
                              "snapshot_sha256": identity["snapshot_sha256"]}
                        cells.append((content, source))
                    policy = resolution.policies[column]
                    values = gateway.originate_batch(cells, policy, request=request,
                                                      decision_id=decision, call_id=query_id)
                    publications.append((values[0], policy, cells[0][1]))
                    for value, (content, source) in zip(values, cells, strict=True):
                        handles.append((value.reference, value.decision_id, value.result_id, type(content).__name__))
                        positions.append((source["row"], column_index))
            if publications:
                editions = self.edition_publisher.publish_query_editions(query_id, publications,
                    directory=self.directory / query_id, application=self.consumer.application,
                    purpose=self.consumer.purpose)
                for info, transport in editions:
                    self.consumer.resolver.admit_edition(info, self.consumer, transport=transport)
                    self.editions.append(info)
                for position, value in zip(positions, self.consumer.hydrate_many(handles), strict=True):
                    rows[position[0]][position[1]] = value
                self.published_values.extend(value for row in rows for value in row)
            return GovernedCursor(columns, [tuple(row) for row in rows])
        except Exception as exc:
            gateway._append("Failure", request, query_id,
                             {"decision_id": decision, "error_type": type(exc).__name__})
            raise
