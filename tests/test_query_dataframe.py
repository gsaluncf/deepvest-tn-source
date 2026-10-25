import json
from decimal import Decimal

import pytest

from experiments.pvldb_evaluation.run import run_case
from tn_polars.registry import UnityCatalogRegistry


@pytest.fixture
def catalog(monkeypatch):
    volumes, calls = {}, []
    def request(self, method, path, payload=None):
        calls.append((method, path))
        if path in {"/catalogs", "/schemas"}:
            return {}
        if method == "POST" and path == "/volumes":
            full = f"{payload['catalog_name']}.{payload['schema_name']}.{payload['name']}"
            volumes[full] = dict(payload)
            return volumes[full]
        if method == "GET" and path.startswith("/volumes/"):
            return volumes[path.removeprefix("/volumes/")]
        raise AssertionError((method, path))
    monkeypatch.setattr(UnityCatalogRegistry, "_request", request)
    return calls


@pytest.mark.parametrize("kind", ["int", "float", "Decimal"])
def test_query_batch_keeps_types_claims_lineage_and_opa(tmp_path, catalog, kind):
    root = tmp_path / kind
    result = run_case(root, "primitive", rows=5, numeric_type=kind)
    assert result["completed"] and result["verified"], result
    assert result["verification"]["source_editions"] == 2
    graph = json.loads((root / "dag.json").read_text())
    cells = [node for node in graph["nodes"] if node["operation"] == "select"]
    assert len(cells) == 10
    assert len({node["reference"]["object_id"] for node in cells}) == 2
    assert len({node["id"] for node in cells}) == 10
    run = json.loads((root / "run.json").read_text())
    facts = run["evaluation"]["dataframe"][0]
    assert len(facts["opa_decisions"]) == 12
    assert facts["opa_decisions"][-1]["request"]["aggregate_support"]["group_sizes"] == [5]
    assert len(facts["input_columns"]["name"]) == len(facts["input_columns"]["value"]) == 1
    assert len([call for call in catalog if call[0] == "GET" and ".source_" in call[1]]) == 1
    assert len([call for call in catalog if call[0] == "POST" and call[1] == "/volumes"]) == 2
    assert len([call for call in catalog if call[0] == "GET"]) == 2


def test_source_batch_boundary_keeps_all_rows_distinct(tmp_path, catalog):
    root = tmp_path / "boundary"
    result = run_case(root, "primitive", rows=65)
    assert result["completed"] and result["verified"], result
    assert result["verification"]["source_editions"] == 4
    graph = json.loads((root / "dag.json").read_text())
    cells = [node for node in graph["nodes"] if node["operation"] == "select"]
    assert len(cells) == len({node["id"] for node in cells}) == 130
    run = json.loads((root / "run.json").read_text())
    facts = run["evaluation"]["dataframe"][0]
    assert facts["opa_decisions"][-1]["request"]["aggregate_support"]["group_sizes"] == [65]
    assert len([call for call in catalog if call[0] == "GET"]) == 2


def test_derived_singleton_keeps_original_batch_row_support(tmp_path, catalog, monkeypatch):
    from experiments.pvldb_evaluation import run as module
    original = module.dataframe_calculate
    def calculate(context, rows):
        rows = list(rows)
        # A value-preserving computation creates a singleton publication while
        # retaining row 1 of the original shared source publication.
        rows[1] = (rows[1][0], rows[1][1] + 0)
        assert rows[1][1].reference.field == "value"
        return original(context, rows)
    monkeypatch.setattr(module, "dataframe_calculate", calculate)
    result = run_case(tmp_path / "derived", "primitive", rows=5)
    assert result["completed"] and result["verified"], result
