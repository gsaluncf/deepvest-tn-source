import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest
import tn

from experiments.pvldb_evaluation import run


@pytest.mark.parametrize("changed", ["version", "native_module_sha256"])
def test_selected_build_mismatch_stops_before_sql(tmp_path, monkeypatch, changed):
    native = Path(tn._native.__file__)
    record = {
        "version": importlib.metadata.version("tn-proto"),
        "wheel_sha256": "unused-before-workflow",
        "install": {"native_module_sha256": hashlib.sha256(native.read_bytes()).hexdigest()},
    }
    if changed == "version":
        record["version"] = "wrong-version"
    else:
        record["install"]["native_module_sha256"] = "0" * 64
    build = tmp_path / "build.json"
    build.write_text(json.dumps(record))
    calls = []

    def connect(*args, **kwargs):
        calls.append(args)
        raise AssertionError("SQL must not start for a different native build")

    monkeypatch.setattr(run.sqlite3, "connect", connect)
    result = run.run_case(tmp_path / "run", "primitive", sdk_build_record=build)
    assert not result["completed"]
    assert result["error"]["stage"] == "preflight"
    assert result["error"]["type"] == "RuntimeError"
    assert calls == []
    assert not (tmp_path / "run/sdk-build.json").exists()


def test_matching_selected_build_reaches_sql(tmp_path, monkeypatch):
    record = {
        "version": importlib.metadata.version("tn-proto"),
        "wheel_sha256": "unused-before-workflow",
        "install": {"native_module_sha256": hashlib.sha256(Path(tn._native.__file__).read_bytes()).hexdigest()},
    }
    build = tmp_path / "build.json"
    build.write_text(json.dumps(record))
    calls = []

    def connect(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("Stop after confirming SQL was reached")

    monkeypatch.setattr(run.sqlite3, "connect", connect)
    result = run.run_case(tmp_path / "run", "primitive", sdk_build_record=build)
    assert len(calls) == 1
    assert result["error"]["stage"] == "setup"
    assert (tmp_path / "run/sdk-build.json").read_bytes() == build.read_bytes()
