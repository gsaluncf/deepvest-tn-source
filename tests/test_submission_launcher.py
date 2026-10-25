"""Checks for service ownership and mandatory model execution in the launcher."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


def test_full_runner_always_calls_model(tmp_path):
    unity = load("install_unity")
    with patch.dict(sys.modules, {"install_unity": unity}):
        runner = load("run_submission")
    args = SimpleNamespace(rows=1, numeric_type="Decimal", binary=tmp_path / "server",
                           model_path=tmp_path / "model.gguf", model_port=18325)
    command = runner.case_command("python", args, tmp_path / "output", "http://127.0.0.1:18080")
    assert "--with-model" in command
    assert command[command.index("--numeric-type") + 1] == "Decimal"
    assert command[command.index("--unity-url") + 1] == "http://127.0.0.1:18080"


def test_installer_does_not_replace_existing_container():
    unity = load("install_unity")
    with patch.object(unity, "docker", side_effect=["27", "owned-elsewhere"]) as docker:
        with pytest.raises(RuntimeError, match="already exists"):
            unity.install()
    assert all(call.args[0] not in {"run", "stop", "rm"} for call in docker.call_args_list)


def test_installer_stops_only_created_container_on_readiness_failure():
    unity = load("install_unity")
    with patch.object(unity, "docker", side_effect=["27", "", "pulled", "/home/unitycatalog", "new-container", "stopped"]) as docker:
        with patch.object(unity, "wait_ready", side_effect=RuntimeError("unavailable")):
            with pytest.raises(RuntimeError, match="unavailable"):
                unity.install()
    run = next(call.args for call in docker.call_args_list if call.args[0] == "run")
    assert "127.0.0.1:18080:8080" in run
    assert docker.call_args_list[-1].args == ("stop", "new-container")
