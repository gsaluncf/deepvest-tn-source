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
    with patch.dict(sys.modules, {"install_unity": unity, "dependency_paths": load("dependency_paths")}):
        runner = load("run_submission")
    args = SimpleNamespace(rows=1, numeric_type="Decimal", binary=tmp_path / "server",
                           model_path=tmp_path / "model.gguf", model_port=18325)
    command = runner.case_command("python", args, tmp_path / "output", "http://127.0.0.1:18080")
    assert "--with-model" in command
    assert command[command.index("--numeric-type") + 1] == "Decimal"
    assert command[command.index("--unity-url") + 1] == "http://127.0.0.1:18080"


def test_server_discovery_keeps_shared_library_directory(tmp_path):
    resolver = load("dependency_paths")
    directory = tmp_path / "tn-llama"
    directory.mkdir()
    server = directory / "llama-server"
    server.write_bytes(b"server")
    assert resolver.resolve_dependency("binary", directory=tmp_path, environ={}) == server.resolve()


def test_server_package_copies_shared_library_targets(tmp_path):
    packager = load("package_llama")
    source, destination = tmp_path / "build", tmp_path / "installed"
    source.mkdir()
    (source / "llama-server").write_bytes(b"server")
    (source / "libllama-server-impl.so").write_bytes(b"governed implementation")
    packager.package(source, destination, "cpu")
    import json
    import hashlib
    record = json.loads((destination / "build.json").read_text())
    assert {item["file"] for item in record["files"]} == {"llama-server", "libllama-server-impl.so"}
    for item in record["files"]:
        assert hashlib.sha256((destination / item["file"]).read_bytes()).hexdigest() == item["sha256"]


def test_missing_dependency_stops_before_server_build(tmp_path):
    unity = load("install_unity")
    with patch.dict(sys.modules, {"install_unity": unity, "dependency_paths": load("dependency_paths")}):
        runner = load("run_submission")
    with patch.object(sys, "argv", ["run_submission.py", "--build-server", "cpu",
                                   "--tn-wheel", str(tmp_path / "missing.whl")]):
        with patch.object(runner.subprocess, "run") as execute:
            with pytest.raises(SystemExit) as error:
                runner.main()
            assert error.value.code == 2
            execute.assert_not_called()


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

def test_dependency_paths_support_spaces_and_explicit_precedence(tmp_path):
    paths = load("dependency_paths")
    selected = tmp_path / "my model.gguf"
    selected.touch()
    assert paths.resolve_dependency("model_path", selected, directory=tmp_path,
                                    environ={"TN_LLAMA_MODEL": "missing"}) == selected.resolve()


def test_invalid_explicit_path_does_not_fall_back(tmp_path):
    paths = load("dependency_paths")
    (tmp_path / "model.gguf").touch()
    with pytest.raises(ValueError, match="does not name a file"):
        paths.resolve_dependency("model_path", tmp_path / "absent", directory=tmp_path, environ={})


def test_multiple_models_require_a_selection(tmp_path):
    paths = load("dependency_paths")
    for name in ["one.gguf", "two.gguf"]:
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="Multiple files"):
        paths.resolve_dependency("model_path", directory=tmp_path, environ={})


def test_environment_overrides_discovery(tmp_path):
    paths = load("dependency_paths")
    chosen = tmp_path / "selected"
    chosen.touch()
    (tmp_path / "llama-server").touch()
    assert paths.resolve_dependency("binary", directory=tmp_path,
                                    environ={"TN_LLAMA_BINARY": str(chosen)}) == chosen.resolve()


def test_repo_dependency_discovery(tmp_path):
    paths = load("dependency_paths")
    for name, (_, _, pattern) in paths.SETTINGS.items():
        filename = "model.gguf" if name == "model_path" else pattern
        target = tmp_path / filename
        target.touch()
        assert paths.resolve_dependency(name, directory=tmp_path, environ={}) == target.resolve()
