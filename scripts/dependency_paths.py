"""Resolve explicitly configured dependencies without machine-specific defaults."""
import os
from pathlib import Path
import shutil

SETTINGS = {
    "tn_wheel": ("TN_WHEEL", "--tn-wheel", "tn_proto-2026.9.10b2.dev2-cp39-abi3-win_amd64.whl"),
    "opa": ("TN_POLARS_OPA", "--opa", "opa.exe"),
    "binary": ("TN_LLAMA_BINARY", "--binary", "llama-server"),
    "model_path": ("TN_LLAMA_MODEL", "--model-path", "*.gguf"),
}


def resolve_dependency(name, explicit=None, *, directory, environ=None):
    environment = os.environ if environ is None else environ
    variable, option, pattern = SETTINGS[name]
    configured = explicit if explicit is not None else environment.get(variable)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{option}/{variable} does not name a file: {path}")
        return path
    candidates = sorted(p.resolve() for p in Path(directory).glob(pattern) if p.is_file())
    if len(candidates) > 1:
        raise ValueError(f"Multiple files match {pattern}; select one with {option} or {variable}")
    if candidates:
        return candidates[0]
    if name == "binary":
        built = Path(directory) / "tn-llama" / "llama-server"
        if built.is_file():
            return built.resolve()
    if name == "opa":
        found = shutil.which("opa", path=environment.get("PATH", ""))
        if found:
            return Path(found).resolve()
    raise ValueError(f"Missing {name}. Place {pattern} in {directory}, set {variable}, or pass {option}.")
