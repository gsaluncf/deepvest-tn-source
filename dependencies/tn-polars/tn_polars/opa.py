import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import wasmtime

from .release import Decision


class OpaModuleCache:
    """Bounded compiled-code reuse for one dataframe preparation.

    Callers verify current signed artifacts before each lookup. The cache holds
    no policy data, evaluator state or decisions. Each evaluator creates a fresh
    Wasmtime Store and Instance, even when it uses an existing compiled Module.
    """

    def __init__(self, *, max_entries=8, max_source_bytes=4 * 1024 * 1024):
        if max_entries < 1 or max_source_bytes < 1:
            raise ValueError("compiled-module cache limits must be positive")
        self.engine = wasmtime.Engine()
        self._modules = OrderedDict()
        self._source_bytes = 0
        self._max_entries = max_entries
        self._max_source_bytes = max_source_bytes

    def module_for(self, policy_revision, wasm_bytes):
        if not isinstance(policy_revision, str) or not policy_revision:
            raise ValueError("compiled-module reuse requires an exact policy revision")
        raw = bytes(wasm_bytes)
        key = (policy_revision, raw)
        if key in self._modules:
            self._modules.move_to_end(key)
            return self._modules[key]
        module = wasmtime.Module(self.engine, raw)
        if len(raw) <= self._max_source_bytes:
            while (len(self._modules) >= self._max_entries
                   or self._source_bytes + len(raw) > self._max_source_bytes):
                removed_key, _ = self._modules.popitem(last=False)
                self._source_bytes -= len(removed_key[1])
            self._modules[key] = module
            self._source_bytes += len(raw)
        return module


class OpaWasmEvaluator:
    """Minimal in-process implementation of OPA's documented Wasm ABI."""

    def __init__(self, wasm_path, data_path):
        self._initialize(Path(wasm_path).read_bytes(), Path(data_path).read_bytes())

    @classmethod
    def from_bytes(cls, wasm_bytes, data_bytes, *, module_cache=None, policy_revision=None):
        """Load the exact artifact bytes already verified by an edition resolver."""
        instance = cls.__new__(cls)
        instance._initialize(bytes(wasm_bytes), bytes(data_bytes),
                             module_cache=module_cache, policy_revision=policy_revision)
        return instance

    def _initialize(self, wasm_bytes, data_bytes, *, module_cache=None, policy_revision=None):
        if module_cache is None:
            self.store = wasmtime.Store()
            module = wasmtime.Module(self.store.engine, wasm_bytes)
        else:
            module = module_cache.module_for(policy_revision, wasm_bytes)
            self.store = wasmtime.Store(module_cache.engine)
        linker = wasmtime.Linker(self.store.engine)
        imported_memory = None
        for imported in module.imports:
            if isinstance(imported.type, wasmtime.FuncType):
                linker.define_func(imported.module, imported.name, imported.type,
                                   self._stub_for(imported.type, imported.name))
            elif isinstance(imported.type, wasmtime.MemoryType):
                imported_memory = wasmtime.Memory(self.store, imported.type)
                linker.define(self.store, imported.module, imported.name, imported_memory)
        self.instance = linker.instantiate(self.store, module)
        self.exports = self.instance.exports(self.store)
        self.memory = self.exports.get("memory") or imported_memory
        self._data_value = self._parse_json(data_bytes)

    @staticmethod
    def _stub_for(func_type, name):
        """Host imports the evaluator does not implement must fail loudly.

        A Rego builtin that OPA cannot compile natively arrives as a host
        import. Returning a placeholder from it would silently corrupt the
        decision, so every unimplemented import raises instead.
        """
        def stub(*_args):
            raise RuntimeError(
                f"OPA wasm called host import '{name}', which this evaluator "
                "does not implement. The compiled Rego uses a builtin that "
                "OPA did not compile natively into wasm; the decision cannot "
                "be trusted, so evaluation halts. Either restrict the policy "
                "to natively compiled builtins or extend OpaWasmEvaluator.")
        return stub

    def _parse_json(self, raw):
        address = self.exports["opa_malloc"](self.store, len(raw))
        self.memory.write(self.store, raw, address)
        return self.exports["opa_json_parse"](self.store, address, len(raw))

    def _read_c_string(self, address):
        raw = bytes(self.memory.read(self.store, address, self.memory.data_len(self.store)))
        return raw.split(b"\0", 1)[0].decode("utf-8")

    def decide(self, policy_input):
        context = self.exports["opa_eval_ctx_new"](self.store)
        self.exports["opa_eval_ctx_set_data"](self.store, context, self._data_value)
        self.exports["opa_eval_ctx_set_input"](
            self.store, context, self._parse_json(json.dumps(policy_input).encode("utf-8")))
        self.exports["opa_eval_ctx_set_entrypoint"](self.store, context, 0)
        self.exports["eval"](self.store, context)
        result_ptr = self.exports["opa_eval_ctx_get_result"](self.store, context)
        dumped_ptr = self.exports["opa_json_dump"](self.store, result_ptr)
        raw_result = json.loads(self._read_c_string(dumped_ptr))
        result = raw_result[0]["result"] if isinstance(raw_result, list) else raw_result["result"]
        decision_id = hashlib.sha256(
            json.dumps(policy_input, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return Decision(allow=result["allow"], reason=result["reason"], decision_id=decision_id)
