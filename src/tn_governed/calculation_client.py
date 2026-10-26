"""Typed adapters for the calculation engine."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class CalculationRequest:
    """An ordered wealth path for the declared drawdown calculation."""

    function: str
    _wealth_path: tuple[float, ...]

    def __init__(self, function: str, arguments: dict):
        if function != "max_drawdown":
            raise ValueError("this calculation profile requires max_drawdown")
        if not isinstance(arguments, dict) or set(arguments) != {"wealth_path"}:
            raise ValueError("max_drawdown requires exactly wealth_path")
        path = arguments["wealth_path"]
        if not isinstance(path, list) or len(path) < 2 or any(
            isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value <= 0 for value in path
        ):
            raise ValueError("wealth_path requires at least two finite positive observations")
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "_wealth_path", tuple(float(value) for value in path))

    @property
    def arguments(self) -> dict:
        return {"wealth_path": list(self._wealth_path)}


@dataclass(frozen=True)
class CalculationResult:
    function: str
    function_version: str
    method: str
    unit: str
    value: float
    calculation_id: str


def _result(request: CalculationRequest, reply: dict) -> CalculationResult:
    expected = {"function", "function_version", "method", "unit", "value", "status"}
    if not isinstance(reply, dict) or set(reply) != expected:
        raise ValueError("Calculation returned an unexpected result schema")
    if (
        reply["function"] != request.function
        or reply["function_version"] != "calculation-engine/1.0.0"
        or reply["method"] != "peak_to_trough_running_max"
        or reply["unit"] != "decimal"
        or reply["status"] != "complete"
        or isinstance(reply["value"], bool)
        or not isinstance(reply["value"], (int, float))
        or not math.isfinite(reply["value"])
        or not -1 <= reply["value"] <= 0
    ):
        raise ValueError("Calculation returned a result outside the declared calculation contract")
    # This identifies the calculation record. The surrounding TN object signs it.
    invocation = {"request": {"function": request.function, "arguments": request.arguments},
                  "result": reply}
    record = json.dumps(invocation, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return CalculationResult(
        function=request.function, function_version=reply["function_version"],
        method=reply["method"], unit=reply["unit"], value=float(reply["value"]),
        calculation_id="sha256:" + hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )


class EngineCalculationPort:
    """Call the pinned engine through its closed function registry."""

    def __init__(self, engine=None):
        if engine is None:
            from .calculation_engine import Engine

            engine = Engine()
        self.engine = engine

    def calculate(self, request: CalculationRequest) -> CalculationResult:
        return _result(request, self.engine.call(request.function, request.arguments))


class HttpCalculationPort:
    """Call the same engine's /calculate endpoint."""

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/calculate"
        self.timeout = timeout

    def calculate(self, request: CalculationRequest) -> CalculationResult:
        body = json.dumps(
            {"function": request.function, "args": request.arguments}, allow_nan=False,
        ).encode("utf-8")
        call = urllib.request.Request(
            self.url, data=body, method="POST", headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(call, timeout=self.timeout) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Calculation reply exceeds the configured response limit")
        return _result(request, json.loads(raw))
