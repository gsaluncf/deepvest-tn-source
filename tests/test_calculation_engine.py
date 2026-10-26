"""Calculation results and the exact governed invocation record."""

import math

import pytest

from tn_governed.calculation_client import CalculationRequest, EngineCalculationPort


def request():
    return CalculationRequest("max_drawdown", {"wealth_path": [100.0, 120.0, 90.0, 110.0]})


def test_actual_engine_computes_peak_to_trough_drawdown():
    result = EngineCalculationPort().calculate(request())
    assert result.value == -0.25
    assert result.function == "max_drawdown"
    assert result.function_version == "calculation-engine/1.0.0"
    assert result.method == "peak_to_trough_running_max"
    assert result.unit == "decimal"
    assert result.calculation_id


@pytest.mark.parametrize("arguments", [
    {"wealth_path": []}, {"wealth_path": [100, 0]}, {"wealth_path": [True, 100]},
    {"wealth_path": [100, math.nan]}, {"wealth_path": [100, math.inf]},
    {"wealth_path": [100, 120], "undeclared": 1},
])
def test_request_rejects_invalid_wealth_paths(arguments):
    with pytest.raises(ValueError):
        CalculationRequest("max_drawdown", arguments)


def test_request_rejects_undeclared_function():
    with pytest.raises(ValueError):
        CalculationRequest("invented_metric", {"wealth_path": [100, 120]})


def test_engine_error_is_propagated_without_fallback():
    error = RuntimeError("calculation unavailable")

    class Engine:
        def call(self, function, args):
            raise error

    with pytest.raises(RuntimeError) as caught:
        EngineCalculationPort(Engine()).calculate(request())
    assert caught.value is error


@pytest.mark.parametrize("change", [
    {"value": math.nan}, {"value": math.inf}, {"function": "other"},
    {"status": "pending"}, {"method": "other"}, {"unit": "USD"},
    {"value": True}, {"function_version": ""}, {"extra": 12},
])
def test_engine_reply_requires_the_declared_complete_finite_result(change):
    class Engine:
        def call(self, function, args):
            return {
                "function": "max_drawdown", "function_version": "calculation-engine/1.0.0",
                "value": -0.25, "method": "peak_to_trough_running_max", "unit": "decimal",
                "status": "complete", **change,
            }

    with pytest.raises(ValueError):
        EngineCalculationPort(Engine()).calculate(request())


def test_request_arguments_are_a_detached_snapshot():
    arguments = {"wealth_path": [100.0, 120.0, 90.0, 110.0]}
    value = CalculationRequest("max_drawdown", arguments)
    arguments["wealth_path"][2] = 1
    value.arguments["wealth_path"][2] = 2
    assert value.arguments == {"wealth_path": [100.0, 120.0, 90.0, 110.0]}


def test_actual_engine_http_service_returns_the_same_calculation():
    import socket
    import threading
    import time

    import uvicorn

    from tn_governed.calculation_engine import build_app
    from tn_governed.calculation_client import HttpCalculationPort

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(build_app(), log_level="error"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        remote = HttpCalculationPort(f"http://127.0.0.1:{port}").calculate(request())
        assert remote == EngineCalculationPort().calculate(request())
    finally:
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
