"""DeepVest deterministic financial engine.

Ten approved functions over numpy, a closed argument schema for each, and an
optional FastAPI surface. The registry is the only calculation path the TN
runtime will admit.
"""
from __future__ import annotations

import numpy as np

ENGINE_VERSION = "deepvest-engine/1.0.0"
FREQ_SCALE = {"daily": 252, "weekly": 52, "monthly": 12, "annual": 1}


def annualized_volatility(returns, frequency, annualize):
    vol = float(np.std(np.asarray(returns, dtype=np.float64), ddof=1))
    return vol * float(np.sqrt(FREQ_SCALE[frequency])) if annualize else vol


def max_drawdown(wealth_path):
    w = np.asarray(wealth_path, dtype=np.float64)
    peak = np.maximum.accumulate(w)
    return float(np.min((w - peak) / peak))


def sharpe_ratio(returns, risk_free_rate, annualization_factor):
    excess = np.asarray(returns, dtype=np.float64) - risk_free_rate
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(annualization_factor))


def rolling_correlation(dates_a, returns_a, dates_b, returns_b, window):
    common = sorted(set(dates_a) & set(dates_b))
    a = dict(zip(dates_a, returns_a))
    b = dict(zip(dates_b, returns_b))
    xa = np.asarray([a[d] for d in common], dtype=np.float64)
    xb = np.asarray([b[d] for d in common], dtype=np.float64)
    return [float(np.corrcoef(xa[i - window:i], xb[i - window:i])[0, 1]) for i in range(window, len(xa) + 1)]


def terminal_portfolio_value(initial_value, weights, returns_matrix):
    r = np.asarray(returns_matrix, dtype=np.float64)
    growth = np.prod(1.0 + r, axis=0)
    return float(initial_value * np.dot(np.asarray(weights, dtype=np.float64), growth))


def annualized_portfolio_return(portfolio_returns, periods_per_year):
    r = np.asarray(portfolio_returns, dtype=np.float64)
    return float(np.prod(1.0 + r) ** (periods_per_year / r.size) - 1.0)


def portfolio_correlation(weights_a, weights_b, returns_matrix):
    r = np.asarray(returns_matrix, dtype=np.float64)
    ra = r @ np.asarray(weights_a, dtype=np.float64)
    rb = r @ np.asarray(weights_b, dtype=np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def rebalance_trades(holdings, prices, target_weights, cash):
    h = np.asarray(holdings, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    w = np.asarray(target_weights, dtype=np.float64)
    total = float(h @ p) + cash
    delta = (total * w - h * p) / p
    return [int(v) for v in np.trunc(delta)]


def min_variance_weights(covariance, lower, upper):
    s = np.asarray(covariance, dtype=np.float64)
    ones = np.ones(s.shape[0])
    x = np.linalg.solve(s, ones)
    w = x / x.sum()
    feasible = bool(np.all(w >= np.asarray(lower)) and np.all(w <= np.asarray(upper)))
    return {"weights": [float(v) for v in w], "feasible": feasible}


def constrained_allocation(expected_returns, lower, upper, sectors, sector_caps):
    mu = np.asarray(expected_returns, dtype=np.float64)
    w = np.asarray(lower, dtype=np.float64).copy()
    used = {}
    for i, sec in enumerate(sectors):
        used[sec] = used.get(sec, 0.0) + w[i]
    remaining = 1.0 - w.sum()
    for i in np.lexsort((np.arange(mu.size), -mu)):
        if remaining <= 1e-15:
            break
        room = min(upper[i] - w[i], sector_caps[sectors[i]] - used[sectors[i]], remaining)
        if room > 0:
            w[i] += room
            used[sectors[i]] += room
            remaining -= room
    return {"weights": [float(v) for v in w], "feasible": bool(abs(remaining) <= 1e-12)}


# name -> (callable, closed argument schema, unit, method label)
REGISTRY = {
    "annualized_volatility": (annualized_volatility, ("returns", "frequency", "annualize"), "decimal", "sample_std_sqrt_scale"),
    "max_drawdown": (max_drawdown, ("wealth_path",), "decimal", "peak_to_trough_running_max"),
    "sharpe_ratio": (sharpe_ratio, ("returns", "risk_free_rate", "annualization_factor"), "ratio", "mean_excess_over_sample_std_annualized"),
    "rolling_correlation": (rolling_correlation, ("dates_a", "returns_a", "dates_b", "returns_b", "window"), "vector", "pearson_inner_join_dates"),
    "terminal_portfolio_value": (terminal_portfolio_value, ("initial_value", "weights", "returns_matrix"), "currency", "buy_and_hold_drift"),
    "annualized_portfolio_return": (annualized_portfolio_return, ("portfolio_returns", "periods_per_year"), "decimal", "geometric_horizon_conversion"),
    "portfolio_correlation": (portfolio_correlation, ("weights_a", "weights_b", "returns_matrix"), "ratio", "pearson_constant_mix_series"),
    "rebalance_trades": (rebalance_trades, ("holdings", "prices", "target_weights", "cash"), "shares_vector", "whole_share_truncate_toward_zero"),
    "min_variance_weights": (min_variance_weights, ("covariance", "lower", "upper"), "weights_vector", "closed_form_fully_invested_with_bounds_check"),
    "constrained_allocation": (constrained_allocation, ("expected_returns", "lower", "upper", "sectors", "sector_caps"), "weights_vector", "lower_bounds_first_greedy_by_expected_return"),
}


ARG_TYPES = {
    "annualized_volatility": {"returns": "array_number", "frequency": "string", "annualize": "boolean"},
    "max_drawdown": {"wealth_path": "array_number"},
    "sharpe_ratio": {"returns": "array_number", "risk_free_rate": "number", "annualization_factor": "number"},
    "rolling_correlation": {"dates_a": "array", "returns_a": "array_number", "dates_b": "array", "returns_b": "array_number", "window": "integer"},
    "terminal_portfolio_value": {"initial_value": "number", "weights": "array_number", "returns_matrix": "matrix_number"},
    "annualized_portfolio_return": {"portfolio_returns": "array_number", "periods_per_year": "number"},
    "portfolio_correlation": {"weights_a": "array_number", "weights_b": "array_number", "returns_matrix": "matrix_number"},
    "rebalance_trades": {"holdings": "array_number", "prices": "array_number", "target_weights": "array_number", "cash": "number"},
    "min_variance_weights": {"covariance": "matrix_number", "lower": "array_number", "upper": "array_number"},
    "constrained_allocation": {"expected_returns": "array_number", "lower": "array_number", "upper": "array_number", "sectors": "array", "sector_caps": "object_number"},
}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _matches(value, spec) -> bool:
    if spec == "number":
        return _is_num(value)
    if spec == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec == "string":
        return isinstance(value, str)
    if spec == "boolean":
        return isinstance(value, bool)
    if spec == "array":
        return isinstance(value, list) and len(value) > 0
    if spec == "array_number":
        return isinstance(value, list) and len(value) > 0 and all(_is_num(v) for v in value)
    if spec == "matrix_number":
        return isinstance(value, list) and len(value) > 0 and all(isinstance(r, list) and r and all(_is_num(v) for v in r) for r in value)
    if spec == "object_number":
        return isinstance(value, dict) and value and all(_is_num(v) for v in value.values())
    return False


def coerce_args(function: str, args: dict):
    """Decode JSON-encoded strings and integral floats into the declared type where that is unambiguous.
    Returns (args, coerced_keys). A value that still does not match is left for validate_args to reject."""
    import json as _json
    spec = ARG_TYPES.get(function, {})
    out, coerced = {}, []
    for k, v in args.items():
        t = spec.get(k)
        if t and not _matches(v, t):
            if isinstance(v, str) and t != "string":
                try:
                    cand = _json.loads(v)
                    if _matches(cand, t):
                        out[k] = cand
                        coerced.append(k)
                        continue
                except (ValueError, TypeError):
                    pass
            if t == "integer" and isinstance(v, float) and v.is_integer():
                out[k] = int(v)
                coerced.append(k)
                continue
        out[k] = v
    return out, coerced


def validate_args(function: str, args: dict) -> None:
    spec = ARG_TYPES[function]
    for k, t in spec.items():
        if k in args and not _matches(args[k], t):
            raise ValueError(f"argument {k} must be {t}, got {type(args[k]).__name__}")


class Engine:
    """Local calculation service. Rejects undeclared functions, open argument sets, and mistyped arguments."""

    version = ENGINE_VERSION

    def call(self, function: str, args: dict) -> dict:
        if function not in REGISTRY:
            raise ValueError(f"undeclared function: {function}")
        fn, schema, unit, method = REGISTRY[function]
        if tuple(sorted(args)) != tuple(sorted(schema)):
            raise ValueError(f"argument set does not match closed schema for {function}")
        validate_args(function, args)
        value = fn(**args)
        return {"value": value, "unit": unit, "method": method, "function": function,
                "function_version": ENGINE_VERSION, "status": "complete"}


def build_app():
    """FastAPI surface for deployment. Importing fastapi is optional for the benchmark."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="DeepVest Financial Engine", version=ENGINE_VERSION)
    engine = Engine()

    class CalcRequest(BaseModel):
        function: str
        args: dict

    def calculate(req: CalcRequest):
        try:
            return engine.call(req.function, req.args)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e))

    # The request model is local to this optional factory. Resolve its annotation
    # before FastAPI inspects the endpoint under postponed annotations.
    calculate.__annotations__["req"] = CalcRequest
    app.post("/calculate")(calculate)
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)
