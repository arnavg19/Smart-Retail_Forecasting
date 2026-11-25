from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.optimize import minimize_scalar


def _finite_number(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def demand_curve(
    predicted_demand: float,
    reference_price: float,
    price: float,
    elasticity: float,
) -> float:
    """Estimate demand using a constant-elasticity price response."""
    predicted_demand = _finite_number("predicted_demand", predicted_demand)
    reference_price = _finite_number("reference_price", reference_price)
    price = _finite_number("price", price)
    elasticity = _finite_number("elasticity", elasticity)
    if predicted_demand < 0:
        raise ValueError("predicted_demand must be non-negative.")
    if price <= 0 or reference_price <= 0:
        raise ValueError("price and reference_price must be greater than zero.")
    if abs(elasticity) > 10:
        raise ValueError("Absolute elasticity must not exceed 10.")
    with np.errstate(over="ignore", invalid="ignore"):
        demand = predicted_demand * np.power(reference_price / price, abs(elasticity))
    if not np.isfinite(demand):
        raise ValueError("Demand curve produced a non-finite value; check price and elasticity.")
    return float(demand)


def optimize_price(
    predicted_demand: float,
    unit_cost: float,
    inventory: float,
    historical_max_price: float,
    reference_price: float,
    elasticity: float,
    *,
    min_margin_floor: float = 0.15,
    max_price_delta_pct: float | None = None,
) -> Dict[str, float]:
    """Maximize expected profit within cost, history, and optional policy bounds."""
    predicted_demand = _finite_number("predicted_demand", predicted_demand)
    unit_cost = _finite_number("unit_cost", unit_cost)
    inventory = _finite_number("inventory", inventory)
    historical_max_price = _finite_number("historical_max_price", historical_max_price)
    reference_price = _finite_number("reference_price", reference_price)
    min_margin_floor = _finite_number("min_margin_floor", min_margin_floor)
    if predicted_demand < 0:
        raise ValueError("predicted_demand must be non-negative.")
    if unit_cost <= 0:
        raise ValueError("unit_cost must be greater than zero.")
    if inventory < 0:
        raise ValueError("inventory must be non-negative.")
    if historical_max_price <= 0:
        raise ValueError("historical_max_price must be greater than zero.")
    if reference_price <= 0:
        raise ValueError("reference_price must be greater than zero.")
    if not 0 <= min_margin_floor < 1:
        raise ValueError("min_margin_floor must be in the range [0, 1).")
    if max_price_delta_pct is not None:
        max_price_delta_pct = _finite_number("max_price_delta_pct", max_price_delta_pct)
        if not 0 <= max_price_delta_pct < 1:
            raise ValueError("max_price_delta_pct must be in the range [0, 1).")

    lower_bound = unit_cost * (1.0 + min_margin_floor)
    upper_bound = historical_max_price
    if max_price_delta_pct is not None:
        lower_bound = max(lower_bound, reference_price * (1.0 - max_price_delta_pct))
        upper_bound = min(upper_bound, reference_price * (1.0 + max_price_delta_pct))
    if lower_bound > upper_bound:
        raise ValueError(
            "No feasible price: the cost margin floor exceeds historical and policy price ceilings."
        )

    def profit(price: float) -> float:
        demand = min(
            demand_curve(predicted_demand, reference_price, price, elasticity),
            inventory,
        )
        return (price - unit_cost) * demand

    candidates = [lower_bound, upper_bound]
    if lower_bound < upper_bound:
        result = minimize_scalar(
            lambda price: -profit(float(price)),
            bounds=(lower_bound, upper_bound),
            method="bounded",
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError(f"Price optimization failed: {result.message}")
        candidates.append(float(result.x))
    optimal_price = max(candidates, key=profit)
    expected_demand = min(
        demand_curve(predicted_demand, reference_price, optimal_price, elasticity),
        inventory,
    )
    return {
        "optimal_price": float(optimal_price),
        "expected_demand": float(expected_demand),
        "expected_profit": float(profit(optimal_price)),
    }


__all__ = ["demand_curve", "optimize_price"]
