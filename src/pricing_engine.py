from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import minimize_scalar


def demand_curve(predicted_demand: float, reference_price: float, price: float, elasticity: float) -> float:
    """Log-log demand model: quantity scales as price^elasticity."""
    if price <= 0:
        raise ValueError("Price must be positive.")
    if reference_price <= 0:
        raise ValueError("Reference price must be positive.")
    elasticity = float(elasticity)
    return float(max(predicted_demand * (reference_price / price) ** abs(elasticity), 0.0))


def optimize_price(
    predicted_demand: float,
    unit_cost: float,
    inventory: float,
    historical_max_price: float,
    reference_price: float,
    elasticity: float,
) -> Dict[str, float]:
    """Solve the profit-maximizing price subject to guardrails."""
    if unit_cost <= 0:
        raise ValueError("Unit cost must be greater than zero.")
    if predicted_demand <= 0:
        predicted_demand = 1.0
    if inventory <= 0:
        inventory = 1.0
    if historical_max_price <= unit_cost:
        historical_max_price = unit_cost * 2.0

    lower_bound = max(unit_cost * 1.15, 1e-6)
    upper_bound = min(historical_max_price * 1.20, max(reference_price * 2.0, unit_cost * 10.0))
    if upper_bound <= lower_bound:
        upper_bound = lower_bound * 1.5

    def objective(price: float) -> float:
        demand = min(demand_curve(predicted_demand, reference_price, price, elasticity), inventory)
        profit = (price - unit_cost) * demand
        return -profit

    res = minimize_scalar(objective, bounds=(lower_bound, upper_bound), method="bounded")
    optimal_price = float(res.x)
    expected_demand = min(demand_curve(predicted_demand, reference_price, optimal_price, elasticity), inventory)
    expected_profit = float((optimal_price - unit_cost) * expected_demand)

    return {
        "optimal_price": float(max(lower_bound, min(optimal_price, upper_bound))),
        "expected_demand": float(expected_demand),
        "expected_profit": float(max(expected_profit, 0.0)),
    }


__all__ = ["demand_curve", "optimize_price"]
