
# analysis/solvers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from analysis.sensitivity import metric_percentile, patched_inputs


@dataclass(frozen=True)
class SolveResult:
    input_name: str
    value: float
    metric: str
    target_value: float
    achieved: float
    iterations: int
    converged: bool

def solve_input_for_metric(
    model: Any,
    input_name: str,
    target_metric: str,
    *,
    target_value: float,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    bounds: Tuple[float, float],
    p: float = 50.0,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> SolveResult:
    """
    Solve for a scalar input value such that percentile(target_metric) ~= target_value.
    Uses robust bisection on [lo, hi]. Raises if f(lo) and f(hi) are on the same side.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        raise ValueError("Invalid bounds")
    def f(x: float) -> float:
        with patched_inputs(model, **{input_name: x}):
            return metric_percentile(model, target_metric, p=p, key=key, year=year) - target_value

    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        raise ValueError("Metric evaluated to NaN at one of the bounds")
    if flo == 0.0:
        val = lo; achieved = target_value
        return SolveResult(input_name, val, target_metric, target_value, target_value, 0, True)
    if fhi == 0.0:
        val = hi; achieved = target_value
        return SolveResult(input_name, val, target_metric, target_value, target_value, 0, True)
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("Target not bracketed by bounds (monotonicity issue or wrong bounds)")

    it = 0
    a, b = lo, hi
    fa, fb = flo, fhi
    mid = (a + b) / 2.0
    while it < max_iter:
        it += 1
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) <= tol or abs(b - a) <= tol:
            break
        # keep the sign change interval
        if np.sign(fa) * np.sign(fm) < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    achieved = target_value + fm
    return SolveResult(input_name, mid, target_metric, target_value, achieved, it, abs(fm) <= tol)

# Convenience wrappers

def breakeven_input_for_zero(
    model: Any,
    input_name: str,
    target_metric: str,
    *,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    bounds: Tuple[float, float],
    p: float = 50.0,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> SolveResult:
    """Find input value such that percentile(target_metric) = 0 (e.g., NPV = 0)."""
    return solve_input_for_metric(
        model, input_name, target_metric,
        target_value=0.0, key=key, year=year, bounds=bounds, p=p, tol=tol, max_iter=max_iter
    )

def price_for_target_irr(
    model: Any,
    price_input_name: str,
    target_irr_percent: float,
    *,
    bounds: Tuple[float, float],
    p: float = 50.0,
) -> SolveResult:
    """Solve for price/tariff that achieves a target equity IRR% at P50."""
    return solve_input_for_metric(
        model, price_input_name, "equity_irr", target_value=target_irr_percent, bounds=bounds, p=p
    )

def capex_for_zero_npv(
    model: Any,
    *,
    bounds: Tuple[float, float],
    p: float = 50.0,
) -> SolveResult:
    """Solve for CAPEX (scalar) such that total NPV = 0 at P50."""
    return breakeven_input_for_zero(model, "capex", "npv_total", bounds=bounds, p=p)

def price_for_min_dscr(
    model: Any,
    price_input_name: str,
    min_dscr_target: float,
    *,
    bounds: Tuple[float, float],
    p: float = 50.0,
) -> SolveResult:
    """Solve for price/tariff such that P50(min DSCR across years) meets target."""
    return solve_input_for_metric(
        model, price_input_name, "dscr", target_value=min_dscr_target, year="min", bounds=bounds, p=p
    )
