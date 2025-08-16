from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go


from analysis.solvers import (
    SolveResult,
    solve_input_for_metric,
    breakeven_input_for_zero,
    price_for_target_irr,
    capex_for_zero_npv,
    price_for_min_dscr,
)


from analysis.sensitivity import metric_percentile, patched_inputs


def run_solver(
    model: Any,
    solver: str,
    *,
    input_name: Optional[str] = None,
    target_metric: Optional[str] = None,
    target_value: Optional[float] = None,
    key: Optional[str] = None,
    year: int | str | None = "final",
    bounds: Tuple[float, float] = (0.0, 1.0),
    p: float = 50.0,
    tol: float = 1e-6,
    max_iter: int = 60,
    min_dscr_target: Optional[float] = None,
    target_irr_percent: Optional[float] = None,
) -> "SolveResult":
    solver = solver.lower()
    if solver == "capex for npv=0":
        return capex_for_zero_npv(model, bounds=bounds, p=p)
    if solver == "price for target irr%":
        if input_name is None or target_irr_percent is None:
            raise ValueError("input_name and target_irr_percent required")
        return price_for_target_irr(
            model, input_name, target_irr_percent, bounds=bounds, p=p
        )
    if solver == "price for min dscr":
        if input_name is None or min_dscr_target is None:
            raise ValueError("input_name and min_dscr_target required")
        return price_for_min_dscr(
            model, input_name, min_dscr_target, bounds=bounds, p=p
        )
    if solver == "breakeven (metric=0)":
        if input_name is None or target_metric is None:
            raise ValueError("input_name and target_metric required")
        return breakeven_input_for_zero(
            model,
            input_name,
            target_metric,
            key=key,
            year=year,
            bounds=bounds,
            p=p,
            tol=tol,
            max_iter=max_iter,
        )
    if solver == "solve input for metric":
        if input_name is None or target_metric is None or target_value is None:
            raise ValueError("input_name, target_metric, target_value required")
        return solve_input_for_metric(
            model,
            input_name,
            target_metric,
            target_value=target_value,
            key=key,
            year=year,
            bounds=bounds,
            p=p,
            tol=tol,
            max_iter=max_iter,
        )
    raise ValueError(f"Unknown solver: {solver}")


def response_curve(
    model: Any,
    input_name: str,
    target_metric: str,
    *,
    key: Optional[str] = None,
    year: int | str | None = "final",
    bounds: Tuple[float, float],
    p: float = 50.0,
    num: int = 41,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(bounds[0]), float(bounds[1])
    xs = np.linspace(lo, hi, int(num))
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        with patched_inputs(model, **{input_name: float(x)}):
            ys[i] = metric_percentile(model, target_metric, p=p, key=key, year=year)
    return xs, ys


def result_df(res: "SolveResult") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_name": res.input_name,
                "value": res.value,
                "metric": res.metric,
                "target_value": res.target_value,
                "achieved": res.achieved,
                "iterations": res.iterations,
                "converged": res.converged,
            }
        ]
    )


def fig_response_curve(
    model: Any,
    input_name: str,
    target_metric: str,
    *,
    key: Optional[str] = None,
    year: int | str | None = "final",
    bounds: Tuple[float, float],
    p: float = 50.0,
    solver_result: "SolveResult | None" = None,
    target_value: float | None = None,
    num: int = 61,
    title: str | None = None,
) -> go.Figure:
    xs, ys = response_curve(
        model,
        input_name,
        target_metric,
        key=key,
        year=year,
        bounds=bounds,
        p=p,
        num=num,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{p:.0f}th percentile")
    )
    if target_value is not None:
        fig.add_hline(
            y=target_value,
            line_dash="dash",
            annotation_text="Target",
            annotation_position="top left",
        )
    if solver_result is not None and np.isfinite(solver_result.value):
        fig.add_vline(
            x=solver_result.value,
            line_dash="dot",
            annotation_text=f"Solution: {solver_result.value:.4g}",
        )
        fig.add_trace(
            go.Scatter(
                x=[solver_result.value],
                y=[solver_result.achieved],
                mode="markers",
                name="Solution",
                marker=dict(size=10, symbol="x"),
            )
        )
    fig.update_layout(
        title=title
        or f"Response curve • {target_metric}{'[' + key + ']' if key else ''} vs {input_name}",
        xaxis_title=input_name,
        yaxis_title=target_metric,
        showlegend=False,
    )
    return fig
