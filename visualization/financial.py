from __future__ import annotations

from typing import Iterable

import numpy as np

import plotly.graph_objects as go


def _discount(
    arr: np.ndarray, r: np.ndarray, start_at_year1: bool = True
) -> np.ndarray:
    """
    Discount each iteration's series using its own rate r (as fraction).
    arr: shape (I, T)
    r: shape (I,)
    """
    I, T = arr.shape
    periods = np.arange(1, T + 1) if start_at_year1 else np.arange(T)
    disc = (1.0 + r)[:, None] ** periods[None, :]
    return arr / disc


def _agg(a: np.ndarray, how: str) -> float | np.ndarray:
    how = how.lower()
    if how in ("p50", "median"):
        return np.nanpercentile(a, 50, axis=0)
    if how in ("p10", "p90"):
        q = int(how[1:])
        return np.nanpercentile(a, q, axis=0)
    if how in ("mean", "avg"):
        return np.nanmean(a, axis=0)
    raise ValueError(f"Unknown aggregation: {how}")


def npv_decomposition_waterfall(project, how: str = "p50") -> go.Figure:
    """
    Waterfall of PV components at the selected statistic across iterations.
    Components: Initial Investment (t0), Revenue PV, Opex PV, Debt Service PV, Net PV.
    """
    # Required tensors (shape discipline: (I, T) except init (I,1))
    rev = project.revenue.data
    opex = project.opex.data  # expected negative
    debt = project.debt_service.data  # costs, non-negative already
    init = np.abs(project.capex.data[:, 0])  # (I,)

    r = project.discount_rate.data[:, 0] / 100.0

    rev_pv = np.sum(_discount(rev, r), axis=1)  # (I,)
    opex_pv = np.sum(_discount(opex, r), axis=1)  # (I,) negative
    debt_pv = np.sum(_discount(-debt, r), axis=1)  # (I,) negative (as outflow)

    total_pv = -init + rev_pv + opex_pv + debt_pv  # (I,)

    vals = {
        "Initial Investment": -init,  # negative bar
        "Revenue (PV)": rev_pv,
        "Opex (PV)": opex_pv,  # negative
        "Debt Service (PV)": debt_pv,  # negative
        "Net PV": total_pv,
    }

    agg_vals = {k: float(_agg(v, how)) for k, v in vals.items()}

    measure = ["absolute", "relative", "relative", "relative", "total"]
    fig = go.Figure(
        go.Waterfall(
            name="NPV Decomposition",
            orientation="v",
            measure=measure,
            x=list(agg_vals.keys()),
            text=[f"{agg_vals[k]:,.0f}" for k in agg_vals.keys()],
            y=list(agg_vals.values()),
        )
    )
    fig.update_layout(
        title=f"NPV Decomposition ({how.upper()})",
        showlegend=False,
        yaxis_title="USD (PV)",
    )
    return fig


def payback_cdf(project, discounted: bool = True, which: str = "project") -> go.Figure:
    """
    CDF of payback (years). Uses discounted payback if available/selected.
    which: 'project' or 'equity' (uses corresponding property if available).
    """
    key = (
        (
            "discounted_payback_equity"
            if which == "equity"
            else "discounted_payback_period"
        )
        if discounted
        else "payback_period"
    )
    if not hasattr(project, key):
        raise AttributeError(f"{key} not found on project")
    arr = getattr(project, key).data[:, 0]  # (I,)

    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="CDF"))
    fig.update_layout(
        title=f"{'Discounted ' if discounted else ''}{which.capitalize()} Payback CDF",
        xaxis_title="Years",
        yaxis_title="Cumulative probability",
        yaxis=dict(ticksuffix=""),
    )
    return fig


def irr_histogram(project, which: str = "project", bins: int = 40) -> go.Figure:
    """
    Histogram of IRR (%). which: 'project' or 'equity'.
    """
    key = "project_irr" if which == "project" else "equity_irr"
    if not hasattr(project, key):
        raise AttributeError(f"{key} not found on project")
    irr = getattr(project, key).data[:, 0]  # (I,) in %

    fig = go.Figure(go.Histogram(x=irr, nbinsx=bins))
    fig.update_layout(
        title=f"{which.capitalize()} IRR Distribution",
        xaxis_title="IRR (%)",
        yaxis_title="Count",
    )
    return fig


def dscr_heatmap(project, percentiles: Iterable[int] = (10, 50, 90)) -> go.Figure:
    """
    Heatmap of DSCR percentiles by year. Red line at DSCR=1.0 is implied reference (annot).
    """
    if not hasattr(project, "dscr"):
        raise AttributeError("dscr not found on project")
    dscr = project.dscr.data  # (I, T)
    percs = sorted(int(p) for p in percentiles)
    Z = np.stack([np.nanpercentile(dscr, p, axis=0) for p in percs], axis=0)  # (P, T)

    years = np.arange(1, dscr.shape[1] + 1)
    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=years,
            y=[f"P{p}" for p in percs],
            coloraxis="coloraxis",
        )
    )
    fig.update_layout(
        title="DSCR Percentile Heatmap",
        xaxis_title="Year",
        yaxis_title="Percentile",
        coloraxis=dict(colorscale="RdYlGn"),
    )
    return fig
