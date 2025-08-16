
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go



from analysis.sensitivity import metric_array 

from  analysis import risk


def var_cvar(
    model: Any,
    target: str,
    *,
    key: Optional[str] = None,
    year: int | str | None = "final",
    alpha: float = 0.05,
    side: str = "lower",  # 'lower' or 'upper'
    title: Optional[str] = None,
) -> go.Figure:
    res = risk.var_cvar(model, target, alpha=alpha, side=side, key=key, year=year)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR", x=["VaR"], y=[res.var]))
    fig.add_trace(go.Bar(name="CVaR (ES)", x=["CVaR"], y=[res.cvar]))
    fig.update_layout(
        barmode="group",
        title=title or f"VaR / CVaR • {target}{'[' + key + ']' if key else ''} • α={alpha:.2f} • {side} tail",
        yaxis_title=target,
        showlegend=True,
    )
    return fig

def distribution(
    model: Any,
    target: str,
    *,
    key: Optional[str] = None,
    year: int | str | None = "final",
    bins: int = 40,
    show_mean: bool = True,
    show_median: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    x = np.asarray(metric_array(model, target, key=key, year=year), float).ravel()
    s = risk.distribution_summary(model, target, key=key, year=year)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=bins, name="Distribution"))
    shapes = []
    annotations = []
    if show_mean:
        shapes.append(dict(type="line", x0=s.mean, x1=s.mean, y0=0, y1=1, yref="paper", line=dict(dash="dash")))
        annotations.append(dict(x=s.mean, y=1.02, yref="paper", text="mean", showarrow=False))
    if show_median:
        shapes.append(dict(type="line", x0=s.p50, x1=s.p50, y0=0, y1=1, yref="paper", line=dict(dash="dot")))
        annotations.append(dict(x=s.p50, y=1.02, yref="paper", text="median", showarrow=False))
    fig.update_layout(
        title=title or f"Distribution • {target}{'[' + key + ']' if key else ''}",
        xaxis_title=target,
        yaxis_title="Count",
        shapes=shapes,
        annotations=annotations,
    )
    return fig

def npv_profile(
    model: Any,
    *,
    cashflow: str = "project_cashflow",
    rates: Sequence[float] | None = None,
    title: Optional[str] = None,
) -> go.Figure:
    rs, med = risk.npv_profile(model, cashflow=cashflow, rates=rates)
    fig = go.Figure(go.Scatter(x=[r * 100 for r in rs], y=med, mode="lines+markers", name="Median NPV"))
    fig.update_layout(
        title=title or f"NPV Profile • {cashflow}",
        xaxis_title="Discount rate (%)",
        yaxis_title="NPV",
        showlegend=False,
    )
    return fig

def drawdown_histogram(model: Any, bins: int = 40, title: Optional[str] = None) -> go.Figure:
    """
    Histogram of max drawdown across iterations (equity cashflow cumulative).
    Values are ≤ 0; more negative = deeper drawdowns.
    """
    eq = getattr(model, "equity_cashflow").data  # (I, T+1)
    I, T1 = eq.shape
    dd = np.zeros(I, float)
    for i in range(I):
        c = np.cumsum(eq[i])
        peak = np.maximum.accumulate(c)
        drawdown = c - peak
        dd[i] = float(drawdown.min())
    fig = go.Figure(go.Histogram(x=dd, nbinsx=bins))
    fig.update_layout(
        title=title or "Equity Max Drawdown Distribution",
        xaxis_title="Max drawdown (currency)",
        yaxis_title="Count",
    )
    return fig

# ----------------------------- TABLES -----------------------------

def summary_table(
    model: Any,
    items: Sequence[Tuple[str, Optional[str], int | str | None]],
) -> pd.DataFrame:
    """
    Build a summary table over multiple targets/keys.
    items: sequence of (target, key, year) triples.
    """
    rows: List[Dict[str, Any]] = []
    for target, key, year in items:
        s = risk.distribution_summary(model, target, key=key, year=year)
        rows.append({
            "target": target if key is None else f"{target}[{key}]",
            "year": year,
            "mean": s.mean,
            "std": s.std,
            "p5": s.p5,
            "p25": s.p25,
            "p50": s.p50,
            "p75": s.p75,
            "p95": s.p95,
            "skew": s.skew,
            "kurtosis": s.kurtosis,
        })
    df = pd.DataFrame(rows)
    return df

def tail_risk_table(
    model: Any,
    target: str,
    *,
    key: Optional[str] = None,
    year: int | str | None = "final",
    alphas: Sequence[float] = (0.01, 0.05, 0.10),
    side: str = "lower",
) -> pd.DataFrame:
    rows = []
    for a in alphas:
        res = risk.var_cvar(model, target, key=key, year=year, alpha=a, side=side)
        rows.append({"alpha": a, "var": res.var, "cvar": res.cvar})
    return pd.DataFrame(rows)

# Convenience presets for typical financial models
DEFAULT_RISK_ITEMS = (
    ("npv_total", None, "final"),
    ("project_irr", None, "final"),
    ("equity_irr", None, "final"),
    ("dscr", None, "final"),
)
