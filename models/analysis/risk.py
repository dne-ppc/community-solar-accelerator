# analysis/risk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurtosis  # type: ignore


from analysis.sensitivity import metric_array


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    skew: float
    kurtosis: float  # excess kurtosis


def _skew(arr: np.ndarray) -> float:
    a = np.asarray(arr, float)
    if _scipy_skew is not None:
        return float(_scipy_skew(a, bias=False))
    # fallback: Fisher-Pearson
    m = a.mean()
    s = a.std(ddof=1)
    if s == 0:
        return 0.0
    n = a.size
    g1 = ((a - m) ** 3).sum() / (n * (s**3))
    return float(g1)


def _kurtosis(arr: np.ndarray) -> float:
    a = np.asarray(arr, float)
    if _scipy_kurtosis is not None:
        return float(_scipy_kurtosis(a, fisher=True, bias=False))
    # fallback: excess kurtosis
    m = a.mean()
    s2 = a.var(ddof=1)
    if s2 == 0:
        return -3.0
    n = a.size
    g2 = ((a - m) ** 4).sum() / (n * (s2**2)) - 3.0
    return float(g2)


def distribution_summary(
    model: Any,
    target: str,
    *,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> MetricSummary:
    x = metric_array(model, target, key=key, year=year)
    p = np.percentile(x, percentiles)
    return MetricSummary(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        p5=float(p[0]),
        p25=float(p[1]),
        p50=float(p[2]),
        p75=float(p[3]),
        p95=float(p[4]),
        skew=float(_skew(x)),
        kurtosis=float(_kurtosis(x)),
    )


@dataclass(frozen=True)
class VaRResult:
    var: float
    cvar: float  # expected shortfall


def var_cvar(
    model: Any,
    target: str,
    *,
    alpha: float = 0.05,
    side: str = "lower",  # 'lower' → left tail loss; 'upper' → right tail
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> VaRResult:
    x = np.asarray(metric_array(model, target, key=key, year=year), float)
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if side == "lower":
        v = float(np.percentile(x, 100 * alpha))
        tail = x[x <= v]
        c = float(np.mean(tail)) if tail.size else v
    else:
        v = float(np.percentile(x, 100 * (1 - alpha)))
        tail = x[x >= v]
        c = float(np.mean(tail)) if tail.size else v
    return VaRResult(var=v, cvar=c)


def npv_profile(
    model: Any,
    *,
    cashflow: str = "project_cashflow",  # or 'equity_cashflow'
    rates: Sequence[float] = tuple(np.linspace(0.0, 0.15, 31)),  # decimals
    year_start_at_one: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NPV vs discount rate, median across iterations.
    Returns (rates, median_npv) with rates in decimals.
    """
    cf = getattr(model, cashflow)
    if not hasattr(cf, "data"):
        raise AttributeError(f"{cashflow} is not a tensor with .data")
    arr = np.asarray(cf.data, float)  # (I, T) or (I, T+1) if equity
    if arr.ndim != 2:
        raise ValueError("cashflow tensor must be 2D")
    I, T = arr.shape
    if cashflow == "equity_cashflow":
        # expects t=0..T, we will discount from t=0
        t = np.arange(T)  # already includes t=0
    else:
        # project cashflow starts at year 1
        t = np.arange(1, T + 1) if year_start_at_one else np.arange(T)
    rs = np.asarray(list(rates), float)
    med = np.zeros_like(rs, dtype=float)
    for k, r in enumerate(rs):
        disc = (1 + r) ** t[None, :]
        if cashflow == "equity_cashflow":
            npv = (arr / disc).sum(axis=1)
        else:
            npv = (arr / disc).sum(axis=1)
        med[k] = float(np.percentile(npv, 50))
    return rs, med


def equity_drawdown_stats(model: Any) -> dict:
    """
    Compute max drawdown statistics on cumulative equity cashflows per iteration.
    Returns {'max_drawdown': median across I, 'p95_drawdown': p95 across I}
    """
    eq = getattr(model, "equity_cashflow").data  # (I, T+1)
    I, T1 = eq.shape
    dd = np.zeros(I, float)
    for i in range(I):
        c = np.cumsum(eq[i])
        peak = np.maximum.accumulate(c)
        drawdown = c - peak  # ≤ 0
        dd[i] = float(drawdown.min())
    return {
        "max_drawdown_med": float(np.percentile(dd, 50)),
        "max_drawdown_p95": float(np.percentile(dd, 95)),
    }
