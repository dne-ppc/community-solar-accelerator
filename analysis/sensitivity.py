
# analysis/sensitivity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from contextlib import contextmanager

import numpy as np


from models.core.types import Input,Tensor


# -------------------- util: temporary input overrides --------------------

@contextmanager
def patched_inputs(model: Any, **overrides: Any):
    """
    Temporarily override .data of one or more Input tensors on `model`.
    Values may be scalars or arrays broadcastable to (iterations, years).
    """
    saved: Dict[str, np.ndarray] = {}
    try:
        for name, val in overrides.items():
            obj = getattr(model, name)
            saved[name] = np.array(obj.data, copy=True)
            I, T = int(obj.iterations), int(obj.years)
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0:
                obj.data[:] = float(arr)
            else:
                if arr.shape == (I, T):
                    obj.data[:] = arr
                elif arr.shape == (I, 1):
                    obj.data[:] = np.tile(arr, (1, T))
                elif arr.shape == (1, T):
                    obj.data[:] = np.tile(arr, (I, 1))
                elif arr.shape == (T,):
                    obj.data[:] = np.tile(arr[None, :], (I, 1))
                elif arr.shape == (I,):
                    obj.data[:] = np.tile(arr[:, None], (1, T))
                else:
                    raise ValueError(f"Override for {name!r} has incompatible shape {arr.shape}, expected {(I,T)}")
        yield
    finally:
        for name, data in saved.items():
            getattr(model, name).data = data


# -------------------- metric extraction ----------------------------------

def _as_1d_metric(arr: np.ndarray, *, year: Union[str, int, None]) -> np.ndarray:
    """Reduce a tensor surface to (iterations,) using the chosen year selector.
    Supports: None/'final' (last year), integer (1-based), 'min'/'max'/'mean' across years.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if year in (None, "final", -1):
            return a[:, -1]
        if year == "min":
            return np.min(a, axis=1)
        if year == "max":
            return np.max(a, axis=1)
        if year == "mean":
            return np.mean(a, axis=1)
        if isinstance(year, int):
            # accept 1-based year index for user-friendliness
            idx = year - 1 if year > 0 else year
            return a[:, idx]
        raise ValueError(f"Unsupported year selector: {year!r}")
    raise ValueError(f"Metric array must be 1D/2D; got shape {a.shape}")


def metric_array(
    model: Any,
    target: str,
    *,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> np.ndarray:
    """
    Resolve a target metric on the model to a 1-D (iterations,) array.
    Rules:
      - If target refers to a ModelTensor-like object (has .data), reduce by `year`.
      - If it refers to a dict (e.g., model.npv), you must pass `key` to pick one.
      - If it refers to a callable (e.g., model.npv_total()), it must return (I,) or (I,T).
    """
    obj = getattr(model, target)
    # dict-style (e.g., npv: Dict[str, np.ndarray])
    if isinstance(obj, dict):
        if key is None:
            raise ValueError(f"target {target!r} is a dict; provide key=...")
        return _as_1d_metric(np.asarray(obj[key]), year=year)

    # callable (method)
    if callable(obj):
        out = obj()
        return _as_1d_metric(np.asarray(out), year=year)

    # tensor-like
    if hasattr(obj, "data"):
        return _as_1d_metric(np.asarray(obj.data), year=year)

    raise AttributeError(f"Cannot resolve target {target!r} on model")


def metric_percentile(
    model: Any,
    target: str,
    *,
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> float:
    """Percentile of the chosen metric."""
    arr = metric_array(model, target, key=key, year=year)
    return float(np.percentile(arr, p))


# -------------------- one-way tornado ------------------------------------

@dataclass(frozen=True)
class TornadoRow:
    name: str
    low: float
    high: float
    delta: float


def _bounds_for_input(inp: Input, mode: str = "p10_p90") -> Tuple[float, float]:
    """
    Return scalar (low, high) representative values for an input.
    Modes: 'p10_p90', 'min_max', 'p25_p75', or custom like 'p5_p95'.
    """
    mode = mode.lower()
    def pick(attr: str, default: Optional[float]) -> float:
        v = getattr(inp, attr, None)
        return float(v if v is not None else (default if default is not None else 0.0))

    if mode == "p10_p90":
        return pick("p10", None), pick("p90", None)
    if mode == "min_max":
        return pick("min_value", None), pick("max_value", None)
    if mode == "p25_p75":
        return pick("p25", pick("p10", None)), pick("p75", pick("p90", None))
    if "_" in mode and mode.startswith("p"):
        # e.g., p5_p95
        a, b = mode.split("_")
        return pick(a, None), pick(b, None)
    raise ValueError(f"Unknown mode {mode!r}")


def one_way_tornado(
    model: Any,
    target: str,
    *,
    inputs: Optional[Iterable[str]] = None,
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    bounds_mode: str = "p10_p90",
) -> List[TornadoRow]:
    """
    Compute one-way tornado rows for a target metric percentile.
    For each input, evaluate metric at low/high bounds and report delta (high-low).
    """
    if inputs is None:
        # only inputs that aren't fixed
        inputs = [n for n in getattr(model, "input_names") if not getattr(model, n).use_fixed]

    rows: List[TornadoRow] = []
    for name in inputs:
        inp = getattr(model, name)
        low_v, high_v = _bounds_for_input(inp, mode=bounds_mode)

        with patched_inputs(model, **{name: low_v}):
            low = metric_percentile(model, target, p=p, key=key, year=year)
        with patched_inputs(model, **{name: high_v}):
            high = metric_percentile(model, target, p=p, key=key, year=year)

        rows.append(TornadoRow(name=name, low=low, high=high, delta=high - low))

    # Sort by absolute impact descending
    rows.sort(key=lambda r: abs(r.delta), reverse=True)
    return rows


# -------------------- two-way grid ---------------------------------------

@dataclass(frozen=True)
class GridResult:
    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    Z: np.ndarray  # shape (len(y_values), len(x_values)) with metric at percentile p


def two_way_grid(
    model: Any,
    target: str,
    *,
    x_input: str,
    y_input: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> GridResult:
    """Compute a 2D grid of percentile(target) over two input sweeps."""
    xs = np.asarray(list(x_values), dtype=float)
    ys = np.asarray(list(y_values), dtype=float)
    Z = np.zeros((ys.size, xs.size), dtype=float)
    for j, xv in enumerate(xs):
        for i, yv in enumerate(ys):
            with patched_inputs(model, **{x_input: xv, y_input: yv}):
                Z[i, j] = metric_percentile(model, target, p=p, key=key, year=year)
    return GridResult(x_name=x_input, y_name=y_input, x_values=xs, y_values=ys, Z=Z)


# -------------------- correlation-based global sensitivity ---------------

@dataclass(frozen=True)
class CorrelationRow:
    name: str
    pearson_r: float
    spearman_r: float


def _flatten_input_value(inp: Input) -> np.ndarray:
    """Reduce an input surface to 1D per-iteration values; use year-1 column by default."""
    arr = np.asarray(inp.data, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0]
    raise ValueError("Input data must be 1D or 2D")


def correlation_sensitivity(
    model: Any,
    target: str,
    *,
    inputs: Optional[Iterable[str]] = None,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> List[CorrelationRow]:
    """
    Estimate global importance via Pearson/Spearman correlation between sampled
    input values and the per-iteration target metric.
    """
    try:
        from scipy.stats import spearmanr, pearsonr  # type: ignore
    except Exception:
        spearmanr = pearsonr = None  # type: ignore

    if inputs is None:
        inputs = [n for n in getattr(model, "input_names")]

    y = metric_array(model, target, key=key, year=year)
    rows: List[CorrelationRow] = []
    for name in inputs:
        x = _flatten_input_value(getattr(model, name))
        if x.shape != y.shape:
            # simple broadcast if x constant across time
            if x.ndim == 1 and y.ndim == 1 and x.size == 1:
                x = np.full_like(y, float(x[0]))
            elif x.ndim == 1 and y.ndim == 1:
                # ok
                pass
            else:
                raise ValueError(f"Shape mismatch for correlation: {name} has {x.shape}, metric has {y.shape}")
        if pearsonr is None:
            # fallback quick estimates
            pr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
            # crude Spearman via argsort ranks
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            sr = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0
        else:
            pr = float(pearsonr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
            sr = float(spearmanr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else 0.0  # type: ignore
        rows.append(CorrelationRow(name=name, pearson_r=pr, spearman_r=sr))

    # Sort by |pearson| descending as default
    rows.sort(key=lambda r: abs(r.pearson_r), reverse=True)
    return rows


# -------------------- elasticity -----------------------------------------

@dataclass(frozen=True)
class ElasticityRow:
    name: str
    base_value: float
    bump_value: float
    metric_base: float
    metric_bump: float
    elasticity: float


def elasticity(
    model: Any,
    target: str,
    *,
    input_name: str,
    bump: float,
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> ElasticityRow:
    """
    Log-log elasticity of percentile(target) w.r.t. an input around its current fixed value (or p50).
      E = (ΔM/M) / (ΔX/X), using finite-difference with base at current (or p50) and bump at given value.
    """
    inp = getattr(model, input_name)
    # base X
    base_x = float(getattr(inp, "fixed_value", None) if getattr(inp, "use_fixed", False) else getattr(inp, "p50", 0.0))

    with patched_inputs(model, **{input_name: base_x}):
        m0 = metric_percentile(model, target, p=p, key=key, year=year)
    with patched_inputs(model, **{input_name: bump}):
        m1 = metric_percentile(model, target, p=p, key=key, year=year)

    # Avoid div-by-zero; use small epsilon
    eps = 1e-12
    E = ((m1 - m0) / max(abs(m0), eps)) / ((bump - base_x) / max(abs(base_x), eps))
    return ElasticityRow(
        name=input_name, base_value=base_x, bump_value=float(bump), metric_base=float(m0), metric_bump=float(m1), elasticity=float(E)
    )


# -------------------- scenario sweep -------------------------------------

@dataclass(frozen=True)
class ScenarioPoint:
    label: str
    value: float


def scenario_sweep(
    model: Any,
    target: str,
    *,
    scenarios: Mapping[str, Mapping[str, Any]],
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
) -> List[ScenarioPoint]:
    """
    Evaluate percentile(target) across a dictionary of scenarios.
    Each item is {label: {input_name: value_or_array, ...}}.
    """
    out: List[ScenarioPoint] = []
    for label, overrides in scenarios.items():
        with patched_inputs(model, **overrides):
            val = metric_percentile(model, target, p=p, key=key, year=year)
        out.append(ScenarioPoint(label=label, value=float(val)))
    return out

# -------------------- tornado serializers ---------------------------------

def tornado_rows_to_dict(
    rows: Iterable[TornadoRow],
    *,
    include_rank: bool = True,
    sort: bool = True,
) -> list[dict]:
    """
    Convert tornado rows to a list of dicts suitable for JSON serialization.
    """
    rows_list = list(rows)
    if sort:
        rows_list = sorted(rows_list, key=lambda r: abs(r.delta), reverse=True)
    out: list[dict] = []
    for i, r in enumerate(rows_list, start=1):
        item = {
            "name": r.name,
            "low": float(r.low),
            "high": float(r.high),
            "delta": float(r.delta),
            "abs_delta": float(abs(r.delta)),
        }
        if include_rank:
            item["rank"] = i
        out.append(item)
    return out


def tornado_dataframe(rows: Iterable[TornadoRow], *, sort: bool = True):
    """
    Convert tornado rows to a pandas.DataFrame with useful columns:
    ['name', 'low', 'high', 'delta', 'abs_delta', 'rank']
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for tornado_dataframe") from e

    data = [dict(name=r.name, low=r.low, high=r.high, delta=r.delta) for r in rows]
    if not data:
        return pd.DataFrame(columns=["name", "low", "high", "delta", "abs_delta", "rank"])
    df = pd.DataFrame(data)
    df["abs_delta"] = df["delta"].abs()
    if sort:
        df = df.sort_values("abs_delta", ascending=False, kind="mergesort", ignore_index=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def tornado_df(
    model: any,
    target: str,
    *,
    inputs: Optional[Iterable[str]] = None,
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    bounds_mode: str = "p10_p90",
    sort: bool = True,
):
    """
    Convenience wrapper: computes one_way_tornado and returns a pandas.DataFrame.
    """
    rows = one_way_tornado(
        model, target, inputs=inputs, p=p, key=key, year=year, bounds_mode=bounds_mode
    )
    return tornado_dataframe(rows, sort=sort)


def tornado_dict(
    model: any,
    target: str,
    *,
    inputs: Optional[Iterable[str]] = None,
    p: float = 50.0,
    key: Optional[str] = None,
    year: Union[str, int, None] = "final",
    bounds_mode: str = "p10_p90",
    include_rank: bool = True,
    sort: bool = True,
) -> list[dict]:
    """
    Convenience wrapper: computes one_way_tornado and returns a JSON-serializable list of dicts.
    """
    rows = one_way_tornado(
        model, target, inputs=inputs, p=p, key=key, year=year, bounds_mode=bounds_mode
    )
    return tornado_rows_to_dict(rows, include_rank=include_rank, sort=sort)