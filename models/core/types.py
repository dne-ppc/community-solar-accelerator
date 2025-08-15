# models/core/types.py
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable, Dict, Any
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, field_validator
import yaml

from models.core.registry import register_role
from models.core.tensor import TensorOps
from metalog import metalog


@runtime_checkable
class Sampler(Protocol):
    """Strategy for generating sample arrays for ModelInput.

    Returns an array of shape (iterations, years).
    """

    def __call__(
        self,
        *,
        iterations: int,
        years: int,
        p10: float,
        p50: float,
        p90: float,
        boundedness: str = "u",
    ) -> np.ndarray: ...


class Tensor(TensorOps, BaseModel):
    """Base tensor-like type with metadata.

    - Pure numpy `data` with shape (iterations, years) unless documented otherwise.
    - No plotting, no file I/O, no UI imports.
    """

    scenario: str = Field(default="Base")
    label: str = Field(default="")
    units: str = Field(default="")
    years: int = Field(default=1, gt=0)
    iterations: int = Field(default=1, gt=0)
    data: np.ndarray = Field(default_factory=lambda: np.zeros((1, 1), dtype=float))

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    @field_validator("data")
    @classmethod
    def _ensure_2d(cls, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            v = v.reshape((-1, 1))
        if v.ndim != 2:
            raise ValueError("data must be 2D: (iterations, years)")
        return v

    def with_data(
        self,
        data: np.ndarray,
        *,
        label: Optional[str] = None,
        units: Optional[str] = None,
    ):
        """Convenience: return a copy with new data/metadata (preserves dims and scenario)."""
        return type(self)(
            scenario=self.scenario,
            label=label if label is not None else self.label,
            units=units if units is not None else self.units,
            years=self.years,
            iterations=self.iterations,
            data=np.asarray(data, dtype=float),
        )

    # ----- IO ---------------------------------------------------
    @classmethod
    def from_config(cls, scenario, name, config_path: str = "inputs/Base.yaml", **data):
        with open(config_path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        settings: Dict[str, Any] = cfg.get(name)
        settings["scenario"] = scenario
        if "years" in settings:
            data.pop("years", None)
        settings.update(data)
        return cls(**settings)

    # ---- Finance-friendly helper (still pure math) -------------------------
    def npv(self, rates: np.ndarray, *, start_at_year1: bool = True) -> np.ndarray:
        """Per-iteration NPV over time axis.

        Args:
            rates: shape (iterations,) or scalar discount rate in decimal (e.g., 0.07 for 7%).
            start_at_year1: if True, discount periods are [1..years]; else [0..years-1].

        Returns:
            np.ndarray with shape (iterations,), NPV per iteration.
        """
        cashflows = self.data  # (I, T)
        I, T = cashflows.shape

        r = np.asarray(rates, dtype=float)
        if r.ndim == 0:
            r = np.full(I, r)
        if r.shape[0] != I:
            raise ValueError(
                f"rates must broadcast to (iterations,) – got {r.shape}, expected {(I,)}"
            )

        periods = np.arange(1, T + 1) if start_at_year1 else np.arange(T)
        # (I, T) / (I, 1) ** (T,) -> broadcast to (I, T)
        denom = (1.0 + r[:, None]) ** periods[None, :]
        return np.sum(cashflows / denom, axis=1)


@register_role("input")
class Input(Tensor):
    """A stochastic or fixed input to a model.

    Use `update_data()` with a Sampler strategy to fill `data` for
    (iterations, years). If `use_fixed` is True, the sampler is ignored.
    """

    # Fixed vs stochastic
    use_fixed: bool = False
    fixed_value: Optional[float] = None

    # Optional hard bounds
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Tri-quantile summary (distribution spec; interpretation belongs to sampler)
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0

    # Distribution bounds style for samplers like metalog ("u","sl","su","b")
    boundedness: str = "u"

    # Optional UI-adjacent metadata (kept here because calculations may rely on units/step)
    step: float = 0.01

    def update_data(self, sampler: Optional[Sampler] = None) -> None:
        """Populate `data` using either a fixed value or Metalog sampling.

        Priority:
        1) If use_fixed=True and fixed_value is set → full constant array.
        2) Else use `sampler` if provided.
        3) Else **fit Metalog** to (p10,p50,p90) with boundedness/bounds and sample.
        """
        shape = (int(self.iterations), int(self.years))

        # 1) Fixed path overrides everything
        if bool(self.use_fixed) and (self.fixed_value is not None):
            self.data = np.full(shape, float(self.fixed_value), dtype=float)
            return

        lo = (
            float(self.min_value)
            if getattr(self, "min_value", None) is not None
            else None
        )
        hi = (
            float(self.max_value)
            if getattr(self, "max_value", None) is not None
            else None
        )
        st = float(getattr(self, "step", 0.0) or 0.0)
        bounds = None
        if lo is not None and hi is not None:
            bounds = [lo - st, hi + st]

        dist = metalog.fit(
            x=[float(self.p10), float(self.p50), float(self.p90)],
            boundedness=str(self.boundedness or "u"),
            bounds=bounds,
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )
        arr = metalog.r(dist, n=int(self.iterations) * int(self.years))
        arr = np.asarray(arr, dtype=float).reshape(shape)

        # Clip to bounds if provided
        lo = getattr(self, "min_value", None)
        hi = getattr(self, "max_value", None)
        if lo is not None:
            arr = np.maximum(arr, float(lo))
        if hi is not None:
            arr = np.minimum(arr, float(hi))

        self.data = np.asarray(arr, dtype=float)


@register_role("calc")
class Calculation(Tensor):
    """Intermediate derived quantity (not directly a cashflow)."""

    pass


@register_role("output")
class Output(Tensor):
    """Final model outputs; mark cashflows by role elsewhere if needed."""

    pass
