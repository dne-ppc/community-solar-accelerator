"""
Merged FinancialModel (compat with sensitivity.py)

- Uses explicit .data numpy arrays for all tensors.
- Imports from models.core.* if available; falls back to legacy models.*.
- Exposes:
    - operating_cashflow, project_cashflow, debt_service
    - equity_cashflow, equity_irr, project_irr, project_mirr
    - npv (Dict[str, np.ndarray]), npv_total()
    - payback_period (undiscounted), discounted_payback_project/equity
    - profitability_index, roi, dscr
- Compatibility aliases:
    - operating_margin := revenue - opex
    - retained_earnings := project_cashflow
    - capex := initial_investment (shape (I,1))
- Sensitivity-ready:
    - Inputs have .data with shape (iterations, years) and appear in model.input_names
    - Methods return (I,) or (I,T) arrays; Calculation/Output carry .data, .years, .iterations
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Any, TypeVar

import numpy as np
from pydantic import BaseModel, computed_field


from models.types import Input, Calculation, Output  # type: ignore
from models.graph import TensorGraph
from models.utils import depends_on, RoleRegistry, register_role

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")


# ----------------- helpers -----------------


def _npv(
    cashflows: np.ndarray, rates: np.ndarray, *, start_at_year1: bool = True
) -> np.ndarray:
    I, T = cashflows.shape
    r = rates
    if r.ndim == 0:
        r = np.full(I, r, dtype=float)
    if r.shape != (I,):
        raise ValueError(f"rates must be scalar or (iterations,), got {r.shape}")
    periods = np.arange(1, T + 1) if start_at_year1 else np.arange(T)
    denom = (1.0 + r[:, None]) ** periods[None, :]
    return np.sum(cashflows / denom, axis=1)


def _has_sign_change(cf: np.ndarray) -> bool:
    s = np.sign(cf)
    nz = s[s != 0]
    return nz.size > 1 and np.any(nz[:-1] != nz[1:])


def _irr_rows(cash: np.ndarray) -> np.ndarray:
    try:
        import numpy_financial as npf  # type: ignore

        vals = np.full(cash.shape[0], np.nan, dtype=float)
        for i in range(cash.shape[0]):
            row = cash[i]
            if _has_sign_change(row):
                try:
                    vals[i] = float(npf.irr(row))
                except Exception:
                    vals[i] = np.nan
        return vals
    except Exception:
        # Bisection fallback
        def npv_at(r, x):
            t = np.arange(x.size)
            return np.sum(x / (1 + r) ** t)

        lo, hi = -0.999, 10.0
        vals = np.full(cash.shape[0], np.nan, dtype=float)
        for i, row in enumerate(cash):
            if not _has_sign_change(row):
                continue
            a, b = lo, hi
            fa, fb = npv_at(a, row), npv_at(b, row)
            if np.sign(fa) == np.sign(fb):
                continue
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = npv_at(m, row)
                if abs(fm) < 1e-9:
                    break
                if np.sign(fa) * np.sign(fm) < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            vals[i] = m
        return vals


def _mirr_rows(
    cash: np.ndarray, finance_rate: np.ndarray, reinvest_rate: np.ndarray
) -> np.ndarray:
    I, T1 = cash.shape
    T = T1 - 1
    fr = finance_rate if finance_rate.ndim else np.full(I, finance_rate, float)
    rr = reinvest_rate if reinvest_rate.ndim else np.full(I, reinvest_rate, float)
    out = np.full(I, np.nan, float)
    for i in range(I):
        cf = cash[i]
        neg = cf.copy()
        neg[neg > 0] = 0.0
        pos = cf.copy()
        pos[pos < 0] = 0.0
        pv_neg = 0.0
        for t, v in enumerate(neg):
            pv_neg += v / ((1 + fr[i]) ** t)
        fv_pos = 0.0
        for t, v in enumerate(pos):
            fv_pos += v * ((1 + rr[i]) ** (T - t))
        if pv_neg == 0:
            out[i] = np.nan
        else:
            out[i] = (fv_pos / -pv_neg) ** (1.0 / T) - 1.0
    return out


# ----------------- Merged FinancialModel -----------------


class FinancialModel(TensorGraph, BaseModel):
    """
    Base finance with explicit members (no hasattr), sensitivity-ready.
    Required Inputs (declare on subclass or provide at init):
      - discount_rate (%), inflation_rate (%), start_year_proportion (0..1)
      - initial_investment ($ at t=0, ≥0)

    Subclass must implement these computed members or provide as Inputs:
      - revenue (Output, $/yr, length T)
      - opex (Output, $/yr, ≤0, length T)
      - finance_costs (Calculation, $/yr, length T)  [or declare as Input]

    Conventions:
      - Outflows at t>=1 are negative in per-year cashflow surfaces.
      - initial_investment stored as positive number (cash out occurs as -initial_investment at t=0).
    """

    # Core dimensions (TensorGraph generally supplies pydantic config)
    scenario: str = "Base"
    years: int = 25
    iterations: int = 1000

    # Canonical rate Inputs (percent)
    discount_rate: Input
    inflation_rate: Input
    start_year_proportion: Input

    # Investment (shape (I,1) or (I,T) where col0 used)
    capex: Input

    def __init__(
        self,
        scenario: str = "Base",
        years: int = 25,
        iterations: int = 1000,
        config_path: str = "inputs/Base.yaml",
        **data: Any,
    ) -> None:
        for field_name in self.input_names:
            if field_name not in data:
                print(f"Loading Input {field_name} from config {config_path}")
                data[field_name] = Input.from_config(
                    scenario,
                    field_name,
                    years=years,
                    iterations=iterations,
                    config_path=config_path,
                )
        data["scenario"] = scenario
        data["years"] = years
        data["iterations"] = iterations
        super().__init__(**data)

    # ----------------- abstract/computed basics -----------------
    @computed_field
    @property
    def input_names(self) -> list[str]:
        # rely on TensorGraph scanning if available; otherwise compute using hasattr check
        try:
            return super().input_names  # type: ignore[attr-defined]
        except Exception:
            names = []
            for n in dir(self):
                obj = getattr(self, n)
                if isinstance(obj, Input):
                    names.append(n)
            return names

    # ----------------- Schedules & convenience -----------------
    @computed_field
    @property
    def years_array(self) -> NdArray:
        return np.arange(self.years - 1)

    # ---- Compatibility & clarity helpers ----
    @computed_field
    @property
    def operating_margin(self) -> Calculation:
        """Compatibility: revenue - opex (same shape as revenue/opex)."""
        data = self.revenue.data + self.opex.data
        return Calculation(
            scenario=self.scenario,
            label="Operating Margin",
            units="$",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # @computed_field
    # @property
    # def capex(self) -> Calculation:
    #     """Compatibility: alias to initial_investment as a (I,1) Calculation."""
    #     arr = np.atleast_2d(self.initial_investment.data)
    #     if arr.shape[1] > 1:
    #         arr = arr[:, :1]
    #     return Calculation(self.scenario, "Capex", "$", 1, self.iterations, np.abs(arr))

    # ----------------- Operating cashflow -----------------
    @computed_field
    @property
    def operating_cashflow(self) -> Output:
        data = self.revenue.data + self.opex.data  # opex expected negative
        # Scale first year by start_year_proportion (clipped to [0,1])
        data[:, 0] *= np.clip(self.start_year_proportion.data[:, 0], 0.0, 1.0)
        return Output(
            scenario=self.scenario,
            label="Operating Cashflow",
            units="$",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # ----------------- Debt service & project cashflow -----------------
    @computed_field
    @property
    def debt_service(self) -> Calculation:
        # non-negative denominator for DSCR
        arr = np.abs(self.finance_costs.data)
        return Calculation(
            scenario=self.scenario,
            label="Debt Service",
            units="$",
            years=self.years,
            iterations=self.iterations,
            data=arr,
        )

    @computed_field
    @property
    def project_cashflow(self) -> Calculation:
        data = self.operating_cashflow.data - self.debt_service.data
        return Calculation(
            scenario=self.scenario,
            label="Project Cashflow",
            units="$",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # Compatibility: retained_earnings == project_cashflow
    @computed_field
    @property
    def retained_earnings(self) -> Output:
        return Output(
            scenario=self.scenario,
            label="Retained Earnings ($)",
            units="$",
            years=self.years,
            iterations=self.iterations,
            data=self.project_cashflow.data.copy(),
        )

    # ----------------- Equity cashflow & IRR -----------------
    @computed_field
    @property
    @depends_on("project_cashflow")
    def equity_cashflow(self) -> Output:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1), dtype=float)
        # t=0 negative outflow
        cash[:, 0] = -np.abs(self.capex.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        return Output(
            scenario=self.scenario,
            label="Equity Cashflow",
            units="$",
            years=T + 1,
            iterations=I,
            data=cash,
        )

    @computed_field
    @property
    @depends_on("equity_cashflow")
    def equity_irr(self) -> Calculation:
        irr = _irr_rows(self.equity_cashflow.data) * 100.0
        return Calculation(
            scenario=self.scenario,
            label="Equity IRR (%)",
            units="%",
            years=1,
            iterations=self.iterations,
            data=irr.reshape(self.iterations, 1),
        )

    # ----------------- NPV & Profitability Index -----------------
    @computed_field
    @property
    def npv(self) -> Dict[str, NdArray]:
        r = self.discount_rate.data[:, 0] / 100.0
        out: Dict[str, np.ndarray] = {}
        # include all $ Outputs with exactly T columns (skip T+1 like equity_cashflow)
        for name in getattr(
            self, "output_names", []
        ):  # TensorGraph usually provides output_names
            obj = getattr(self, name)
            try:
                is_money = (obj.units or "").find("$") >= 0
                if is_money and obj.years == self.years:
                    out[name] = _npv(obj.data, r, start_at_year1=True)
            except Exception:
                continue
        # Always include Project Cashflow NPV as a key for convenience
        out["project_cashflow"] = _npv(
            self.project_cashflow.data, r, start_at_year1=True
        )
        return out

    @computed_field
    def npv_total(self) -> NdArray:
        total = np.zeros((self.iterations,), dtype=float)
        for _, vals in self.npv.items():
            total += np.asarray(vals, dtype=float)
        return total

    @computed_field
    @property
    def profitability_index(self) -> Calculation:
        I, T = self.iterations, self.years
        cf = np.zeros((I, T + 1), dtype=float)
        cf[:, 0] = -np.abs(self.capex.data[:, 0])
        cf[:, 1:] = self.project_cashflow.data
        r = self.discount_rate.data[:, 0] / 100.0
        periods = np.arange(T + 1)
        disc = (1 + r)[:, None] ** periods[None, :]
        pv = cf / disc
        pv_in = np.sum(np.where(pv > 0, pv, 0.0), axis=1)
        pv_out = np.sum(np.where(pv < 0, -pv, 0.0), axis=1)
        pi = np.where(pv_out > 0, pv_in / pv_out, np.nan)
        return Calculation(
            scenario=self.scenario,
            label="Profitability Index",
            units="unitless",
            years=1,
            iterations=I,
            data=pi.reshape(I, 1),
        )

    # ----------------- Payback (undiscounted & discounted) -----------------
    @computed_field
    @property
    def payback_period(self) -> Calculation:
        I, T = self.iterations, self.years
        cf = np.zeros((I, T + 1), dtype=float)
        cf[:, 0] = -np.abs(self.capex.data[:, 0])
        cf[:, 1:] = self.project_cashflow.data
        cum = np.cumsum(cf, axis=1)
        res = np.full(I, 9999.0, dtype=float)
        for i in range(I):
            idx = np.where(cum[i] >= 0)[0]
            if idx.size:
                res[i] = float(idx[0])
        return Calculation(
            scenario=self.scenario,
            label="Payback Period",
            units="year",
            years=1,
            iterations=I,
            data=res.reshape(I, 1),
        )

    @computed_field
    @property
    def discounted_payback_period(self) -> Calculation:
        I, T = self.iterations, self.years
        r = self.discount_rate.data[:, 0] / 100.0
        cf = np.zeros((I, T + 1), dtype=float)
        cf[:, 0] = -np.abs(self.capex.data[:, 0])
        cf[:, 1:] = self.project_cashflow.data
        res = np.full(I, 9999.0, float)
        for i in range(I):
            disc = cf[i] / (1 + r[i]) ** np.arange(T + 1)
            cum = np.cumsum(disc)
            idx = np.where(cum >= 0)[0]
            if idx.size == 0:
                continue
            k = idx[0]
            if k == 0:
                res[i] = 0.0
            else:
                prev = cum[k - 1]
                inc = disc[k]
                frac = 0.0 if inc == 0 else (-prev) / inc
                res[i] = (k - 1) + max(0.0, min(1.0, frac))
        return Calculation(
            scenario=self.scenario,
            label="Discounted Payback (Project)",
            units="year",
            years=1,
            iterations=I,
            data=res.reshape(I, 1),
        )

    @computed_field
    @property
    def discounted_payback_equity(self) -> Calculation:
        I, T1 = self.iterations, self.equity_cashflow.years
        r = self.discount_rate.data[:, 0] / 100.0
        res = np.full(I, 9999.0, float)
        for i in range(I):
            disc = self.equity_cashflow.data[i] / (1 + r[i]) ** np.arange(T1)
            cum = np.cumsum(disc)
            idx = np.where(cum >= 0)[0]
            if idx.size == 0:
                continue
            k = idx[0]
            if k == 0:
                res[i] = 0.0
            else:
                prev = cum[k - 1]
                inc = disc[k]
                frac = 0.0 if inc == 0 else (-prev) / inc
                res[i] = (k - 1) + max(0.0, min(1.0, frac))
        return Calculation(
            scenario=self.scenario,
            label="Discounted Payback (Equity)",
            units="year",
            years=1,
            iterations=I,
            data=res.reshape(I, 1),
        )

    # ----------------- ROI, IRR, MIRR, DSCR -----------------
    @computed_field
    @property
    def roi(self) -> Calculation:
        I = self.iterations
        init = np.abs(self.capex.data[:, 0])
        gain = np.sum(self.project_cashflow.data, axis=1) - init
        roi = np.where(init > 0, gain / init, np.nan)
        return Calculation(
            scenario=self.scenario,
            label="ROI",
            units="unitless",
            years=1,
            iterations=I,
            data=roi.reshape(I, 1),
        )

    @computed_field
    @property
    def project_irr(self) -> Calculation:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1), dtype=float)
        cash[:, 0] = -np.abs(self.capex.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        irr = _irr_rows(cash) * 100.0
        return Calculation(
            scenario=self.scenario,
            label="Project IRR (%)",
            units="%",
            years=1,
            iterations=I,
            data=irr.reshape(I, 1),
        )

    @computed_field
    @property
    def project_mirr(self) -> Calculation:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1), dtype=float)
        cash[:, 0] = -np.abs(self.capex.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        r = self.discount_rate.data[:, 0] / 100.0
        mirr = _mirr_rows(cash, r, r) * 100.0
        return Calculation(
            scenario=self.scenario,
            label="Project MIRR (%)",
            units="%",
            years=1,
            iterations=I,
            data=mirr.reshape(I, 1),
        )

    @computed_field
    @property
    def dscr(self) -> Calculation:
        eps = 1e-12
        ocf = np.maximum(self.operating_cashflow.data, 0.0)
        ds = self.debt_service.data
        denom = np.where(ds > eps, ds, np.nan)
        ratio = ocf / denom
        ratio = np.where(np.isnan(denom), np.inf, ratio)
        return Calculation(
            scenario=self.scenario,
            label="DSCR",
            units="unitless",
            years=self.years,
            iterations=self.iterations,
            data=ratio,
        )
