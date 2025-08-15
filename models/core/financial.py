# models/finance/financial.py  (refactor to remove hasattr)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, computed_field

from models.core.types import Input, Calculation, Output
from models.core.graph import TensorGraph, depends_on


# ----------------- helpers (unchanged) -----------------
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


# ----------------- FinancialModel -----------------


@dataclass(frozen=True)
class NPVRow:
    name: str
    p5: float
    p50: float
    p95: float


def _zeros(shape, dtype=float):
    return np.zeros(shape, dtype=dtype)


class FinancialModel(TensorGraph):
    """
    Base finance with explicit members (no hasattr):
      Required members (all provided with safe zero defaults):
        - revenue: $/yr (≥0)
        - opex:   $/yr (≤0)
        - finance_costs: $/yr (≥0) used as debt service for DSCR
        - initial_investment: $ at t=0 (≥0)
    Conventions:
      - Outflows at t>=1 are negative in per-year cashflow surfaces.
      - discount_rate/inflation_rate stored as percent inputs (e.g., 7 => 7%).
    """

    # Core dimensions
    scenario: str = Field(default="Base")
    years: int = Field(default=25, gt=0)
    iterations: int = Field(default=1000, gt=0)

    # Canonical rates (percent)
    discount_rate: Input
    inflation_rate: Input
    start_year_proportion: Input

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    # Explicit financial members (no hasattr fallbacks)
    @computed_field
    @property
    def revenue(self) -> Output:
        raise NotImplementedError("FinancialModel subclass must implement revenue")

    @computed_field
    @property
    def opex(self) -> Output:
        raise NotImplementedError("FinancialModel subclass must implement opex")

    @computed_field
    @property
    def finance_costs(self) -> Calculation:
        raise NotImplementedError(
            "FinancialModel subclass must implement finance_costs"
        )

    @computed_field
    @property
    def initial_investment(self) -> Calculation:
        raise NotImplementedError(
            "FinancialModel subclass must implement initial_investment"
        )

    # ------ Schedules --------------------------------------------------------
    @computed_field
    @property
    def escalation_factor(self) -> Calculation:
        I, T = self.iterations, self.years
        pi = self.inflation_rate.data[:, 0] / 100.0
        fac = np.ones((I, T), dtype=float)
        if T > 1:
            exponents = np.arange(1, T, dtype=float)
            fac[:, 1:] = (1.0 + pi)[:, None] ** exponents[None, :]
        return Calculation(self.scenario, "Escalation Factor", "unitless", T, I, fac)

    # ------ Operating cashflow ----------------------------------------------
    @computed_field
    @property
    def operating_cashflow(self) -> Output:
        data = self.revenue.data + self.opex.data
        data[:, 0] *= np.clip(self.start_year_proportion.data[:, 0], 0.0, 1.0)
        return Output(
            self.scenario, "Operating Cashflow", "$", self.years, self.iterations, data
        )

    # ------ Financing hooks & composed flows --------------------------------
    @computed_field
    @property
    def debt_service(self) -> Calculation:
        # Already declared explicitly; ensure non-negative for DSCR denominator
        arr = np.abs(self.finance_costs.data)
        return Calculation(
            self.scenario, "Debt Service", "$", self.years, self.iterations, arr
        )

    @computed_field
    @property
    def project_cashflow(self) -> Calculation:
        return Calculation(
            self.scenario,
            "Project Cashflow",
            "$",
            self.years,
            self.iterations,
            self.operating_cashflow.data - self.debt_service.data,
        )

    # ------ Equity cashflow & metrics ---------------------------------------
    @computed_field
    @property
    @depends_on("project_cashflow")
    def equity_cashflow(self) -> Output:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1))
        # Convention: initial_investment >= 0 (cash out), t=0 flow is negative
        cash[:, 0] = -np.abs(self.initial_investment.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        return Output(self.scenario, "Equity Cashflow", "$", T + 1, I, cash)

    @computed_field
    @property
    @depends_on("equity_cashflow")
    def equity_irr(self) -> Calculation:
        irr = _irr_rows(self.equity_cashflow.data) * 100.0
        return Calculation(
            self.scenario,
            "Equity IRR (%)",
            "%",
            1,
            self.iterations,
            irr.reshape(self.iterations, 1),
        )

    # ------ NPV & PI ---------------------------------------------------------
    @computed_field
    @property
    def npv(self) -> Dict[str, np.ndarray]:
        """NPV for all $ Outputs with exactly T years (skips T+1 like equity_cashflow)."""
        r = self.discount_rate.data[:, 0] / 100.0
        out: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            obj = getattr(self, name)
            if (
                isinstance(obj, Output)
                and (obj.units or "").find("$") >= 0
                and obj.years == self.years
            ):
                out[name] = _npv(obj.data, r, start_at_year1=True)
        return out

    def npv_total(self) -> np.ndarray:
        total = np.zeros((self.iterations,))
        for _, vals in self.npv.items():
            total += vals
        return total

    @computed_field
    @property
    def profitability_index(self) -> Calculation:
        I, T = self.iterations, self.years
        cf = np.zeros((I, T + 1))
        cf[:, 0] = -np.abs(self.initial_investment.data[:, 0])
        cf[:, 1:] = self.project_cashflow.data
        r = self.discount_rate.data[:, 0] / 100.0
        periods = np.arange(T + 1)
        disc = (1 + r)[:, None] ** periods[None, :]
        pv = cf / disc
        pv_in = np.sum(np.where(pv > 0, pv, 0.0), axis=1)
        pv_out = np.sum(np.where(pv < 0, -pv, 0.0), axis=1)
        pi = np.where(pv_out > 0, pv_in / pv_out, np.nan)
        return Calculation(
            self.scenario, "Profitability Index", "unitless", 1, I, pi.reshape(I, 1)
        )

    # ------ Payback (undiscounted & discounted) ------------------------------
    @computed_field
    @property
    def payback_period(self) -> Calculation:
        I, T = self.iterations, self.years
        cf = np.zeros((I, T + 1))
        cf[:, 0] = -np.abs(self.initial_investment.data[:, 0])
        cf[:, 1:] = self.project_cashflow.data
        cum = np.cumsum(cf, axis=1)
        res = np.full(I, 9999.0, dtype=float)
        for i in range(I):
            idx = np.where(cum[i] >= 0)[0]
            if idx.size:
                res[i] = float(idx[0])
        return Calculation(
            self.scenario, "Payback Period", "year", 1, I, res.reshape(I, 1)
        )

    @computed_field
    @property
    def discounted_payback_project(self) -> Calculation:
        I, T = self.iterations, self.years
        r = self.discount_rate.data[:, 0] / 100.0
        cf = np.zeros((I, T + 1))
        cf[:, 0] = -np.abs(self.initial_investment.data[:, 0])
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
            self.scenario,
            "Discounted Payback (Project)",
            "year",
            1,
            I,
            res.reshape(I, 1),
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
            self.scenario,
            "Discounted Payback (Equity)",
            "year",
            1,
            I,
            res.reshape(I, 1),
        )

    # ------ ROI --------------------------------------------------------------
    @computed_field
    @property
    def roi(self) -> Calculation:
        I = self.iterations
        init = np.abs(self.initial_investment.data[:, 0])
        gain = np.sum(self.project_cashflow.data, axis=1) - init
        roi = np.where(init > 0, gain / init, np.nan)
        return Calculation(self.scenario, "ROI", "unitless", 1, I, roi.reshape(I, 1))

    # ------ IRR & MIRR -------------------------------------------------------
    @computed_field
    @property
    def project_irr(self) -> Calculation:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1))
        cash[:, 0] = -np.abs(self.initial_investment.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        irr = _irr_rows(cash) * 100.0
        return Calculation(
            self.scenario, "Project IRR (%)", "%", 1, I, irr.reshape(I, 1)
        )

    @computed_field
    @property
    def project_mirr(self) -> Calculation:
        I, T = self.iterations, self.years
        cash = np.zeros((I, T + 1))
        cash[:, 0] = -np.abs(self.initial_investment.data[:, 0])
        cash[:, 1:] = self.project_cashflow.data
        r = self.discount_rate.data[:, 0] / 100.0
        mirr = _mirr_rows(cash, r, r) * 100.0
        return Calculation(
            self.scenario, "Project MIRR (%)", "%", 1, I, mirr.reshape(I, 1)
        )

    # ------ DSCR -------------------------------------------------------------
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
            self.scenario, "DSCR", "unitless", self.years, self.iterations, ratio
        )

    # ------ Summaries --------------------------------------------------------
    def npv_summary(self, percentiles: Sequence[float] = (5, 50, 95)) -> List[NPVRow]:
        rows: List[NPVRow] = []
        for name, vals in self.npv.items():
            p = np.percentile(vals, percentiles)
            rows.append(
                NPVRow(name=name, p5=float(p[0]), p50=float(p[1]), p95=float(p[2]))
            )
        total = self.npv_total()
        p = np.percentile(total, percentiles)
        rows.append(
            NPVRow(name="TOTAL", p5=float(p[0]), p50=float(p[1]), p95=float(p[2]))
        )
        return rows
