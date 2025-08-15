from __future__ import annotations
from typing import TypeVar, Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field

# Project modules
from models.core.types import Input, Output, Calculation, PandasDataFrame, NdArray
from models.financial import FinancialModel

np.float_ = np.float64


class ValuePerAcre(FinancialModel):
    """
    Starter template to model municipal value-per-acre using the CommonModelMixin/FinancialModel
    pattern found in this codebase.

    ▶ How to use
      1) Add your inputs to an inputs YAML (e.g., inputs/Base.yaml) with the same field names.
      2) Instantiate: m = ValuePerAcreProject(scenario="Base", years=30, iterations=2000)
      3) Explore .npv, .npv_summary, .kpi, charts, and sensitivity tools from FinancialModel.

    Design notes
      • All inputs are stochastic ModelInput objects (draws shape = [iterations, years]).
      • Computations use @computed_field and return ModelCalculation/ModelOutput with arrays of the
        same shape, enabling timeseries charts and NPV via FinancialModel.
      • Per-acre vs. total: we keep inputs as *per-acre* where possible, then multiply by acres to
        get totals. A dedicated per-acre KPI is provided.
    """

    # ----------------------
    # Core land/intensity inputs
    # ----------------------
    parcel_acres: Input                    # Total parcel size (acres)
    assessed_value_per_acre: Input         # Taxable assessed value per acre ($/acre)
    assessed_value_growth_rate: Input      # Annual % growth of assessed value

    # ----------------------
    # Revenues (per acre)
    # ----------------------
    property_tax_rate: Input               # Mill rate in %, applied to assessed value
    other_revenue_per_acre: Input          # Fees/utility/franchise/etc. ($/acre/yr)
    other_revenue_growth_rate: Input       # %/yr escalation of other revenue

    # ----------------------
    # Costs (public burden per acre)
    # ----------------------
    capex_public_per_acre: Input           # One-time public capital cost at year 0 ($/acre)
    opex_public_per_acre: Input            # Recurring public service cost ($/acre/yr)
    opex_growth_rate: Input                # %/yr escalation of service cost

    # ----------------------
    # Finance
    # ----------------------
    discount_rate: Input                   # Required by FinancialModel for NPV

    # Optional helper: proportion of first year active (e.g., commissioning mid-year)
    start_year_proportion: Input           # 0–1 fraction applied to year 0 flows

    # ========== Helper schedules ==========

    @computed_field
    @property
    def years_array(self) -> NdArray:
        # Override to include year 0 properly when we seed capex at t=0
        return np.arange(self.years)

    @computed_field
    @property
    def growth_schedule_assessed(self) -> Calculation:
        """(1 + g) ** t for assessed value growth."""
        arr = (1 + (self.assessed_value_growth_rate / 100)) ** self.years_array
        return Calculation(
            scenario=self.scenario,
            label="assessed_growth",
            description="Assessed value growth factor",
            units="unitless",
            data=arr,
        )

    @computed_field
    @property
    def growth_schedule_other_rev(self) -> Calculation:
        arr = (1 + (self.other_revenue_growth_rate / 100)) ** self.years_array
        return Calculation(
            scenario=self.scenario,
            label="other_rev_growth",
            description="Other revenue growth factor",
            units="unitless",
            data=arr,
        )

    @computed_field
    @property
    def growth_schedule_opex(self) -> Calculation:
        arr = (1 + (self.opex_growth_rate / 100)) ** self.years_array
        return Calculation(
            scenario=self.scenario,
            label="opex_growth",
            description="OPEX growth factor",
            units="unitless",
            data=arr,
        )

    # ========== Totals from per-acre inputs ==========

    @computed_field
    @property
    def assessed_value_total(self) -> Calculation:
        """Total assessed value each year (stochastic): $/ac x acres x growth."""
        base = self.assessed_value_per_acre * self.parcel_acres
        arr = base * self.growth_schedule_assessed
        return Calculation(
            scenario=self.scenario,
            label="assessed_total",
            description="Total assessed value (tax base)",
            units="$",
            data=arr,
        )

    @computed_field
    @property
    def property_tax_revenue(self) -> Output:
        """Annual property tax: assessed_total x (rate/100)."""
        data = self.assessed_value_total * (self.property_tax_rate / 100)
        # Apply partial first year if needed
        data[:, 0] *= self.start_year_proportion[:, 0]
        return Output(
            scenario=self.scenario,
            label="property_tax_revenue",
            description="Annual property tax revenue",
            units="$",
            data=data,
        )

    @computed_field
    @property
    def other_revenue_total(self) -> Output:
        """Other recurring revenue: $/ac/yr x acres x growth schedule."""
        base = self.other_revenue_per_acre * self.parcel_acres
        data = base * self.growth_schedule_other_rev
        data[:, 0] *= self.start_year_proportion[:, 0]
        return Output(
            scenario=self.scenario,
            label="other_revenue",
            description="Other municipal revenues attributable to parcel",
            units="$",
            data=data,
        )

    @computed_field
    @property
    def capex(self) -> Output:
        """One-time public CAPEX at year 0: $/ac x acres (zeros thereafter)."""
        n_iter, n_years = self.iterations, self.years
        data = np.zeros((n_iter, n_years))
        data[:, 0] = (self.capex_public_per_acre * self.parcel_acres)[:, 0]
        # apply partial first-year factor if commissioning mid-year (optional)
        data[:, 0] *= self.start_year_proportion[:, 0]
        return Output(
            scenario=self.scenario,
            label="capex",
            description="Initial public capital outlay",
            units="$",
            data=data,
        )

    @computed_field
    @property
    def opex(self) -> Output:
        """Annual public OPEX: $/ac/yr x acres x growth schedule."""
        base = self.opex_public_per_acre * self.parcel_acres
        data = base * self.growth_schedule_opex
        data[:, 0] *= self.start_year_proportion[:, 0]
        return Output(
            scenario=self.scenario,
            label="opex",
            description="Annual public operating costs",
            units="$",
            data=data,
        )

    # ========== Roll-ups ==========

    @computed_field
    @property
    def total_revenue(self) -> Output:
        data = self.property_tax_revenue + self.other_revenue_total
        return Output(
            scenario=self.scenario,
            label="total_revenue",
            description="Total municipal revenue from the parcel",
            units="$",
            data=data,
        )

    @computed_field
    @property
    def net_cashflow(self) -> Output:
        """Yearly net cash available: revenue − opex − capex (capex only in y0)."""
        data = self.total_revenue - self.opex - self.capex
        return Output(
            scenario=self.scenario,
            label="net_cashflow",
            description="Net fiscal cash flow attributable to the parcel",
            units="$",
            data=data,
        )

    @computed_field
    @property
    def value_per_acre(self) -> Calculation:
        """Per-acre net cash each year for KPI/visualization (stochastic)."""
        # Guard against divide-by-zero
        acres = np.where(self.parcel_acres.data == 0, np.nan, self.parcel_acres.data)
        data = self.net_cashflow.data / acres
        return Calculation(
            scenario=self.scenario,
            label="value_per_acre",
            description="Annual net fiscal value per acre",
            units="$ per acre",
            data=data,
        )

    # ---------- Percentile helpers for dashboards ----------

    @computed_field
    @property
    def value_per_acre_percentiles(self) -> PandasDataFrame:
        raw = self.value_per_acre.data
        p10 = np.nanpercentile(raw, 10, axis=0)
        p50 = np.nanpercentile(raw, 50, axis=0)
        p90 = np.nanpercentile(raw, 90, axis=0)
        df = pd.DataFrame({"P10": p10, "P50": p50, "P90": p90})
        df.index.name = "Year"
        return df

    # ---------- Convenience accessors for NPV ----------

    @computed_field
    @property
    def npv_metrics(self) -> Dict[str, NdArray]:
        """
        Alias to FinancialModel.npv so downstream code can do m.npv_metrics["net_cashflow"], etc.
        Requires discount_rate input.
        """
        return self.npv

    @computed_field
    @property
    def per_acre_npv(self) -> Calculation:
        """NPV per acre of the net cashflow stream."""
        rates = self.discount_rate[:, 0] / 100
        # Build cashflow including year 0 explicitly
        arr = self.net_cashflow.data
        # Value per acre NPV (iteration-wise)
        acres = np.where(self.parcel_acres.data[:, 0] == 0, np.nan, self.parcel_acres.data[:, 0])
        # Prepend zero to align with numpy_financial conventions if needed (FinancialModel handles) –
        # here we directly compute with the helper on ModelOutput when desired.
        # Use same utility as ModelOutput.npv for consistency
        from numpy_financial import npv as _npv
        iters = arr.shape[0]
        # Convert to list of iteration NPVs
        vals = np.array([_npv(rates[i], np.concatenate([[0.0], arr[i]]) ) for i in range(iters)])
        per_acre_vals = vals / acres
        data = per_acre_vals.reshape(iters, 1)
        return Calculation(
            scenario=self.scenario,
            label="per_acre_npv",
            description="NPV per acre of net municipal cash flows",
            units="$ per acre",
            data=data,
        )



