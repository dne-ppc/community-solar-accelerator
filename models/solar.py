from typing import TypeVar
from pydantic import computed_field
from models.base import ModelInput, ModelOutput, ModelCalculation, FinancialModel
import numpy as np
import pandas as pd
import numpy_financial as npf
import streamlit as st

from models.simulation import Simulation

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
np.float_ = np.float64

HOURS_PER_YEAR = 8766


class SystemOutputCalcMixin:

    panel_power: ModelInput
    number_of_panels: ModelInput

    @computed_field
    @property
    def system_output(self) -> ModelCalculation:
        """
        Total system output in kWh per year:
        panel_power (W) * number_of_panels * capacity_factor (%) * HOURS_PER_YEAR, divided by 1000.
        """
        arr = (
            self.panel_power
            * self.number_of_panels
            * (self.capacity_factor / 100)
            * HOURS_PER_YEAR
            / 1000
        )
        return ModelCalculation(
            scenario=self.scenario,
            label="system_output",
            description="Annual system output (kWh)",
            units="kWh",
            data=arr,
        )

    @computed_field
    @property
    def capex(self) -> ModelOutput:
        """
        Capital expenditure (CAPEX) as install_cost ($/W) * panel_power (W) * number_of_panels.
        """
        arr = self.install_cost * self.panel_power * self.number_of_panels
        return ModelOutput(
            scenario=self.scenario,
            label="CAPEX ($CAD)",
            description="Capital expenditure",
            units="$CAD",
            data=arr,
        )


class SystemInputMixin:

    system_output: ModelInput

    @computed_field
    @property
    def capex(self) -> ModelOutput:
        """
        Capital expenditure (CAPEX) as install_cost ($/W) * panel_power (W) * number_of_panels.
        """
        arr = self.install_cost * (
            self.system_output / HOURS_PER_YEAR / (self.capacity_factor / 100)
        )
        return ModelOutput(
            scenario=self.scenario,
            label="CAPEX ($CAD)",
            description="Capital expenditure",
            units="$CAD",
            data=arr,
        )


class SolarProject(FinancialModel):

    capacity_factor: ModelInput
    degradation_rate: ModelInput
    install_cost: ModelInput

    public_funding_percent: ModelInput
    funding_buffer_percent: ModelInput
    itc_rate: ModelInput
    itc_year: ModelInput
    start_year_proportion: ModelInput

    electricity_price: ModelInput
    inflation_rate: ModelInput
    discount_rate: ModelInput
    maintenance_rate: ModelInput
    admin_rate: ModelInput
    insurance_rate: ModelInput

    dividend_rate: ModelInput

    dividend_start_year: ModelInput
    capital_return_year: ModelInput
    return_period: ModelInput

    system_output: ModelInput

    # TODO
    # commissioning_fraction: ModelInput  # (or just a float, but ModelInput is more flexible)

    @computed_field
    @property
    def capex(self) -> ModelOutput:
        """
        Capital expenditure (CAPEX) as install_cost ($/W) * panel_power (W) * number_of_panels.
        """
        arr = (
            self.install_cost
            * (self.system_output / HOURS_PER_YEAR / (self.capacity_factor / 100))
            * 1000
        )
        return ModelOutput(
            scenario=self.scenario,
            label="CAPEX ($CAD)",
            description="Capital expenditure",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def seed_capital(self) -> ModelCalculation:
        """
        Seed capital needed: capex * (1 + self.funding_buffer_percent / 100.0).
        """
        arr = self.capex * (1 + self.funding_buffer_percent / 100.0)
        return ModelCalculation(
            scenario=self.scenario,
            label="seed_capital",
            description="Initial seed capital from public funding",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def public_investment(self) -> ModelCalculation:
        """
        Public investment portion: seed_capital + funding_buffer_percent buffer.
        """
        arr = self.seed_capital * self.public_funding_percent / 100.0
        return ModelCalculation(
            scenario=self.scenario,
            label="public_investment",
            description="Total public investment including buffer",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def private_investment(self) -> ModelCalculation:
        """
        Private investment portion: capex - public_investment.
        """
        arr = self.seed_capital * (1 - self.public_funding_percent / 100.0)
        return ModelCalculation(
            scenario=self.scenario,
            label="private_investment",
            description="Equity sought from private investors",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def degradation_schedule(self) -> ModelCalculation:
        """
        Yearly degradation factor: (1 - degradation_rate/100) ** years_array.
        """
        arr = np.ones((self.iterations, self.years))
        arr[:,1:] = (1 - self.degradation_rate / 100) ** self.years_array
        return ModelCalculation(
            scenario=self.scenario,
            label="degradation_schedule",
            description="Annual degradation multiplier",
            units="unitless",
            data=arr,
        )

    @computed_field
    @property
    def production(self) -> ModelCalculation:
        """
        Annual energy production (kWh): system_output * degradation_schedule.
        """
        arr = self.system_output * self.degradation_schedule
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return ModelCalculation(
            scenario=self.scenario,
            label="production",
            description="Annual energy produced",
            units="kWh",
            data=arr,
        )

    # TODO Update all capex derived calcs with similar logic
    # @computed_field
    # @property
    # def production(self) -> ModelCalculation:
    #     """
    #     Annual energy production (kWh): system_output * degradation_schedule,
    #     with year 0 scaled by commissioning_fraction.
    #     """
    #     arr = self.system_output * self.degradation_schedule  # shape: (iterations, years)

    #     # Scale year 0 by commissioning_fraction, all other years by 1.0
    #     # commissioning_fraction: shape (iterations, 1) or (iterations,)
    #     frac = np.asarray(self.commissioning_fraction.data).reshape(-1, 1)
    #     scale = np.ones(arr.shape)
    #     scale[:, 0] = frac[:, 0]  # Apply fraction only to year 0

    #     arr = arr * scale

    #     return ModelCalculation(
    #         scenario=self.scenario,
    #         label="production",
    #         description="Annual energy produced (with partial Year 0)",
    #         units="kWh",
    #         data=arr,
    #     )

    @computed_field
    @property
    def escalation_schedule(self) -> ModelCalculation:
        """
        Inflation escalation factor: (1 + inflation_rate/100) ** years_array.
        """
        arr = np.zeros((self.iterations, self.years))
        arr[:,0] = self.capex[:,0]
        arr[:,1:] = ((1 + self.inflation_rate / 100) ** self.years_array) * self.capex
        return ModelCalculation(
            scenario=self.scenario,
            label="escalation",
            description="Inflation escalation factor",
            units="unitless",
            data=arr,
        )

    @computed_field
    @property
    def maintenance_cost(self) -> ModelCalculation:
        """
        Annual maintenance cost: (maintenance_rate/100) * escalation * capex.
        """
        arr = (self.maintenance_rate / 100) * self.escalation_schedule

        arr[:, 0] *= self.start_year_proportion[:, 0]

        return ModelCalculation(
            scenario=self.scenario,
            label="maintenance_cost",
            description="Annual maintenance cost adjusted for inflation",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def admin_cost(self) -> ModelCalculation:
        """
        Annual administrative cost: (admin_rate/100) * escalation * capex.
        """
        arr = (self.admin_rate / 100) * self.escalation_schedule
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return ModelCalculation(
            scenario=self.scenario,
            label="admin_cost",
            description="Annual administrative cost adjusted for inflation",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def insurance_cost(self) -> ModelCalculation:
        """
        Annual insurance cost: (insurance_rate/100) * escalation * capex.
        """
        arr = (self.insurance_rate / 100) * self.escalation_schedule
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return ModelCalculation(
            scenario=self.scenario,
            label="insurance_cost",
            description="Annual insurance cost adjusted for inflation",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def opex(self) -> ModelOutput:
        """
        Total operating expenses (OPEX) per year: maintenance_cost + admin_cost + insurance_cost.
        """
        costs = self.maintenance_cost + self.admin_cost + self.insurance_cost
        return ModelOutput(
            scenario=self.scenario,
            label="opex",
            description="Total operating expenses per year",
            units="$CAD",
            data=costs,
        )

    @computed_field
    @property
    def revenue(self) -> ModelOutput:
        """
        Annual revenue: production * electricity_price.
        """
        return ModelOutput(
            scenario=self.scenario,
            label="revenue",
            description="Annual revenue from energy sales",
            units="$CAD",
            data=self.production * self.electricity_price + self.tax_revenue,
        )

    @computed_field
    @property
    def operating_margin(self) -> ModelCalculation:
        """
        Operating margin per year: revenue - opex.
        """
        return ModelCalculation(
            scenario=self.scenario,
            label="operating_margin",
            description="Operating margin per year",
            units="$CAD",
            data=self.revenue - self.opex,
        )

    @computed_field
    @property
    def capital_returned(self) -> ModelCalculation:
        """
        Annual capital returned to investors: spread equally across return_period,
        starting at capital_return_year, ending at capital_return_year + return_period - 1.
        """
        n_iter, n_years = self.iterations, self.years
        payments = np.zeros((n_iter, n_years))

        # All are (n_iter,)
        capital_start = self.capital_return_year[:, 0].astype(int) - 1  # 0-based index
        return_periods = self.return_period[:, 0].astype(int)
        private_investment = self.private_investment[:, 0]

        # Compute annual payment per iteration (avoid div by zero)
        # If return_period is 0, just set to 0 (no repayment)
        with np.errstate(divide="ignore", invalid="ignore"):
            annual_payment = np.where(
                return_periods > 0, private_investment / return_periods, 0
            )

        # Set up years array for broadcasting (n_years,)
        years = np.arange(n_years)
        # Broadcast to (n_iter, n_years)
        years_matrix = np.broadcast_to(years, (n_iter, n_years))

        # Mask: pay only from capital_start (inclusive) to capital_start + return_period - 1 (inclusive)
        mask = (years_matrix >= capital_start[:, None]) & (
            years_matrix < (capital_start + return_periods)[:, None]
        )

        # Apply payments
        payments = np.where(mask, annual_payment[:, None], 0)

        return ModelCalculation(
            scenario=self.scenario,
            label="capital_returned",
            description="Annual capital returned to investors (repayment finishes after return_period)",
            units="$CAD",
            data=payments,
        )

    @computed_field
    @property
    def remaining_private_investment(self) -> ModelCalculation:
        """
        Remaining private investment after dividend payments.
        """
        remaining = np.zeros((self.iterations, self.years))
        current_investment = self.private_investment[:, 0]

        for year in range(0, self.years):
            current_investment: np.ndarray = (
                current_investment - self.capital_returned[:, year]
            )
            current_investment[np.abs(current_investment) < 1e-9] = 0
            remaining[:, year] = current_investment
        return ModelCalculation(
            scenario=self.scenario,
            label="remaining_private_investment",
            description="Outstanding private investment over time",
            units="$CAD",
            data=remaining,
        )

    @computed_field
    @property
    def investor_dividends(self) -> ModelCalculation:
        """
        Dividends distributed to investors once capital_returned phases out.
        """
        dividends = np.zeros((self.iterations, self.years))

        if self.years <= 1:
            # Not enough years for dividends (need at least year 1)
            return ModelCalculation(
                scenario=self.scenario,
                label="investor_dividends",
                description="Annual dividends paid to investors",
                units="$CAD",
                data=dividends,
            )

        investment = self.remaining_private_investment
        rates = self.dividend_rate[:, 0] / 100

        # Extract dividend start years (convert to 0-based indexing)
        dividend_start_years = self.dividend_start_year[:, 0].astype(int) - 1

        # Ensure dividend start years are within valid range
        dividend_start_years = np.clip(dividend_start_years, 0, self.years - 1)

        # Create year indices for broadcasting (excluding year 0 since we start from year 1)
        year_indices = np.arange(1, self.years)  # [1, 2, 3, ..., years-1]

        # Create mask for when dividends should be paid: (iterations x years-1)
        dividend_mask = year_indices >= dividend_start_years[:, np.newaxis]

        # Get previous year's investment for dividend calculation
        # Ensure we don't go out of bounds
        prev_year_indices = np.clip(year_indices - 1, 0, investment.shape[1] - 1)
        prev_year_investment = investment[:, prev_year_indices]

        # Calculate dividends for all valid positions
        # rates[:, np.newaxis] broadcasts rates across years
        dividend_amounts = prev_year_investment * rates[:, np.newaxis]

        # Apply dividends only where mask is True
        dividends[:, 1:] = np.where(dividend_mask, dividend_amounts, 0)

        return ModelCalculation(
            scenario=self.scenario,
            label="investor_dividends",
            description="Annual dividends paid to investors",
            units="$CAD",
            data=dividends,
        )

    @computed_field
    @property
    def finance_costs(self) -> ModelOutput:
        """
        Annual financing cost: interest on remaining_private_investment at capital_return_rate.
        """
        return ModelOutput(
            scenario=self.scenario,
            label="Finance Costs ($CAD)",
            description="Annual financing cost",
            units="$CAD",
            data=self.capital_returned + self.investor_dividends,
        )

    @computed_field
    @property
    def retained_earnings(self) -> ModelOutput:
        """
        Retained earnings each year: cumulative sum of cashflow.
        """

        return ModelOutput(
            scenario=self.scenario,
            label="Retained Earnings ($CAD)",
            description="Cumulative retained earnings",
            units="$CAD",
            data=self.operating_margin - self.finance_costs,
        )

    @computed_field
    @property
    def total_cash(self) -> ModelCalculation:
        """
        Total cumulative cash in the project over time.
        Starts with initial investment and accumulates operating margin minus finance costs.
        """
        # Start with total seed capital (public + private investment)
        initial_cash = (
            self.seed_capital[:, 0] - self.capex[:, 0]
        )  # Shape: (iterations,)

        # Net cash flow each year (operating margin - finance costs)
        annual_net_cash_flow = (
            self.retained_earnings
        )  # This is already operating_margin - finance_costs

        # Initialize cumulative cash array
        cumulative_cash = np.zeros((self.iterations, self.years))

        # Set initial cash for year 0
        cumulative_cash[:, 0] = initial_cash + annual_net_cash_flow[:, 0]

        # Accumulate cash flows for subsequent years
        for year in range(1, self.years):
            cumulative_cash[:, year] = (
                cumulative_cash[:, year - 1] + annual_net_cash_flow[:, year]
            )

        return ModelCalculation(
            scenario=self.scenario,
            label="Total Project Cash ($CAD)",
            description="Cumulative cash balance in the project over time",
            units="$CAD",
            data=cumulative_cash,
        )

    def _has_sign_change(self, cash: np.ndarray) -> bool:
        """
        IRR only exists if cash flows switch sign at least once.
        Example: [-100, +120, -10] has two sign changes → IRR might exist.
        No sign change (all <= 0 or all >= 0) → IRR cannot exist.
        """
        sign = np.sign(cash)
        # remove any zeros (exact zeros can break the logic)
        nonzero = sign[sign != 0]
        return np.any(nonzero[:-1] != nonzero[1:])

    @computed_field
    @property
    def private_investor_irr(self) -> ModelCalculation:
        """
        Vectorized IRR for private investors, but only attempt solver if there's a sign change.
        Returns IRR in percent. Any iteration with no valid IRR → np.nan.
        """
        n_iter = self.iterations
        n_years = self.years

        # Build the cash‐flow matrix (n_iter × (n_years+1))
        cash_flows = np.zeros((n_iter, n_years + 1))
        cash_flows[:, 0] = -self.private_investment[:, 0]  # Year 0 outflow
        cash_flows[:, 1:] = self.capital_returned + self.investor_dividends

        irr_array = np.full((n_iter, 1), np.nan, dtype=float)

        for i in range(n_iter):
            cf = cash_flows[i, :]
            if self._has_sign_change(cf):
                try:
                    raw = npf.irr(cf)
                except Exception:
                    raw = np.nan
                # Convert to percentage, but only if raw is finite
                irr_array[i, 0] = float(raw * 100) if np.isfinite(raw) else np.nan
            # else: leave as np.nan

        return ModelCalculation(
            scenario=self.scenario,
            label="Private Investor IRR (%)",
            description="Internal Rate of Return for private‐equity investors",
            units="%",
            data=irr_array,
        )

    @computed_field
    @property
    def project_irr(self) -> ModelCalculation:
        """
        Vectorized IRR for the entire project (all capital).
        Same pattern: check for sign change, then npf.irr, else np.nan.
        """
        n_iter = self.iterations
        n_years = self.years

        cash_flows = np.zeros((n_iter, n_years + 1))
        cash_flows[:, 0] = -self.capex[:, 0]  # Year 0 outflow
        cash_flows[:, 1:] = self.retained_earnings

        irr_array = np.full((n_iter, 1), np.nan, dtype=float)

        for i in range(n_iter):
            cf = cash_flows[i, :]
            if self._has_sign_change(cf):
                try:
                    raw = npf.irr(cf)
                except Exception:
                    raw = np.nan
                irr_array[i, 0] = float(raw * 100) if np.isfinite(raw) else np.nan

        return ModelCalculation(
            scenario=self.scenario,
            label="Project IRR (%)",
            description="Internal Rate of Return for the overall project",
            units="%",
            data=irr_array,
        )

    @computed_field
    @property
    def private_investor_irr_percentiles(self) -> PandasDataFrame:
        """
        P10/P50/P90 of private_investor_irr (ignoring NaNs).
        """
        raw = self.private_investor_irr.data.flatten()
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.nanpercentile(finite, [10, 50, 90])
        df = pd.DataFrame(
            {self.private_investor_irr.label: [p10, p50, p90]},
            index=["P10", "P50", "P90"],
        )
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def project_irr_percentiles(self) -> PandasDataFrame:
        """
        P10/P50/P90 of project_irr (ignoring NaNs).
        """
        raw = self.project_irr.data.flatten()
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.nanpercentile(finite, [10, 50, 90])
        df = pd.DataFrame(
            {self.project_irr.label: [p10, p50, p90]}, index=["P10", "P50", "P90"]
        )
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def years_to_self_fund(self) -> ModelCalculation:
        """
        (Exactly as before) Returns a ModelCalculation whose .data is
        shape (n_iter, 1), with each row = first 1-based year where
        total_project_cash ≥ capex, or –1 if never self-funded.
        """
        cash_data = self.total_cash.data  # shape: (n_iter, n_years)
        capex_thresh = self.capex[:, 0]  # shape: (n_iter,)

        n_iter, n_years = cash_data.shape
        years_arr = np.full((n_iter, 1), 9999, dtype=int)

        mask = cash_data >= capex_thresh[:, None]
        ever = mask.any(axis=1)
        if ever.any():
            first_idx = mask.argmax(axis=1)
            years_arr[ever, 0] = first_idx[ever] + 1

        return ModelCalculation(
            scenario=self.scenario,
            label="Years to Self Fund",
            data=years_arr,
            years=self.years,
            units="#",
            description="Years taken until the project has sufficient cash to fund capex again",
        )

    @computed_field
    @property
    def years_to_self_fund_percentiles(self) -> PandasDataFrame:
        """
        Summarize `years_to_self_fund.data` at the (inverted) percentiles:
          • “10%” → the 90th percentile of raw payback (worst‐case, longest years)
          • “50%” → the 50th percentile (median)
          • “90%” → the 10th percentile (best‐case, shortest years)
        Returns a DataFrame indexed by [10, 50, 90].
        """
        # 1) Grab the raw (n_iter × 1) array and flatten to (n_iter,)
        raw = self.years_to_self_fund.data.flatten()  # values in {1…n_years} or –1

        # 2) If you’d rather ignore “–1” (never self-funded), filter them out. Otherwise
        #    including “–1” will pull the percentiles downward (i.e. counting never-funded as a zeroish value).
        valid = raw  # drop any –1 entries
        if valid.size == 0:
            # If no iteration ever self-funds, just return –1 for all percentiles:
            df = pd.DataFrame({"years_to_self_fund": [-1, -1, -1]}, index=[10, 50, 90])
            df.index.name = "percentile"
            return df

        # 3) We want:
        #      row “10” → 90th percentile of valid
        #      row “50” → 50th percentile
        #      row “90” → 10th percentile
        raw_pcts = np.percentile(valid, [90, 50, 10])

        df = pd.DataFrame(
            {self.years_to_self_fund.label: raw_pcts}, index=["P10", "P50", "P90"]
        ).replace(9999, np.nan)
        df.index.name = "Metric"

        return df.T

    @computed_field
    @property
    def cash_depletion_year(self) -> ModelCalculation:
        """
        For iterations that go negative, the year when cash first becomes negative.
        Returns -1 for iterations that never go negative.
        """
        cash_data = self.total_cash.data  # shape: (n_iter, n_years)
        n_iter, n_years = cash_data.shape
        # Boolean mask where cash < 0
        mask = cash_data < 0
        # Initialize with -1
        depletion_years = np.full(n_iter, 9999, dtype=int)
        # Identify iterations with any negative cash
        ever = mask.any(axis=1)
        # Find index of first True along axis=1
        first_idx = mask.argmax(axis=1)
        # Assign first negative year (+1 for 1-based)
        depletion_years[ever] = first_idx[ever] + 1
        # Reshape to (n_iter, 1)
        result_array = depletion_years.reshape(self.iterations, 1)
        return ModelCalculation(
            scenario=self.scenario,
            label="Cash Depletion Year",
            description="Year when project cash first goes negative (-1 if never negative)",
            units="year",
            data=result_array,
        )

    @computed_field
    @property
    def cash_depletion_year_percentiles(self) -> PandasDataFrame:
        """
        Summarize `cash_depletion_year.data` at the percentiles:
          • “P10” → 90th percentile (worst-case, latest depletion)
          • “P50” → 50th percentile (median)
          • “P90” → 10th percentile (best-case, earliest depletion)
        Returns a DataFrame indexed by ['P10', 'P50', 'P90'].
        """
        # Flatten raw depletion years
        raw = self.cash_depletion_year.data.flatten()  # values in {1…n_years} or -1
        valid = raw
        if valid.size == 0:
            df = pd.DataFrame(
                {"cash_depletion_year": [-1, -1, -1]}, index=["P10", "P50", "P90"]
            )
            df.index.name = "Metric"
            return df
        # Compute percentiles: 90th, 50th, 10th
        raw_pcts = np.percentile(valid, [10, 50, 90])
        df = pd.DataFrame(
            {self.cash_depletion_year.label: raw_pcts}, index=["P10", "P50", "P90"]
        ).replace(9999, np.nan)
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def profitability_index(self) -> ModelCalculation:
        """
        Profitability Index (PV of inflows / PV of outflows).
        """
        # Number of iterations and years
        n_iter = self.iterations
        n_years = self.years

        # Build cash-flow matrix: shape (n_iter, n_years+1)
        cash_flows = np.zeros((n_iter, n_years + 1))
        # Initial outflow at t=0: capex
        cash_flows[:, 0] = -self.capex.data[:, 0]
        # Annual net cash flows (retained earnings) for t=1...n_years
        cash_flows[:, 1:] = self.retained_earnings.data

        # Discount rates per iteration
        rates = self.discount_rate.data.flatten()  # shape (n_iter,)
        # Time periods 0 through n_years
        periods = np.arange(n_years + 1)
        # Discount factors: (1 + rate) ** t
        discount_factors = (1 + rates)[:, None] ** periods[None, :]

        # Present value of each cash-flow
        pv = cash_flows / discount_factors

        # Sum positive and negative PVs
        pv_inflows = np.sum(np.where(pv > 0, pv, 0), axis=1)
        pv_outflows = np.sum(np.where(pv < 0, -pv, 0), axis=1)

        # Compute PI, avoiding divide-by-zero
        pi_vals = np.where(pv_outflows > 0, pv_inflows / pv_outflows, np.nan)
        # Reshape to (n_iter, 1) for ModelCalculation
        pi_matrix = pi_vals.reshape(n_iter, 1)

        return ModelCalculation(
            scenario=self.scenario,
            label="Profitability Index",
            description="PV of inflows divided by PV of outflows",
            units="unitless",
            data=pi_matrix,
        )

    @computed_field
    @property
    def payback_period(self) -> ModelCalculation:
        """
        Undiscounted payback period: first 1-based year when cumulative undiscounted cash sum ≥ 0,
        based on initial capex and annual retained earnings. Returns 9999 if never reached.
        """
        n_iter = self.iterations
        n_years = self.years
        # Construct cash flows: t=0 outflow and t=1..n retained earnings
        cash_flows = np.zeros((n_iter, n_years + 1))
        cash_flows[:, 0] = -self.capex.data[:, 0]
        cash_flows[:, 1:] = self.retained_earnings.data

        # Compute cumulative sum along time axis
        cum_cf = np.cumsum(cash_flows, axis=1)

        # Initialize payback with sentinel
        payback = np.full((n_iter, 1), 9999, dtype=int)

        # For each iteration, find first year index where cum_cf >=0
        mask = cum_cf >= 0
        ever = mask.any(axis=1)
        if ever.any():
            first_indices = mask.argmax(axis=1)
            # Convert to 1-based years (index corresponds to year)
            payback[ever, 0] = first_indices[ever]

        return ModelCalculation(
            scenario=self.scenario,
            label="Payback Period",
            description="Undiscounted payback period in years (first year where cumulative cash ≥ 0)",
            units="year",
            data=payback,
        )

    @computed_field
    @property
    def payback_year_percentiles(self) -> PandasDataFrame:
        """
        Compute payback period percentiles across iterations.

        Parameters
        ----------
        percentiles : tuple of int
            Percentile levels to compute (e.g., (10, 50, 90)).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by percentile label ('p10', 'p50', etc.) with a single
            column 'year' giving the payback period at that percentile.
        """
        raw = self.payback_period.data.flatten()
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.nanpercentile(finite, [10, 50, 90])
        df = pd.DataFrame(
            {self.payback_period.label: [p10, p50, p90]}, index=["P90", "P50", "P10"]
        ).replace(9999, np.nan)
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def roi(self) -> ModelCalculation:
        """
        Undiscounted return on investment: net cash at end divided by initial investment.
        """
        n_iter = self.iterations
        n_years = self.years
        cash_flows = np.zeros((n_iter, n_years + 1))
        cash_flows[:, 0] = -self.capex.data[:, 0]
        cash_flows[:, 1:] = self.retained_earnings.data

        cum_cf = np.cumsum(cash_flows, axis=1)
        final_cf = cum_cf[:, -1]
        initial_inv = -cash_flows[:, 0]
        roi_vals = np.where(initial_inv > 0, final_cf / initial_inv, np.nan)
        roi_matrix = roi_vals.reshape(n_iter, 1)

        return ModelCalculation(
            scenario=self.scenario,
            label="ROI (%)",
            description="Undiscounted return on investment: net cash at end divided by initial investment",
            units="unitless",
            data=roi_matrix,
        )

    @computed_field
    @property
    def roi_percentiles(self) -> PandasDataFrame:
        """
        Compute ROI percentiles across iterations.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ['P10', 'P50', 'P90'] with a single column 'ROI' giving the ROI at that percentile.
        """
        raw = self.roi.data.flatten()
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.nanpercentile(finite, [10, 50, 90])
        df = pd.DataFrame(
            {self.roi.label: [p10, p50, p90]}, index=["P10", "P50", "P90"]
        ).replace(9999, np.nan)
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def tax_revenue(self) -> ModelCalculation:
        """
        Investment Tax Credit revenue: initial CAPEX × ITC rate, claimed in ITC year.
        """
        n_iter = self.iterations
        n_years = self.years
        credits = np.zeros((n_iter, n_years))

        # ITC year and rate
        itc_years = self.itc_year.data.flatten().astype(int)
        rates = self.itc_rate.data.flatten() / 100.0
        # Initial CAPEX per iteration
        init_capex = self.private_investment.data[:, 0]

        # Only assign credits where ITC year is valid
        valid = (itc_years >= 0) & (itc_years <= n_years)
        idx = np.arange(n_iter)[valid]
        years = itc_years[valid]
        credits[idx, years] = init_capex[valid] * rates[valid]

        return ModelCalculation(
            scenario=self.scenario,
            label="Tax Revenue",
            description="Investment tax credit claimed in specified year",
            units=self.capex.units,
            data=credits,
        )

    def iteration_summary(self) -> None:

        iteration = st.number_input(
            "Iteration",
            min_value=0,
            max_value=self.iterations - 1,
            value=0,
            step=1,
            help="Select the iteration to view detailed financials.",)

        investment = pd.DataFrame(
            data=[
                self.system_output[iteration],
                self.private_investment[iteration],
                self.public_investment[iteration],
                self.seed_capital[iteration],
                self.dividend_rate[iteration],
                self.dividend_start_year[iteration],
                self.capital_return_year[iteration],
                self.return_period[iteration],
                self.discount_rate[iteration],
            ],
            index=[
                "System Output",
                "Private Investment",
                "Public Investment",
                "Seed Capital",
                "Dividend Rate",
                "Dividend Start Year",
                "Capital Return Start Year",
                "Return Periods",
                "Discount Rate",
            ],
            columns=[""],
        )

        system = pd.DataFrame(
            data=[
                self.system_output[iteration],
                # self.electricity_price[iteration,0:1],
                self.degradation_rate[iteration],
            ],
            index=[
                "System Output",
                # "Electricity Price",
                "Panel Degradation Rate",
            ],
            columns=[""],
        )

        costs = pd.DataFrame(
            data=[
                self.maintenance_rate[iteration],
                self.admin_rate[iteration],
                self.insurance_rate[iteration],
                self.inflation_rate[iteration],
            ],
            index=[
                "Maintenance Cost",
                "Administrative Cost",
                "Insurance Cost",
                "Inflation Rate",
            ],
            columns=[""],
        )

        annual = pd.DataFrame(
            data=[
                self.production[iteration],
                self.maintenance_cost[iteration],
                self.admin_cost[iteration],
                self.insurance_cost[iteration],
                self.opex[iteration],
                self.tax_revenue[iteration],
                self.revenue[iteration],
                self.operating_margin[iteration],
                self.investor_dividends[iteration],
                self.capital_returned[iteration],
                self.finance_costs[iteration],
                self.remaining_private_investment[iteration],
                self.retained_earnings[iteration],
            ],
            index=[
                "Energy Production (kWh)",
                "Maintenance Cost",
                "Admin Cost",
                "Insurance Cost",
                "OPEX",
                "Tax Revenue",
                "Revenue",
                "Operating Margin",
                "Investor Dividends",
                "Capital Returned",
                "Finance Costs",
                "Remaining Private Investment",
                "Retained Earnings",
            ],
            columns=range(2026, 2026 + self.years),
        )

        revenue = self.revenue.npv_iter(iteration, self.discount_rate / 100)
        opex = self.opex.npv_iter(iteration, self.discount_rate / 100)
        captial = self.private_investment[iteration]
        finance = self.investor_dividends.npv_iter(iteration, self.discount_rate / 100)
        total = revenue - opex - captial - finance

        npv = pd.DataFrame(
            data=[
                revenue,
                opex,
                captial,
                finance,
                total

            ],
            index=[
                "NPV Revenue",
                "NPV OPEX",
                "NPV Capital",
                "NPV Financing",
                "NPV Total"
            ],
            columns=[""],
        )

        cols = annual.columns

        annual['Total'] = annual.sum(axis=1)

        annual = annual[['Total'] + list(cols)]


        col1,col2,col3 = st.columns(3)

        with col1:
            st.subheader("Investment Assumptions")
            st.dataframe(investment.round(2))

        with col2:
            st.subheader("System Assumptions")
            st.dataframe(system.round(2))
        with col3:
            st.subheader("Cost Assumptions")
            st.dataframe(costs.round(2))

        st.subheader("Annual Financials")
        st.dataframe(annual.round(0),height=500)

        col1,col2,_ = st.columns(3)

        st.subheader("Net Present Value")
        st.dataframe(npv.round(0))

    @computed_field
    @property
    def total_npv(self)-> ModelCalculation:
        dr = self.discount_rate / 100
        revenue = self.revenue.npv(dr)
        opex = self.opex.npv(dr)
        captial = self.private_investment
        finance = self.investor_dividends.npv(dr)
        total = revenue - opex - captial - finance
        return ModelCalculation(
            scenario=self.scenario,
            label="Total NPV",
            description="Net Present Value of the project",
            units="$CAD",
            data=total,
        )

    @computed_field
    @property
    def free_cash_flow(self) -> ModelCalculation:
        """
        Cash available to the overall pool once private investors are fully repaid:
        zero until remaining_private_investment ≤ 0, then equal to retained_earnings.
        """
        # remaining private capex over time
        rem = self.remaining_private_investment.data
        # project retained earnings each year
        ret = self.retained_earnings.data                      

        # only after private capex is fully returned
        free = np.where(rem <= 0, ret, 0)

        return ModelCalculation(
            scenario=self.scenario,
            label="Free Cash Flow ($CAD)",
            description="Cashflow to pool after private investors fully repaid",
            units="$CAD",
            data=free,
        )

class SolarSimulator(Simulation):


    _model_type = SolarProject
