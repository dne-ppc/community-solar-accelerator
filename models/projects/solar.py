from typing import TypeVar
from pydantic import computed_field


from models.core.types import Input, Output, Calculation
from models.core.financial import FinancialModel
import numpy as np
import pandas as pd

# Aliases for type hints (not enforced at runtime)
PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")

HOURS_PER_YEAR = 8766


class SolarProject(FinancialModel):
    """
    Solar project subclass that preserves the existing revenue logic
    (production * electricity_price + tax_revenue) and wires into the
    shared FinancialModel metrics (DSCR, payback, IRR, etc.).

    Conventions:
      - Outflows are negative in streams (opex, finance_costs).
      - Prices/rates are in percent where indicated in Bowen.yaml.
    """

    # --- Inputs (must match Bowen.yaml keys) ---
    capex: Input
    degradation_rate: Input

    public_funding_percent: Input
    funding_buffer_percent: Input
    itc_rate: Input
    itc_year: Input
    start_year_proportion: Input

    electricity_price: Input
    inflation_rate: Input
    discount_rate: Input
    maintenance_rate: Input
    admin_rate: Input
    insurance_rate: Input

    dividend_rate: Input
    dividend_start_year: Input
    capital_return_year: Input
    return_period: Input

    system_output: Input  # Year-0 expected output (kWh)


    # ----- helpers -----
    @computed_field
    @property
    def years_array(self) -> NdArray:
        # used for exponentiation schedules (Year1..T-1)
        return np.arange(self.years - 1, dtype=float)

    # ----- capital structure -----
    @computed_field
    @property
    def seed_capital(self) -> Calculation:
        """Capex including contingency/buffer."""
        arr = self.capex * (1 + self.funding_buffer_percent / 100.0)
        return Calculation(
            scenario=self.scenario,
            label="seed_capital",
            description="Capex incl. funding buffer",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def public_investment(self) -> Calculation:
        """Public portion of seed capital."""
        arr = self.seed_capital * (self.public_funding_percent / 100.0)
        return Calculation(
            scenario=self.scenario,
            label="public_investment",
            description="Public contribution to seed capital",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def private_investment(self) -> Calculation:
        """Private portion of seed capital."""
        arr = self.seed_capital * (1.0 - self.public_funding_percent / 100.0)
        return Calculation(
            scenario=self.scenario,
            label="private_investment",
            description="Equity sought from private investors",
            units="$CAD",
            data=arr,
        )

    # Optional: explicit t=0 capex outflow for project-IRR (project level)
    @computed_field
    @property
    def capex_cashflow(self) -> Output:
        """Cash outflow at t=0 for total project capex (negative)."""
        I, T = self.iterations, self.years
        data = np.zeros((I, T), dtype=float)
        data[:, 0] = -self.capex.data[:, 0]
        return Output(
            scenario=self.scenario,
            label="Capex Outflow",
            description="t=0 project capex outflow",
            units="$CAD",
            years=T,
            iterations=I,
            data=data,
        )

    # ----- schedules -----
    @computed_field
    @property
    def degradation_schedule(self) -> Calculation:
        """(1 - degradation_rate/100) ** years for t>=1. Year0 = 1.0."""
        I, T = self.iterations, self.years
        arr = np.ones((I, T), dtype=float)
        if T > 1:
            arr[:, 1:] = (1.0 - self.degradation_rate.data[:, 0] / 100.0)[
                :, None
            ] ** self.years_array[None, :]
        return Calculation(
            scenario=self.scenario,
            label="degradation_schedule",
            description="Annual degradation multiplier",
            units="unitless",
            data=arr,
        )

    @computed_field
    @property
    def production(self) -> Calculation:
        """
        Annual kWh produced = system_output (year-0) * degradation_schedule.
        Year-0 is scaled by start_year_proportion.
        """
        arr = self.system_output * self.degradation_schedule
        # scale the commissioning (year-0) by fraction active
        arr[:, :0] *= self.start_year_proportion[:, 0]
        return Calculation(
            scenario=self.scenario,
            label="production",
            description="Annual energy produced (kWh)",
            units="kWh",
            data=arr.data,
        )

    @computed_field
    @property
    def capex_inflated(self) -> Calculation:
        """
        Capex escalated with inflation as a base for % of capex opex estimates.
        Year-0 = capex; Year t = capex * (1+pi)^(t-1).
        """
        I, T = self.iterations, self.years
        arr = np.zeros((I, T), dtype=float)
        arr[:, 0] = self.capex.data[:, 0]
        if T > 1:
            fac = (1.0 + self.inflation_rate.data[:, 0] / 100.0)[
                :, None
            ] ** self.years_array[None, :]
            arr[:, 1:] = (fac * self.capex.data[:, 0])[:, None]
        return Calculation(
            scenario=self.scenario,
            label="capex_inflated",
            description="Capex escalated with inflation (for % of capex opex rules)",
            units="$CAD",
            data=arr,
        )

    # ----- costs (positive numbers here; we make opex negative later) -----
    @computed_field
    @property
    def maintenance_cost(self) -> Calculation:
        """(maintenance_rate% of inflated capex); Year-0 scaled by start_year_proportion."""
        arr = (self.maintenance_rate / 100.0) * self.capex_inflated
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return Calculation(
            scenario=self.scenario,
            label="maintenance_cost",
            description="Annual maintenance cost",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def admin_cost(self) -> Calculation:
        arr = (self.admin_rate / 100.0) * self.capex_inflated
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return Calculation(
            scenario=self.scenario,
            label="admin_cost",
            description="Annual admin cost",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def insurance_cost(self) -> Calculation:
        arr = (self.insurance_rate / 100.0) * self.capex_inflated
        arr[:, 0] *= self.start_year_proportion[:, 0]
        return Calculation(
            scenario=self.scenario,
            label="insurance_cost",
            description="Annual insurance cost",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def opex(self) -> Output:
        """Total operating expenses – **negative** sign convention for cashflows."""
        costs = self.maintenance_cost + self.admin_cost + self.insurance_cost
        data = -costs.data
        return Output(
            scenario=self.scenario,
            label="opex",
            description="Total operating expenses (negative)",
            units="$CAD",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # ----- revenues -----
    @computed_field
    @property
    def tax_revenue(self) -> Calculation:
        """
        Investment tax credit as a positive inflow in self.itc_year (1-based).
        Applied on base capex (not inflated).
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

        return Calculation(
            scenario=self.scenario,
            label="Tax Revenue",
            description="Investment tax credit claimed in specified year",
            units="$CAD",
            data=credits,
        )



    @computed_field
    @property
    def revenue(self) -> Output:
        """Energy sales + ITC credit."""
        data = (
            self.production.data * self.electricity_price.data + self.tax_revenue.data
        )
        return Output(
            scenario=self.scenario,
            label="revenue",
            description="Annual revenue from energy sales + ITC",
            units="$CAD",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # ----- financing flows (negative = outflow from project) -----
    @computed_field
    @property
    def capital_returned(self) -> Output:
        """
        Return of private capital in equal installments over return_period,
        starting at capital_return_year (1-based). Negative (outflow).
        """
        I, T = self.iterations, self.years
        data = np.zeros((I, T), dtype=float)
        # number of installments; guard against 0
        n = np.maximum(1, np.rint(self.return_period.data[:, 0]).astype(int))
        start = np.clip(np.rint(self.capital_return_year.data[:, 0]).astype(int), 1, T)
        per = self.private_investment.data[:, 0] / n
        for i in range(I):
            s = start[i] - 1
            e = min(T, s + n)
            data[i, s:e] = -per[i]
        return Output(
            scenario=self.scenario,
            label="capital_returned",
            description="Return of private capital (negative)",
            units="$CAD",
            years=T,
            iterations=I,
            data=data,
        )

    @computed_field
    @property
    def remaining_private_investment(self) -> Calculation:
        """Outstanding private principal after scheduled returns (non-negative)."""
        I, T = self.iterations, self.years
        data = np.zeros((I, T), dtype=float)
        principal0 = self.private_investment.data[:, 0]
        out = np.full((I,), principal0, dtype=float)
        for t in range(T):
            out = out + self.capital_returned.data[:, t]  # capital_returned is negative
            data[:, t] = np.maximum(out, 0.0)
        return Calculation(
            scenario=self.scenario,
            label="remaining_private_investment",
            description="Remaining private principal",
            units="$CAD",
            data=data,
        )

    @computed_field
    @property
    def investor_dividends(self) -> Output:
        """
        Dividends at dividend_rate% of remaining private principal,
        starting at dividend_start_year. Negative (outflow).
        """
        I, T = self.iterations, self.years
        data = np.zeros((I, T), dtype=float)
        start = np.clip(np.rint(self.dividend_start_year.data[:, 0]).astype(int), 1, T)
        rate = self.dividend_rate.data[:, 0] / 100.0
        for i in range(I):
            s = start[i] - 1
            if s < T:
                data[i, s:] = -rate[i] * self.remaining_private_investment.data[i, s:]
        return Output(
            scenario=self.scenario,
            label="investor_dividends",
            description="Dividends paid to private investors (negative)",
            units="$CAD",
            years=T,
            iterations=I,
            data=data,
        )

    @computed_field
    @property
    def finance_costs(self) -> Output:
        """Total financing outflows (used by FinancialModel.debt_service)."""
        data = self.capital_returned.data + self.investor_dividends.data
        return Output(
            scenario=self.scenario,
            label="finance_costs",
            description="Capital returned + dividends (negative)",
            units="$CAD",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # Retained earnings = operating cashflow + financing flows (outflows negative)
    @computed_field
    @property
    def retained_earnings(self) -> Output:
        data = self.operating_cashflow.data + self.finance_costs.data
        return Output(
            scenario=self.scenario,
            label="retained_earnings",
            description="Net cashflow after opex and financing",
            units="$CAD",
            years=self.years,
            iterations=self.iterations,
            data=data,
        )

    # Running total cash (no opening balance; accumulates net cash)
    @computed_field
    @property
    def total_cash(self) -> Output:
        I, T = self.iterations, self.years
        data = np.cumsum(self.retained_earnings.data, axis=1)
        return Output(
            scenario=self.scenario,
            label="total_cash",
            description="Cumulative retained cash",
            units="$CAD",
            years=T,
            iterations=I,
            data=data,
        )

    # ---- override equity cashflow to reflect PRIVATE investor view ----
    @computed_field
    @property
    def equity_cashflow(self) -> Output:
        """
        Private equity viewpoint:
          t=0 = -private_investment
          t>=1 = capital_returned + investor_dividends (both negative in project view -> invert sign here).
        FinancialModel.equity_irr will use this stream.
        """
        I, T = self.iterations, self.years
        data = np.zeros((I, T), dtype=float)
        data[:, 0] = -self.private_investment.data[:, 0]
        # Invert sign: inflows to equity holder should be positive
        data[:, 1:] = -(
            self.capital_returned.data[:, 1:] + self.investor_dividends.data[:, 1:]
        )
        return Output(
            scenario=self.scenario,
            label="equity_cashflow",
            description="Cashflows to private equity (t=0 negative, later positive)",
            units="$CAD",
            years=T,
            iterations=I,
            data=data,
        )

    # --------- iteration summary (external validation) ----------
    @computed_field
    @property
    def iteration_summary(self) -> PandasDataFrame:
        """
        Tabular per-iteration NPVs for key components, with a total.
        This mirrors the accepted external model summary for validation.
        """
        rates = self.discount_rate.data[:, 0] / 100.0

        def npv_rows(surface: np.ndarray) -> np.ndarray:
            # NPV starting at year 1 (Y0 is commissioning; conventional NPV often starts at Y1)
            T = surface.shape[1]
            periods = np.arange(1, T + 1, dtype=float)
            denom = (1.0 + rates[:, None]) ** periods[None, :]
            return np.sum(surface / denom, axis=1)

        rev_npv = npv_rows(self.revenue.data)
        opex_npv = npv_rows(self.opex.data)  # opex negative
        capex0 = -self.capex.data[:, 0]  # initial outflow (negative)
        fin_npv = npv_rows(self.finance_costs.data)  # negative
        total = rev_npv + opex_npv + capex0 + fin_npv

        df = pd.DataFrame(
            {
                "Revenue (NPV)": rev_npv,
                "Opex (NPV)": opex_npv,
                "Capex Y0": capex0,
                "Finance (NPV)": fin_npv,
                "TOTAL (NPV)": total,
            }
        )
        return df
