from typing import Dict, List, Optional, TypeVar, Any, Tuple
from pydantic import BaseModel, computed_field
from models import ModelInput
import numpy as np
import pandas as pd
import numpy_financial as npf

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
np.float_ = np.float64

HOURS_PER_YEAR = 8766


class ProjectFinancialModel(BaseModel):
    panel_power: ModelInput  # W
    number_of_panels: ModelInput
    capacity_factor: ModelInput
    degradation_rate: ModelInput
    install_cost: ModelInput

    public_funding_percent: ModelInput
    funding_buffer_percent: ModelInput

    electricity_price: ModelInput
    inflation_rate: ModelInput
    discount_rate: ModelInput
    maintenance_rate: ModelInput
    admin_rate: ModelInput
    insurance_rate: ModelInput

    capital_return_rate: ModelInput
    dividend_start_year: int = 3
    capital_return_year: int = 5
    return_period: int = 10

    scenario: str
    years: int
    iterations: int

    def __init__(self, scenario: str, years: int, iterations: int, **data: Any) -> None:
        for field_name in self.inputs_names:
            if field_name not in data:
                data[field_name] = ModelInput.from_config(
                    scenario,
                    field_name,
                    years=years,
                    iterations=iterations,
                )
        data["scenario"] = scenario
        data["years"] = years
        data["iterations"] = iterations
        super().__init__(**data)

    @computed_field
    @property
    def inputs_names(self) -> List[str]:
        """
        Return a list of all model input names.
        """
        return [
            field_name
            for field_name, field in self.__annotations__.items()
            if field is ModelInput
        ]

    @computed_field
    @property
    def inputs_fields(self) -> Dict[str, ModelInput]:
        """
        Return a dictionary of all model inputs.
        """
        return {
            field_name: getattr(self, field_name)
            for field_name, field in self.__annotations__.items()
            if field is ModelInput
        }

    @computed_field
    @property
    def model_assumptions(self) -> PandasDataFrame:
        """
        Return a DataFrame of model assumptions with P10, P50, and P90 values.
        This is useful for displaying the model inputs in a structured format.
        """
        return pd.DataFrame(
            {
                field_name: {
                    "p10": input_model.p10,
                    "p50": input_model.p50,
                    "p90": input_model.p90,
                    "units": input_model.units,
                    "description": input_model.description,
                }
                for field_name, input_model in self.inputs_fields.items()
            }
        ).T

    @computed_field
    @property
    def years_array(self) -> NdArray:
        return np.arange(self.years)

    @computed_field
    @property
    def system_output(self) -> NdArray:
        """
        Calculate the total system output in kWh per year.
        This is based on the panel power, number of panels, capacity factor, and hours per year.
        """
        # kWh per year
        # panel_power (W) * number_of_panels (#) * capacity_factor (%) * HOURS_PER_YEAR (h)
        # Convert to kWh by dividing by 1000
        return (
            self.panel_power  # W
            * self.number_of_panels  # #
            * self.capacity_factor
            / 100  # %
            * HOURS_PER_YEAR
        ) / 1000  # kWh per year

    @computed_field
    @property
    def capex(self) -> NdArray:
        """
        Calculate the capital expenditure (CAPEX) for the solar system.
        This is based on the number of panels, panel power, and installation cost.
        """
        return self.number_of_panels * self.panel_power * self.install_cost

    @computed_field
    @property
    def seed_capital(self) -> NdArray:
        """
        Calculate the seed capital required for the solar project.
        This is the initial investment needed to start the project.
        """
        return self.capex * (1 + self.funding_buffer_percent / 100)

    @computed_field
    @property
    def public_investment(self) -> NdArray:
        """
        Calculate the public investment based on the public funding percentage and the capital expenditure (CAPEX).
        This is the amount of public funding allocated to the solar project.
        """
        return self.seed_capital * self.public_funding_percent / 100

    @computed_field
    @property
    def private_investment(self) -> NdArray:
        """
        Calculate the private investment based on the total CAPEX and public investment.
        This is the amount of private funding allocated to the solar project.
        """
        return self.seed_capital * (1 - self.public_funding_percent / 100)

    @computed_field
    @property
    def degradation_schedule(self) -> NdArray:
        """
        Calculate the degradation schedule for the solar system.
        This is based on the degradation rate and the number of years.
        The degradation schedule is a factor that reduces the system output each year.
        """
        return (1 - self.degradation_rate / 100) ** self.years_array

    @computed_field
    @property
    def production(self) -> NdArray:
        """
        Calculate the annual production of the solar system.
        This is the system output adjusted for degradation.
        """
        return self.degradation_schedule * self.system_output

    @computed_field
    @property
    def escalation(self) -> NdArray:
        """
        Calculate the inflation schedule for the financial model.
        This is a factor that adjusts costs and revenues for inflation over the years.
        It is calculated as (1 + inflation_rate / 100) raised to the power of the number of years.
        """
        return (1 + self.inflation_rate / 100) ** self.years_array

    @computed_field
    @property
    def maintenance_cost(self) -> NdArray:
        """
        Calculate the annual maintenance cost for the solar system.
        This is based on the maintenance rate and the capital expenditure (CAPEX).
        The maintenance cost is adjusted for inflation.
        """
        return (self.maintenance_rate / 100) * self.escalation * self.capex

    @computed_field
    @property
    def admin_cost(self) -> NdArray:
        return (self.admin_rate / 100) * self.escalation * self.capex

    @computed_field
    @property
    def insurance_cost(self) -> NdArray:
        return (self.insurance_rate / 100) * self.escalation * self.capex

    @computed_field
    @property
    def opex(self) -> NdArray:
        return self.maintenance_cost + self.admin_cost + self.insurance_cost

    @computed_field
    @property
    def revenue(self) -> NdArray:
        return self.production * self.electricity_price  # * self.inflation_schedule

    @computed_field
    @property
    def operating_margin(self) -> NdArray:
        """
        Calculate the operating margin, which is the difference between revenue and operating expenses (OPEX).
        This is a measure of the profitability of the solar system.
        """
        return self.revenue - self.opex

    @computed_field
    @property
    def cashflow(self) -> NdArray:
        annual_cashflow = self.revenue - self.opex
        annual_cashflow[:, 0:1] -= self.capex
        return annual_cashflow

    @computed_field
    @property
    def capital_returned(self) -> NdArray:
        """
        Calculate the capital returned to investors.
        This is the amount of private investment that has been returned to investors.
        """

        repayment_amount = self.private_investment / (
            self.return_period - self.capital_return_year
        )
        payments = np.zeros((self.iterations, self.years))

        for year in range(0, self.years):
            if year >= self.capital_return_year - 1 and year < self.return_period - 1:
                payments[:, year] = repayment_amount[:, 0]
        return payments

    @computed_field
    @property
    def remaining_private_investment(self) -> NdArray:
        """
        Calculate the remaining private investment after accounting for dividends.
        This is a measure of the private investment that is still in the solar system.
        """
        remaining = np.zeros((self.iterations, self.years))
        current_investment = self.private_investment[:, 0]

        for year in range(0, self.years):
            current_investment: np.ndarray = (
                current_investment - self.capital_returned[:, year]
            )
            current_investment[np.abs(current_investment) < 1e-9] = 0
            remaining[:, year] = current_investment

        return remaining

    @computed_field
    @property
    def investor_dividends(self) -> NdArray:
        """
        Calculate the dividends paid to investors.
        This is based on the remaining private investment and the capital return rate.
        Dividends are paid starting from the dividend start year.
        """
        dividends = np.zeros((self.iterations, self.years))
        investment = self.remaining_private_investment
        rates = self.capital_return_rate[:, 0] / 100

        for year in range(1, self.years):
            if year >= self.dividend_start_year - 1:
                dividends[:, year] = investment[:, year - 1] * rates

        return dividends

    @computed_field
    @property
    def finance_costs(self) -> NdArray:
        """
        Calculate the financial costs for the solar system.
        This is the sum of the capital return and investor dividends.
        """
        return self.capital_returned + self.investor_dividends

    @computed_field
    @property
    def retained_earnings(self) -> NdArray:
        """
        Calculate the retained earnings for the solar system.
        This is the operating margin minus the dividends paid to investors.
        Retained earnings are accumulated over the years.
        """
        retained = np.zeros((self.iterations, self.years))
        for year in range(0, self.years):

            retained[:, year] = (
                self.operating_margin[:, year] - self.finance_costs[:, year]
            )

        return retained

    @computed_field
    @property
    def revenue_npv(self) -> NdArray:

        rates = self.discount_rate / 100
        return np.array([npf.npv(rate, cf) for rate, cf in zip(rates, self.revenue)])

    @computed_field
    @property
    def retained_earnings_npv(self) -> NdArray:
        """
        Calculate the NPV of retained earnings.
        This is the net present value of the retained earnings over the years.
        """
        rates = self.discount_rate / 100
        return np.array(
            [npf.npv(rate, cf) for rate, cf in zip(rates, self.retained_earnings)]
        )

    @computed_field
    @property
    def npv(self) -> NdArray:

        rates = self.discount_rate / 100
        return np.array([npf.npv(rate, cf) for rate, cf in zip(rates, self.cashflow)])

    @computed_field
    @property
    def irr(self) -> NdArray:
        return np.array([npf.irr(returns) for returns in self.cashflow - self.capex])
