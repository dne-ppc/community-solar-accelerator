# models.py
from __future__ import annotations
from typing import Dict, TypeVar, Any
import os
import yaml

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field


from models.types import Input, Calculation, Output
from models.graph import TensorGraph


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")

np.float_ = np.float64


class FinancialModel(TensorGraph, BaseModel):

    scenario: str
    years: int
    iterations: int

    discount_rate: Input

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

    @computed_field
    @property
    def years_array(self) -> NdArray:
        return np.arange(self.years - 1)
    
    @computed_field
    @property
    def operating_margin(self) -> Calculation:
        """
        Operating margin per year: revenue - opex.
        """
        return Calculation(
            scenario=self.scenario,
            label="operating_margin",
            description="Operating margin per year",
            units="$CAD",
            data=self.revenue - self.opex,
        )

    
    @computed_field
    @property
    def retained_earnings(self) -> Output:
        """
        Retained earnings each year: cumulative sum of cashflow.
        """

        return Output(
            scenario=self.scenario,
            label="Retained Earnings ($CAD)",
            description="Cumulative retained earnings",
            units="$CAD",
            data=self.operating_margin - self.finance_costs,
        )


    @computed_field
    @property
    def npv(self) -> Dict[str, NdArray]:
        """
        Calculate the Net Present Value (NPV) for each output property.
        Returns a dictionary where keys are output names and values are NPV arrays.
        The NPV is calculated using the discount rate provided in the model.
        """

        rates = self.discount_rate[:, 0] / 100

        data: Dict[str, NdArray] = {}

        for name in self.output_names:

            output = getattr(self, name)  # t ype: ModelOutput

            data[name] = output.npv(rates)

        return data

    def save_model(self) -> None:

        fname = f"{self.scenario}.yaml"
        save_path = os.path.join("inputs", fname)
        data = {}
        exclude = [
            "data",
            "dist",
            "boundedness",
            "iterations",
        ]
        for name, input in self.input_fields.items():
            data[name] = input.model_dump(exclude=exclude)
        with open(save_path, "w") as f:
            yaml.safe_dump(data, f)

    @computed_field
    @property
    def roi(self) -> Calculation:
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

        return Calculation(
            scenario=self.scenario,
            label="ROI (%)",
            description="Undiscounted return on investment: net cash at end divided by initial investment",
            units="unitless",
            data=roi_matrix,
        )
    

    @computed_field
    @property
    def simple_payback_period(self) -> Calculation:
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

        return Calculation(
            scenario=self.scenario,
            label="Payback Period",
            description="Undiscounted payback period in years (first year where cumulative cash ≥ 0)",
            units="year",
            data=payback,
        )
    
    @computed_field
    @property
    def project_irr(self) -> Calculation:
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

        return Calculation(
            scenario=self.scenario,
            label="Project IRR (%)",
            description="Internal Rate of Return for the overall project",
            units="%",
            data=irr_array,
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
    def simple_payback_year_percentiles(self) -> PandasDataFrame:
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
        raw = self.simple_payback_period.data.flatten()
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.nanpercentile(finite, [10, 50, 90])
        df = pd.DataFrame(
            {self.simple_payback_period.label: [p10, p50, p90]}, index=["P90", "P50", "P10"]
        ).replace(9999, np.nan)
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