# models.py
from __future__ import annotations
from typing import Dict, List, Any, ClassVar, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from pydantic import BaseModel, Field, computed_field

from models.core.types import (
    ModelCalculation,
    ModelOutput,
    ModelInput,
    NdArray,
    PandasDataFrame,
)
from models.solar import SolarProject
from models.mixin import CommonModelMixin


class CommunityPortfolio(CommonModelMixin, BaseModel):

    model: SolarProject = SolarProject()
    total_public_funding: ModelInput
    max_projects_per_year: ModelInput

    metrics: ClassVar[List[str]] = [
        "opex",
        "retained_earnings",
        "revenue",
        "finance_costs",
        "production",
        "total_cash",
        "free_cash_flow",
        "system_output",
    ]

    def __init__(
        self,
        scenario: str = "Accelerator",
        years: int = 25,
        iterations: int = 1000,
        config_path: str = "inputs/Accelerator.yaml",
        **data: Any,
    ) -> None:
        for field_name in self.input_names:
            if field_name not in data:
                data[field_name] = ModelInput.from_config(
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
    def results(self) -> Tuple[Dict[str, NdArray], NdArray]:
        # 1) instantiate one template to pull all six metrics
        model = self.model
        metrics = CommunityPortfolio.metrics

        # each is ModelOutput.data: (n_solar, years) except public_investment: (n_solar, 1)
        public_inv = model.public_investment.data[:, 0]  # shape (n_solar,)
        solar_data = {m: getattr(model, m).data for m in metrics}
        n_solar = public_inv.shape[0]

        # 2) unpack accelerator inputs (shape (n_accel,1) → (n_accel,))
        n_accel = self.iterations
        total_fund = self.total_public_funding.data[:, 0]
        max_per_yr = self.max_projects_per_year.data[:, 0]

        # 3) preallocate
        #    counts of built projects
        built = np.zeros((n_accel, self.years), dtype=int)
        #    one 2D array per metric
        agg = {m: np.zeros((n_accel, self.years)) for m in metrics}
        #    track available funding
        avail = np.zeros((n_accel, self.years + 1))
        avail[:, 0] = total_fund

        # 4) simulate each accel-replicate
        for k in range(n_accel):
            for y in range(self.years):
                # sample up to max_per_yr[k] project runs (with or w/o replacement)
                picks = np.random.choice(n_solar, size=int(max_per_yr[k]), replace=True)

                for j in picks:
                    cost = public_inv[j]
                    if avail[k, y] >= cost:
                        # we can afford it → build it
                        avail[k, y] -= cost
                        built[k, y] += 1
                        # aggregate each metric for this project at year y
                        for m in metrics:
                            agg[m][k, y:] += solar_data[m][j, : self.years - y]
                    # else: skip—no capacity

                # after all attempts, roll last year’s free cash into next year
                # free_cash_flow is in agg["free_cash_flow"]
                avail[k, y + 1] = avail[k, y] + agg["free_cash_flow"][k, y]

        return agg, built

    @computed_field
    @property
    def number_of_projects(self) -> ModelOutput:
        _, built = self.results
        return ModelOutput(
            scenario=self.scenario,
            label="Projects Funded",
            description="Number of projects funded each year",
            units="#",
            data=built,
        )

    @computed_field
    @property
    def total_projects(self) -> ModelOutput:
        _, built = self.results
        return ModelOutput(
            scenario=self.scenario,
            label="Projects Funded",
            description="Number of projects funded each year",
            units="#",
            data=built.cumsum(axis=1),
        )

    @computed_field
    @property
    def opex(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.opex
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["opex"],
        )

    @computed_field
    @property
    def retained_earnings(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.retained_earnings
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["retained_earnings"],
        )

    @computed_field
    @property
    def revenue(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.revenue
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["revenue"],
        )

    @computed_field
    @property
    def finance_costs(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.finance_costs
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["finance_costs"],
        )

    @computed_field
    @property
    def total_cash(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.total_cash
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["total_cash"],
        )

    @computed_field
    @property
    def free_cash_flow(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.free_cash_flow
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["free_cash_flow"],
        )

    @computed_field
    @property
    def system_output(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.system_output
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["system_output"],
        )

    @computed_field
    @property
    def total_capacity(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model.system_output
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["system_output"].cumsum(axis=1),
        )

    def layout(self) -> None:

        tabs = st.tabs(["Model", "Portfolio"])

        with tabs[0]:

            self.model.layout()
            # self.portfolio.model = self.model

        with tabs[1]:

            self.portfolio_layout()

    def configure_portfolio(self) -> None:

        if f"{self.scenario}-input_selection_row" not in st.session_state:
            st.session_state[f"{self.scenario}-input_selection_row"] = 0

        col1, col2 = st.columns([0.50, 0.50])

        with col1:
            st.subheader("Assumptions")
        selection = col1.dataframe(
            self.assumptions, selection_mode="single-row", on_select="rerun"
        )["selection"]
        if selection["rows"]:
            st.session_state[f"{self.scenario}-input_selection_row"] = selection[
                "rows"
            ][0]

        with col2:
            st.subheader("Inputs")
            name = st.selectbox(
                "Select Input",
                options=self.input_names,
                index=(
                    selection["rows"][0]
                    if selection["rows"]
                    else st.session_state[f"{self.scenario}-input_selection_row"]
                ),
                key=f"{self.scenario}-select_config_input",
                label_visibility="collapsed",
            )
            if name:
                input: ModelInput = getattr(self, name)
                input.controls

    def portfolio_layout(self) -> None:
        """
        Render the portfolio layout in Streamlit.
        """
        st.subheader("Accelerator Portfolio Results")

        tabs = st.tabs(["Inputs", "Sensitivity", "Charts"])

        with tabs[0]:
            self.configure_portfolio()

        with tabs[1]:
            metric = st.selectbox(
                "Select Metric",
                options=self.output_names,
                key=f"{self.scenario}-select_sensitivity_output",
                index=None,
            )
            if metric:

                variables = self.assumptions[self.assumptions.is_fixed == False]
                if variables.empty:
                    st.info("No unfixed inputs available for sensitivity analysis.")
                else:
                    fig = self.sensitivity_plot(metric)
                    st.plotly_chart(fig)


        with tabs[2]:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Timeseries", "Surface", "Histogram"],
                index=1,
                key="accelerator-select_chart_type",
            )
            metric = st.selectbox(
                "Select Metric",
                self.output_names,
                key="accelerator-select_metric",
            )

            if metric:
                obj: ModelCalculation | ModelOutput = getattr(self, metric)
                if chart_type == "Timeseries":
                    fig = obj.timeseries_plot
                elif chart_type == "Surface":
                    col1, col2 = st.columns(2)
                    min_pct = col1.number_input(
                        "Min Percentile",
                        value=10,
                        min_value=1,
                        max_value=50,
                        key="accelerator-min",
                    )
                    max_pct = col2.number_input(
                        "Max Percentile",
                        value=90,
                        min_value=51,
                        max_value=100,
                        key="accelerator-max",
                    )
                    fig = obj.surface_plot(
                        min_percentile=min_pct,
                        max_percentile=max_pct,
                    )
                else:  # Histogram
                    fig = obj.hist_plot()

                st.plotly_chart(fig, use_container_width=True)

    @computed_field
    @property
    def npv(self) -> Dict[str, NdArray]:
        """
        Calculate the NPV across time of each dollar‐denominated output.
        Returns a dict: {metric_name: array of NPVs per iteration}.
        """
        # pull the per‐iteration discount rate from the underlying model
        rates = self.model.discount_rate.data[:, 0] / 100.0  # shape (n_iter,)

        npv_dict: Dict[str, NdArray] = {}
        for name in self.output_names:
            out: ModelOutput = getattr(self, name)
            # only dollar outputs
            if out.units and out.units.startswith("$"):
                # ModelOutput.npv will prepend a zero at t=0 and call npf.npv
                npv_dict[name] = out.npv(rates)
        return npv_dict


    