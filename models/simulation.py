# models.py
from __future__ import annotations
from typing import Dict, List, Any, ClassVar, Tuple
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import yaml
import os

from pydantic import BaseModel, Field, computed_field

from models.base import (
    FinancialModel,
    ModelCalculation,
    ModelOutput,
    ModelInput,
    PandasDataFrame,
    NdArray,
)


class Portfolio(BaseModel):

    # capex: NdArray
    opex: ModelOutput
    finance_costs: ModelOutput
    retained_earnings: ModelOutput
    revenue: ModelOutput
    total_cash: ModelOutput
    free_cash_flow: ModelOutput
    projects_built: ModelOutput



class Accelerator(BaseModel):
    model_type: type
    scenario: str = "Accelerator"
    years: int
    iterations: int
    total_public_funding: ModelInput
    max_projects_per_year: ModelInput

    metrics: ClassVar[List[str]] = [
        "opex",
        "retained_earnings",
        "revenue",
        "finance_costs",
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
        model_type: type = FinancialModel,
        **data: Any,
    ) -> None:
        for field_name in self.inputs_names:
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
        data["model_type"] = model_type
        super().__init__(**data)

    @computed_field
    @property
    def inputs_names(cls) -> List[str]:
        """
        Return a list of all model input names.
        """
        return [
            field_name
            for field_name, field in cls.__annotations__.items()
            if field is 'ModelInput' or field is ModelInput
        ]

    @computed_field
    @property   
    def run(self) -> Tuple[Dict[str, NdArray], NdArray]:
        # 1) instantiate one template to pull all six metrics
        proj = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        )
        metrics = Accelerator.metrics

        # each is ModelOutput.data: (n_solar, years) except public_investment: (n_solar, 1)
        public_inv = proj.public_investment.data[:, 0]  # shape (n_solar,)
        solar_data = {m: getattr(proj, m).data for m in metrics}
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
                            agg[m][k, y:] += solar_data[m][j, :self.years-y]
                    # else: skip—no capacity

                # after all attempts, roll last year’s free cash into next year
                # free_cash_flow is in agg["free_cash_flow"]
                avail[k, y + 1] = avail[k, y] + agg["free_cash_flow"][k, y]

        return agg, built
    

    @cached_property
    def results(self)  -> Tuple[Dict[str, NdArray], NdArray]:
        """
        Run the accelerator simulation and return the results.
        Returns:
            Tuple of aggregated metrics and built projects.
        """
        return self.run
    
    @computed_field
    @property
    def n_built(self) -> ModelOutput:
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
    def opex(self) -> ModelOutput:
        agg, _ = self.results
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).opex
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).retained_earnings
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).revenue
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).finance_costs
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).total_cash
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).free_cash_flow
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
        src = self.model_type(
            scenario=self.scenario,
            years=self.years,
            iterations=self.iterations,
        ).system_output
        return ModelOutput(
            scenario=self.scenario,
            label=src.label.replace("project", "accelerator"),
            description=src.description,
            units=src.units,
            data=agg["system_output"],
        )
    
    



class Simulation(BaseModel):

    models: List[FinancialModel] = Field(default_factory=list)
    portfolio_schedule: List[Dict[str, Any]] = Field(default_factory=list)
    _model_type: type

    class Config:
        json_encoders = {
            NdArray: lambda arr: arr.tolist(),
            PandasDataFrame: lambda df: df.to_dict(orient="records"),
        }

    @computed_field
    @property
    def portfolio(self) -> Portfolio:
        metrics = Portfolio.metrics
        n_iter = None
        max_years = 0
        # Determine overall horizon
        for row in self.portfolio_schedule:
            idx = row["model_idx"]
            start_year = row["start_year"]
            fm = self.models[idx]
            n_years = getattr(fm, metrics[0]).data.shape[1]
            n_iter = getattr(fm, metrics[0]).data.shape[0]
            total_horizon = start_year + n_years
            max_years = max(max_years, total_horizon)

        # Aggregate each metric
        aggregated: Dict[str, ModelOutput] = {}
        for m in metrics:
            temp = np.zeros((n_iter, max_years))
            for row in self.portfolio_schedule:
                idx = row["model_idx"]
                start_year = row["start_year"]
                fm = self.models[idx]
                op = getattr(fm, m)
                _, yrs = op.data.shape
                temp[:, start_year : start_year + yrs] += op.data
            aggregated[m] = ModelOutput(
                scenario="portfolio",
                label=op.label,
                data=temp,
                units=op.units,
                description=op.description.replace("project", "portfolio"),
            )
        return Portfolio(**aggregated)

    def base_name(self) -> str:
        count = 0
        for name in self.model_names:
            if "Base" in name:
                count += 1
        name = "Base"
        if count:
            name += f" {count}"
        return name

    def model_idx(self, model_name: str) -> int:

        for idx, name in enumerate(self.model_names):
            if name == model_name:
                return idx

    @st.dialog("New Model Configuration")
    def new_model_dialog(self) -> None:
        col1, col2 = st.columns(2)

        years = col1.number_input(
            "Years", min_value=1, value=25, step=1, key="new_model_years"
        )
        iters = col2.number_input(
            "Iterations", min_value=1, value=1000, step=1, key="new_model_iters"
        )

        options = self.get_config_paths()
        config_path = st.selectbox(
            "Input Config File",
            options=options,
            index=options.index("inputs/Base.yaml"),
            key="model_type_select",
        )

        if st.button("Create Model", key="create_model"):
            fm = self._model_type(
                scenario=self.base_name(),
                years=int(years),
                iterations=int(iters),
                config_path=config_path,
            )
            self.models.append(fm)
            st.rerun()

    def configure_model(self):
        with st.expander("Manage Models", expanded=True):

            # Global settings apply to all new and existing models

            if st.button("Add New", key="add_new_model"):

                self.new_model_dialog()

            # Render pending scenario inputs
            for i, model in enumerate(self.models):
                col_name, col_save, col_del = st.columns(
                    [2, 1, 1], vertical_alignment="bottom"
                )

                model.scenario = col_name.text_input(
                    "Scenario name", model.scenario, key=f"pending_scenario_{i}"
                )
                if col_save.button(
                    "Save", key=f"pending_save_{i}", use_container_width=True
                ):
                    fname = f"{model.scenario}.yaml"
                    save_path = os.path.join("inputs", fname)
                    data = {}
                    exclude = [
                        "data",
                        "dist",
                        "dist_plot",
                        "hist_plot",
                        "controls",
                        "surface_plot",
                        "timeseries_plot",
                        "boundedness",
                        "iterations",
                    ]
                    for name, input in model.inputs_fields.items():
                        data[name] = input.model_dump(exclude=exclude)
                    with open(save_path, "w") as f:
                        yaml.safe_dump(data, f)

                if col_del.button(
                    "Remove", key=f"pending_remove_{i}", use_container_width=True
                ):
                    self.models.pop(i)
                    st.rerun()

    def get_config_paths(self) -> List[str]:

        return [str(path) for path in Path("inputs").glob("*.yaml")]

    def config_portfolio(self):
        with st.expander("Portfolio Configuration", expanded=True):

            # Add new empty row
            if st.button("Add project"):
                self.portfolio_schedule.append(
                    {"model": self.model_names[0], "start_year": 0}
                )

            # Render each row
            for idx, row in enumerate(self.portfolio_schedule):
                cols = st.columns([2, 1, 1], vertical_alignment="bottom")
                row["model"] = cols[0].selectbox(
                    "Model",
                    options=self.model_names,
                    index=self.model_names.index(row["model"]),
                    key=f"portfolio_model_{idx}",
                )
                row["start_year"] = cols[1].number_input(
                    "Start Year",
                    min_value=0,
                    value=row["start_year"],
                    key=f"portfolio_start_{idx}",
                )
                row["model_idx"] = self.model_idx(row["model"])
                if cols[2].button("Remove", key=f"portfolio_remove_{idx}"):
                    self.portfolio_schedule.pop(idx)
                    st.rerun()

    # @computed_field
    # @property
    def sidenav(self) -> None:
        """
        Streamlit sidebar controls for:
          1) Listing existing models
          2) Removing an existing model
          3) Adding a new model
        """
        with st.sidebar:
            self.configure_model()
            if self.models:
                self.config_portfolio()

    @computed_field
    @property
    def model_names(self) -> List[str]:
        return [model.scenario for model in self.models]

    def tabs(self) -> None:

        model_keys = self.model_names

        tabs = st.tabs(["Comparison", "Portfolio"] + model_keys)

        with tabs[0]:
            if self.models:

                self.compare_percentiles_with_errorbars()
            else:
                st.info("Add models simulation to compare results")

        with tabs[1]:

            timeseries, surface = st.tabs(["Timeseries", "Surface"])

            if self.portfolio_schedule:
                with timeseries:

                    metric = st.selectbox(
                        "Select Metric",
                        options=Portfolio.metrics,
                        key=f"portfolio-select_timeseries",
                        index=None,
                    )
                    if metric:
                        obj: ModelCalculation | ModelOutput = getattr(
                            self.portfolio, metric
                        )
                        fig = obj.timeseries_plot
                        st.plotly_chart(fig)
                with surface:

                    col1, col2, col3 = st.columns(3)

                    metric = col1.selectbox(
                        "Select Metric",
                        options=Portfolio.metrics,
                        key=f"portfolio-select_surface",
                        index=None,
                    )

                    min_percentile = col2.number_input(
                        "Select Minimum Percentile",
                        key=f"portfolio-select_min",
                        value=10,
                        min_value=1,
                        max_value=50,
                    )

                    max_percentile = col3.number_input(
                        "Select Maximum Percentile",
                        key=f"portfolio-select_max",
                        value=90,
                        min_value=51,
                        max_value=100,
                    )

                    if metric:
                        obj: ModelCalculation | ModelOutput = getattr(
                            self.portfolio, metric
                        )
                        fig = obj.surface_plot(
                            min_percentile=min_percentile, max_percentile=max_percentile
                        )
                        st.plotly_chart(fig)

            else:
                st.info("Add models and configure schedule to compare results")

        if len(tabs) == 2:
            return ""

        for i, tab in enumerate(tabs[2:]):

            with tab:
                self.models[i].layout()

        return ""

    def all_percentiles(self) -> pd.DataFrame:

        percentiles = []
        for model in self.models:
            df = model.kpi
            df["Model"] = model.scenario
            percentiles.append(df)

        return pd.concat(percentiles)

    def compare_percentiles_with_errorbars(self) -> None:
        """
        Build a multi‐subplot Plotly figure, with one subplot per metric. Each subplot
        shows P50 as a bar for each model, and asymmetric error bars from P10 to P90.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A vertical stack of subplots. Each subplot title is the metric name.
        """
        if not self.models:
            raise ValueError("No models available to compare.")

        df = self.all_percentiles()  # .reset_index()
        df.index.name = "Metric"
        df.reset_index(inplace=True)

        df["high"] = df.P90 - df.P50
        df["low"] = df.P50 - df.P10

        tabs = st.tabs(["Chart", "Table"])
        with tabs[0]:

            col1, col2 = st.columns(2)

            for i, (metric, metric_df) in enumerate(df.groupby("Metric")):

                fig = px.bar(
                    metric_df,
                    x="Model",
                    y="P50",
                    error_y="high",
                    error_y_minus="low",
                    title=metric,
                    # color="Model",
                )

                if i % 2 == 0:
                    col1.plotly_chart(fig)
                else:
                    col2.plotly_chart(fig)

        with tabs[1]:
            df.drop(columns=["high", "low"], inplace=True)
            df = df.replace(np.nan, "")
            st.dataframe(df.sort_values("Metric"))
