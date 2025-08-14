# models.py
from __future__ import annotations
from typing import Dict, List, Optional, TypeVar, Any, Tuple, Literal, ClassVar
import os
import yaml

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pydantic import BaseModel, Field, computed_field
from contextlib import contextmanager


import itertools

import inspect
import ast
from graphviz import Digraph
import textwrap

from models.core.types import ModelCalculation, ModelOutput, ModelInput
from models.mixin import CommonModelMixin
from utils import get_config_paths

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")

np.float_ = np.float64


@contextmanager
def patched_inputs(model, **overrides):
    saved = {k: getattr(model, k).data.copy() for k in overrides}
    try:
        for k, v in overrides.items():
            getattr(model, k).update_data(sensitivity_value=v)
        yield
    finally:
        for k, data in saved.items():
            getattr(model, k).data = data


class FinancialModel(CommonModelMixin, BaseModel):

    scenario: str
    years: int
    iterations: int

    discount_rate: ModelInput

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
    def years_array(self) -> NdArray:
        return np.arange(self.years - 1)

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

    @computed_field
    @property
    def npv_summary(self) -> PandasDataFrame:
        """
        Return a DataFrame summarizing the NPV of each output property.
        The DataFrame has columns for each output name and rows for percentiles (10th, 50th, 90th).
        """
        summary_data = {}
        for name, values in self.npv.items():
            # Convert each output's NPV array to a DataFrame with percentiles as rows
            summary_data[name] = np.percentile(values, [10, 50, 90])

        df = pd.DataFrame(summary_data, index=["P10", "P50", "P90"])
        df.index.name = "Metric"
        return df.T

    @computed_field
    @property
    def kpi(self) -> PandasDataFrame:
        """
        Gather every property on this model whose name ends with "_percentiles",
        call it, and stack the results into a single DataFrame.

        Each individual `<something>_percentiles` is assumed to return a 1×3 DataFrame
        whose index is [P10, P50, P90] (transposed, so after .T its index is the metric name).
        This method simply concatenates them so that each row is one metric and the columns
        are ["P10","P50","P90"].

        Example output:

                            P10     P50     P90
        cash_depletion_year  5.20    8.00   12.40
        private_investor_irr 2.15    7.80   15.30
        project_irr          1.75    5.50   11.10
        """
        dfs: list[pd.DataFrame] = []

        # Loop over all attributes; pick those whose name ends with "_percentiles".
        for attr_name in dir(self):
            if not attr_name.endswith("_percentiles"):
                continue

            # Avoid recursively calling this property itself
            if attr_name == "all_percentiles":
                continue

            candidate = getattr(self, attr_name, None)
            # Only keep it if it returned a DataFrame
            if isinstance(candidate, pd.DataFrame):
                dfs.append(candidate)

        if len(dfs) == 0:
            # If no _percentiles properties exist, return empty DataFrame
            return pd.DataFrame(columns=["P10", "P50", "P90"])

        # Concatenate so that each DataFrame’s index (its single metric) becomes a row in the final table
        result = pd.concat(dfs, axis=0)

        # Optional: sort by index (i.e. metric name)
        result = result.sort_index()

        for col in ["P10", "P50", "P90"]:
            result[col] = result[col].astype(float)

        return result

    def trace_calculation_graph(self, filename: str = None) -> Digraph:
        """
        Build a Graphviz graph of how each computed output depends on inputs
        and other outputs. If `filename` is provided, the diagram is rendered
        to that file (e.g. 'solar_dependencies.png' or 'solar_dependencies.pdf').
        Returns the `graphviz.Digraph` object.
        """
        # 1. Collect sets of names
        inputs = set(self.input_names)  # e.g. {'cost', 'capacity', …}
        calcs = set(self.calc_names)  # e.g. {'annual_generation', …}
        outputs = set(self.output_names)  # e.g. {'npv', 'lifecycle_cost', …}

        # 2. Create a left‐to‐right Digraph
        dot = Digraph(name="ModelGraph", format="png")
        dot.attr(rankdir="LR", fontsize="10")

        # 3. Add nodes for inputs (gray), calculations (lightblue), and outputs (white)
        for name in sorted(inputs):
            dot.node(name, shape="box", style="filled", fillcolor="lightgray")
        for name in sorted(calcs):
            dot.node(name, shape="ellipse", style="filled", fillcolor="lightblue")
        for name in sorted(outputs):
            dot.node(name, shape="ellipse", style="filled", fillcolor="white")

        # 4. Helper to scan a property`s source for dependencies (self.X)
        def _extract_dependencies(prop_name: str) -> set[str]:
            func = getattr(type(self), prop_name).fget
            src = inspect.getsource(func)
            tree = ast.parse(textwrap.dedent(src))
            deps = set()
            for node in ast.walk(tree):
                # Look for attribute access of the form self.<name>
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    attr = node.attr
                    if attr in inputs or attr in calcs or attr in outputs:
                        deps.add(attr)
            return deps

        # 5. Draw edges
        for calc_name in calcs:
            for dep in _extract_dependencies(calc_name):
                dot.edge(dep, calc_name)

        for out_name in outputs:
            for dep in _extract_dependencies(out_name):
                dot.edge(dep, out_name)

        # 6. Render to file if requested
        if filename:
            dot.render(filename, cleanup=True)

        return dot

    def sensitivity_analysis(
        self, target_metric: str, percentile: int = 50
    ) -> pd.DataFrame:
        """
        Compute sensitivity of `target_metric` at a given 1-based `year`
        to each input assumption that is not fixed. Returns a DataFrame
        with columns: ['Parameter','LowValue','HighValue','MetricLow','MetricHigh','Baseline','Range'].

        For each input parameter:
          - `LowValue` is its p10,
          - `HighValue` is its p90,
          - `MetricLow` is the 50th-percentile of target_metric (at `year`)
             when that parameter is held constant at p10 (and all other inputs unchanged),
          - `MetricHigh` is the 50th-percentile when that parameter is held constant at p90,
          - `Baseline` is the 50th-percentile of the current (unshocked) model,
          - `Range` = abs(MetricHigh – MetricLow).

        This method creates two new FinancialModel instances for each non-fixed input:
        one with the input`s samples forced to p10, one to p90. It does NOT mutate
        the original (self) model. It then reads off the median value of
        `target_metric` at the given year for each shock.
        """

        # 1) Baseline median for target_metric
        metric_baseline = np.percentile(self.npv[target_metric], percentile)

        records = []

        variables = self.assumptions[self.assumptions.is_fixed == False]
        # 2) Loop over each assumption (rows of the DataFrame)

        for param, row in variables.iterrows():

            low_val = row["p10"]
            high_val = row["p90"]

            with patched_inputs(self, **{param: low_val}):
                metric_low = np.percentile(self.npv[target_metric], percentile)
            with patched_inputs(self, **{param: high_val}):
                metric_high = np.percentile(self.npv[target_metric], percentile)
            # 5) Record everything
            records.append(
                {
                    "Parameter": param,
                    "LowValue": low_val,
                    "HighValue": high_val,
                    "MetricLow": metric_low,
                    "MetricHigh": metric_high,
                    "BaselineValue": row["p50"],
                    "Baseline": metric_baseline,
                    "Range": abs(metric_high - metric_low),
                }
            )

        return (
            pd.DataFrame(records)
            .sort_values("Range", ascending=True)
            .reset_index(drop=True)
        )

    def sensitivity_plot(
        self, target_metric: str, percentile: int = 50
    ) -> PlotlyFigure:

        fig = go.Figure()

        df = self.sensitivity_analysis(target_metric, percentile)

        # Add traces
        for idx, (_, r) in enumerate(df.iterrows()):
            param = r["Parameter"]
            m_low = r["MetricLow"]
            base = r["Baseline"]
            m_high = r["MetricHigh"]

            # Low to base (red)
            fig.add_trace(
                go.Scatter(
                    x=[m_low, base],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="orange", width=10),
                    showlegend=True if idx == 0 else False,
                    name=f"P10",
                ),
            )
            # Base to high (green)
            fig.add_trace(
                go.Scatter(
                    x=[base, m_high],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="blue", width=10),
                    showlegend=True if idx == 0 else False,
                    name=f"P90",
                ),
            )
            # Annotate parameter bounds
            fig.add_annotation(
                x=m_low,
                y=param,
                text=f"{r['LowValue']}\t",
                xanchor="right",
                yanchor="middle",
                showarrow=False,
            )
            fig.add_annotation(
                x=m_high,
                y=param,
                text=f"\t{r['HighValue']}",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
            )
            # Baseline marker
            fig.add_trace(
                go.Scatter(
                    x=[base],
                    y=[param],
                    mode="markers",
                    marker=dict(color="black", symbol="circle", size=8),
                    showlegend=False,
                ),
            )

            fig.add_annotation(
                x=base,
                y=param,
                text=f"\t{r['BaselineValue']}",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
            )

        # Layout tweaks
        fig.update_layout(
            height=400 + 20 * len(df),
            width=800,
            # xaxis_title=target_metric,
            # yaxis=dict(title="Parameter", automargin=True),
            title_x=0.5,
            margin=dict(l=200),
            title=f"Sensitivity Analysis: {getattr(self,target_metric).label}",
        )
        return fig

    def sensitivity_analysis_combo(
        self,
        target_metric: str,
        percentiles: list = [10, 90],
        top_n: int = 10,
        metric_percentile=50,
        n=3,
    ) -> pd.DataFrame:
        """
        For every combination of 3 non-fixed inputs, evaluate all combos of pX for those three.
        Returns a DataFrame with each row = (triplet, setting, metric value at that setting).
        Also returns the top N triplets by max-min range across the grid.
        """
        ots = self.sensitivity_analysis(target_metric, metric_percentile)
        variables = self.assumptions[self.assumptions.index.isin(ots.Parameter)]
        names = list(variables.index)
        if len(names) < 3:
            raise ValueError("Fewer than three unfixed inputs available.")

        base = np.percentile(self.npv["retained_earnings"], 50)

        percentile_names = [f"p{p}" for p in percentiles]
        percentile_combos = list(itertools.product(percentile_names, repeat=n))
        names = list(variables.index)
        input_combos = list(itertools.combinations(names, n))
        combos = list(itertools.product(percentile_combos, input_combos))

        dfs = []

        summaries = []
        for percentiles, inputs in combos:

            saved = {k: getattr(self, k).data.copy() for k in inputs}
            labels = []

            rows = []

            for percentile, input in zip(percentiles, inputs):
                v = variables.loc[input, percentile]
                getattr(self, input).update_data(sensitivity_value=v)

                if percentile == "p10":
                    col = "MetricLow"
                else:
                    col = "MetricHigh"

                label = f"{input}_{percentile}"
                labels.append(label)
                rows.append(
                    {
                        "percentile": percentile,
                        "input": input,
                        "individual_value": ots.loc[
                            ots.Parameter == input, col
                        ].squeeze()
                        - base,
                    }
                )

            df = pd.DataFrame(rows)
            combo = ",".join(labels)
            df["combo"] = combo
            df["total_value"] = np.percentile(self.npv["retained_earnings"], 50) - base
            dfs.append(df)
            summaries.append(
                {
                    "combo": combo,
                    "total_value": df.loc[0, "total_value"].squeeze(),
                }
            )
            for k, v in saved.items():
                getattr(self, k).data = v
        df = pd.concat(dfs, axis=0)

        df.sort_values(
            [
                "total_value",
                "combo",
            ],
            ascending=False,
            inplace=True,
        )
        mask = (df.combo != df.combo.shift(periods=1)).cumsum() < top_n
        df = df[mask].reset_index(drop=True)
        summary_df = pd.DataFrame(summaries)
        summary_df.sort_values("total_value", ascending=False, inplace=True)
        summary_df = summary_df.head(top_n)

        return df, summary_df

    def combo_sensitivity_tab(self):

        st.header("Combo Sensitivity Analysis")

        col1, col2, col3 = st.columns(3)

        metric = col1.selectbox(
            "Select Output Metric",
            options=self.output_names,
            format_func=lambda x: (
                getattr(self, x).label if hasattr(getattr(self, x), "label") else x
            ),
            index=0,
            key=f"{self.scenario}_combos_metric_select",
        )

        top_n = col2.number_input(
            "Show Top N Combinations",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            key=f"{self.scenario}_combo_top_n",
        )

        n = col3.number_input(
            "Number of Metrics to Combine",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key=f"{self.scenario}_combo_n_metrics",
        )

        if st.button("Run Analysis", key=f"{self.scenario}_combo_run"):
            with st.spinner("Running analysis..."):
                _, summary_df = self.sensitivity_analysis_combo(
                    metric, top_n=top_n, n=n
                )
                fig = px.bar(summary_df, x="combo", y="total_value", height=700)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"{self.scenario}_combo_metric_plot",
                )

    def save_model(self) -> None:

        fname = f"{self.scenario}.yaml"
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
        for name, input in self.input_fields.items():
            data[name] = input.model_dump(exclude=exclude)
        with open(save_path, "w") as f:
            yaml.safe_dump(data, f)

    def configure_model(self) -> None:

        if f"{self.scenario}-input_selection_row" not in st.session_state:
            st.session_state[f"{self.scenario}-input_selection_row"] = 0

        st.subheader("Configure Financial Model")
        cols = st.columns(
            [
                0.10,
                0.10,
                0.30,
                0.10,
                0.30,
                0.10,
            ],
            vertical_alignment="bottom",
        )
        self.years = cols[0].number_input(
            "Years",
            min_value=1,
            value=self.years,
            step=1,
            key="model_years",
        )
        self.iterations = cols[1].number_input(
            "Iterations",
            min_value=1,
            value=self.iterations,
            step=1,
            key="model_iters",
        )

        options = get_config_paths()
        config_path = cols[2].selectbox(
            "Input Config File",
            options=options,
            index=options.index("inputs/Base.yaml"),
            key=f"{self.scenario}_model_config_path",
        )
        if cols[3].button("Load Model", key="load_model", use_container_width=True):
            self.update_inputs(config_path)

        self.scenario = cols[4].text_input(
            "Scenario Name",
            value=self.scenario,
            key="model_scenario",
        )

        if cols[5].button(
            "Save", key=f"{self.scenario}_save_config", use_container_width=True
        ):
            self.save_model()

        col1, col2 = st.columns([0.50, 0.50])

        with col1:
            st.subheader("Assumptions")
        selection = col1.dataframe(
            self.assumptions, height=800, selection_mode="single-row", on_select="rerun"
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

    def sensitivity_tab(self) -> None:

        sensitivity_type = st.selectbox(
            "Select Sensitivity Type",
            options=["Single Metric", "Combo Sensitivity"],
            index=0,
            key=f"{self.scenario}-select_sensitivity_type",
        )
        if sensitivity_type == "Single Metric":
            self.single_sensitivity_tab()
        elif sensitivity_type == "Combo Sensitivity":
            self.combo_sensitivity_tab()

    def single_sensitivity_tab(self) -> None:

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

    def charts_tab(self) -> None:

        chart_type = st.selectbox(
            "Select Chart Type",
            options=["Timeseries", "Surface", "Histogram"],
            index=1,
            key=f"{self.scenario}-select_chart_type",
        )

        if chart_type == "Histogram":
            names = self.output_names + self.calc_names + self.input_names
        else:
            names = self.output_names + self.calc_names

        metric = st.selectbox(
            "Select Metric",
            options=names,
            key=f"{self.scenario}-select_metric",
            index=None,
        )
        if metric:

            obj: ModelCalculation | ModelOutput | ModelInput = getattr(self, metric)

            if chart_type == "Timeseries":
                fig = obj.timeseries_plot

            elif chart_type == "Surface":

                col1, col2 = st.columns(2)
                min_percentile = col1.number_input(
                    "Select Minimum Percentile",
                    key=f"{self.scenario}-select_min",
                    value=10,
                    min_value=1,
                    max_value=50,
                )

                max_percentile = col2.number_input(
                    "Select Maximum Percentile",
                    key=f"{self.scenario}-select_max",
                    value=90,
                    min_value=51,
                    max_value=100,
                )
                fig = obj.surface_plot(
                    min_percentile=min_percentile,
                    max_percentile=max_percentile,
                )

            elif chart_type == "Histogram":

                response = st.radio("Cumulative", options=["Yes", "No"], index=0)

                if response == "Yes":
                    cumulative = True
                else:
                    cumulative = False

                fig = obj.hist_plot(cumulative=cumulative)

            st.plotly_chart(fig)

    def calculation_graph_tab(self) -> None:

        st.subheader("Calculation Graph")
        filename = f"graphs/{self.scenario}_dependencies"
        dot = self.trace_calculation_graph(filename)
        st.graphviz_chart(dot.source, use_container_width=True)
        st.download_button(
            "Download Graph",
            data=dot.pipe(format="png"),
            file_name=f"{self.scenario}_dependencies.png",
            mime="image/png",
        )

    def layout(self) -> None:

        tabs = st.tabs(
            [
                "Model Configuration",
                "KPI",
                "Sensitivity",
                "Charts",
                "Iteration Outputs",
            ]
        )

        with tabs[0]:
            self.configure_model()

        with tabs[1]:
            st.subheader("KPI")
            st.dataframe(self.kpi.round(3))
            st.dataframe(self.npv_summary.round(0))

        with tabs[2]:
            self.sensitivity_tab()

        with tabs[3]:

            self.charts_tab()
        with tabs[4]:
            self.iteration_summary()

    def iteration_summary(self) -> None:
        raise NotImplementedError(
            "Iteration summary is not implemented yet. This method should be overridden in a subclass."
        )
