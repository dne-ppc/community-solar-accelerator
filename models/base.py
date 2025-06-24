# models.py
from __future__ import annotations
from typing import Dict, List, Optional, TypeVar, Any, Tuple, Literal, ClassVar

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pydantic import BaseModel, Field, computed_field

from metalog import metalog
import yaml
import numpy_financial as npf
import itertools

import inspect
import ast
from graphviz import Digraph
import textwrap


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")

np.float_ = np.float64


class ArrayOperatorMixin:
    """
    Provides element‐wise add, sub, mul, truediv for any class that has a 'data' attribute.
    Any subclass must have a `.data: np.ndarray` field.
    """

    def __add__(self, other: ArrayOperatorMixin | NdArray) -> NdArray:
        if isinstance(other, ArrayOperatorMixin):
            return self.data + other.data
        elif isinstance(other, np.ndarray):
            return self.data + other
        else:
            raise TypeError(
                f"Unsupported type for addition: {type(other)}. "
                "Must be another ArrayArithmeticMixin or NumPy array."
            )

    def __sub__(self, other: ArrayOperatorMixin | NdArray) -> NdArray:
        if isinstance(other, ArrayOperatorMixin):
            return self.data - other.data
        elif isinstance(other, np.ndarray):
            return self.data - other
        else:
            raise TypeError(
                f"Unsupported type for subtraction: {type(other)}. "
                "Must be another ArrayArithmeticMixin or NumPy array."
            )

    def __mul__(self, other: ArrayOperatorMixin | NdArray) -> NdArray:
        if isinstance(other, ArrayOperatorMixin):
            return self.data * other.data
        elif isinstance(other, np.ndarray):
            return self.data * other
        else:
            raise TypeError(
                f"Unsupported type for multiplication: {type(other)}. "
                "Must be another ArrayArithmeticMixin or NumPy array."
            )

    def __truediv__(self, other: ArrayOperatorMixin | NdArray | float | int) -> NdArray:
        if isinstance(other, ArrayOperatorMixin):
            return self.data / other.data
        elif isinstance(other, np.ndarray):
            return self.data / other
        elif isinstance(other, (float, int)):
            return self.data / other
        else:
            raise TypeError(
                f"Unsupported type for division: {type(other)}. "
                "Must be ArrayArithmeticMixin, NumPy array, float, or int."
            )

    def __array__(self, dtype=None) -> NdArray:
        """
        Allow np.asarray(model_output) → model_output.data
        """
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, idx):
        """
        Enable indexing: model_output[i] → model_output.data[i]
        """
        return self.data[idx]

    @computed_field
    @property
    def dist_plot(self) -> PlotlyFigure:
        """
        Creates a distribution plot (PDF + CDF) from a Metalog object.

        Args:
            m (str): A Metalog distribution dictionary returned by metalog.fit().

        Returns:
            go.Figure: A Plotly figure with two traces: PDF (left Y-axis) and CDF (right Y-axis).
        """

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=self.data.flatten(),
                name="Data",
                marker=dict(color="green"),
            )
        )

        fig.update_layout(
            xaxis=dict(title="Value"),
            yaxis=dict(
                title="Count",
                title_font_color="green",
                tickfont_color="green",
            ),
            legend=dict(x=0, y=1.1, orientation="h"),
            template="plotly",
            hovermode="x unified",
        )
        return fig

    @property
    def shape(self):
        return self.data.shape

    @computed_field
    @property
    def timeseries_plot(self) -> PlotlyFigure:

        years = np.arange(self.data.shape[1])
        p10_vals = np.percentile(self.data, 10, axis=0)
        p50_vals = np.percentile(self.data, 50, axis=0)
        p90_vals = np.percentile(self.data, 90, axis=0)

        fig = go.Figure()

        # Fill between P10 and P90
        fig.add_trace(
            go.Scatter(
                x=years + years[::-1],
                y=p90_vals + p10_vals[::-1],
                fill="toself",
                fillcolor="rgba(0, 100, 200, 0.5)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=p90_vals,
                mode="lines",
                line=dict(color="rgba(100, 0, 200, 1)", width=2),
                name=f"P90",
                showlegend=True,
            ),
        )
        # Plot P50 median line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=p50_vals,
                mode="lines",
                line=dict(color="rgba(0, 100, 200, 1)", width=2),
                name=f"P50",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=p10_vals,
                mode="lines",
                line=dict(color="rgba(200, 100, 0, 1)", width=2),
                name=f"P10",
                showlegend=True,
            )
        )

        fig.update_xaxes(
            title_text="Year",
        )
        fig.update_layout(
            height=800,
            hovermode="x unified",
            title_text="Simulator Forecast: Key Metrics P10/P50/P90",
            title_x=0.5,
        )
        return fig

    def surface_plot(
        self,
        percentiles=None,
        title=None,
        min_percentile=None,
        max_percentile=None,
    ) -> PlotlyFigure:
        """
        Create a 3D surface plot (plotly) of value by (time, percentile).

        Args:
            percentiles (array-like): List/array of percentiles to plot (0-100). Default is np.arange(0, 101, 10).
            title (str): Optional plot title.

        Returns:
            plotly.graph_objs._figure.Figure
        """
        # Default percentiles (every 10%)
        if percentiles is None:
            percentiles = np.arange(0, 101, 10)
        percentiles = np.array(percentiles)

        # Apply filtering if specified
        if min_percentile is not None:
            percentiles = percentiles[percentiles >= min_percentile]
        if max_percentile is not None:
            percentiles = percentiles[percentiles <= max_percentile]
        # Ensure at least one percentile remains
        if percentiles.size == 0:
            raise ValueError("No percentiles to plot after filtering.")

        data = self.data
        n_iter, n_years = data.shape

        # Compute percentile values at each year (z shape: (len(percentiles), n_years))
        z = np.percentile(data, percentiles, axis=0)

        # X: years, Y: percentiles
        x = np.arange(n_years)  # Time axis
        y = percentiles  # Percentile axis

        fig = go.Figure(
            data=[
                go.Surface(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    colorbar=dict(title="Value"),
                    showscale=True,
                )
            ]
        )

        fig.update_layout(
            title=title or f"Surface Plot: Value by Year and Percentile",
            scene=dict(
                xaxis_title="Year",
                yaxis_title="Percentile",
                zaxis_title=self.units,
                xaxis=dict(tickmode="linear"),
                yaxis=dict(tickmode="linear", dtick=10),
            ),
            height=800,
            margin=dict(l=10, r=10, b=40, t=40),
            scene_camera_eye=dict(x=-1, y=-1, z=0.5),
        )
        return fig


    def hist_plot(self,cumulative=True) -> PlotlyFigure:
        """
        Create a histogram plot of the data.
        """
        fig = px.histogram(
            x=self.data.flatten(),
            nbins=50,
            title=f"Histogram of {self.label}",
            labels={"x": self.label},
            # marginal="box",
            cumulative=cumulative,
            histnorm="probability",
        )
        fig.update_layout(
            xaxis_title=self.label,
            yaxis_title="Count",
            height=600,
            width=800,
        )
        return fig

    def npv(self, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0],1)), self.data],axis=1)
        return np.array(
            [npf.npv(rates[i], arr[i, :]) for i in range(self.shape[0])]
        )

    def npv_iter(self, i: int, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0],1)), self.data],axis=1)

        return np.array([npf.npv(rates[i], arr[i, :])])


class ModelCalculation(ArrayOperatorMixin, BaseModel):
    """
    Holds the intermediate outputs of a single scenario, with metadata and a NumPy array.
    """

    scenario: str
    label: str
    description: Optional[str] = None
    units: Optional[str] = None
    data: NdArray


class ModelOutput(ArrayOperatorMixin, BaseModel):
    """
    Holds the outputs of a single scenario, with metadata and a NumPy array.
    """

    scenario: str
    label: str
    description: Optional[str] = None
    units: Optional[str] = None
    data: NdArray


class ModelInput(ArrayOperatorMixin, BaseModel):
    """
    Stores the basic configuration for a single parameter's Metalog distribution.
    For example: Price of Gasoline ($/L), with min_value, max_value, p10, p50, p90, etc.
    """

    scenario: str = Field(default="default", description="Scenario name for this input")
    label: str
    min_value: float | int
    max_value: float | int
    p10: float | int
    p50: float | int
    p90: float | int
    step: float | int = Field(
        default=0.01,
        description="Step size for sliders in Streamlit UI",
    )
    boundedness: str = "b"  # 'b' for two-sided bounding in metalog

    years: int = 25
    iterations: int = 1000

    description: str | None = None
    units: str | None = None

    data: NdArray | None = None

    use_fixed: bool = Field(default=False, description="Override to a fixed constant?")
    fixed_value: bool | int | float = Field(
        default=False,
        description="If use_fixed is True, draw every sample from this value",
    )

    def __init__(
        self,
        sensitivity_value: Optional[float] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)

        if "data" not in data:
            self.update_data(
                sensitivity_value=sensitivity_value,
            )

    @classmethod
    def from_config(cls, scenario, name, config_path: str = "inputs/Base.yaml", **data):
        with open(config_path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        settings: Dict[str, Any] = cfg.get(name)
        settings["scenario"] = scenario
        if "years" in settings:
            data.pop("years", None)
        settings.update(data)
        return cls(**settings)

    def update_data(
        self,
        sensitivity_value: float | int | None = None,
    ):
        """
        Draws samples for `iterations` * `years`.
        Priority:
          1) If sensitivity override matches label → full constant array of sensitivity_value.
          2) Else if use_fixed=True and fixed_value is set → full constant array of fixed_value.
          3) Otherwise sample from Metalog as before.
        """
        shape = (self.iterations, self.years)

        # 1) Sensitivity‐analysis override
        if sensitivity_value is not None:
            self.data = np.full(shape, sensitivity_value, dtype=float)
            return

        # 2) Fixed‐value override
        if self.use_fixed:
            # Fallback if fixed_value is None: use median
            val = self.fixed_value if self.fixed_value is not None else self.p50
            self.data = np.full(shape, val, dtype=float)
            return

        # 3) Metalog sampling
        if not metalog:
            raise RuntimeError("metalog library not available")

        dist = metalog.fit(
            x=[self.p10, self.p50, self.p90],
            boundedness=self.boundedness,
            bounds=[self.min_value - self.step, self.max_value + self.step],
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )
        values: np.ndarray = metalog.r(dist, n=self.iterations * (self.years))

        self.data = values.reshape(shape).clip(self.min_value, self.max_value)

    @computed_field
    @property
    def controls(self) -> None:
        """
        Streamlit UI: lets the user toggle between a fixed value or editable
        P10/P50/P90 sliders.
        """
        container = st.container(border=True)
        container.write(f"### {self.label}")

        # Fixed‐value toggle
        fixed_key = f"{self.scenario}_{self.label}_use_fixed"
        self.use_fixed = container.checkbox(
            "Use fixed value", value=self.use_fixed, key=fixed_key
        )

        if self.use_fixed:
            # Ask for the constant to use
            val_key = f"{self.scenario}_{self.label}_fixed_value"
            default = self.fixed_value if self.fixed_value is not None else self.p50
            self.fixed_value = container.number_input(
                "Fixed value",
                value=default,
                step=self.step,
                key=val_key,
            )
        else:
            # Original P10/P50/P90 controls
            p_low_key = f"{self.scenario}_{self.label}_low"
            p_med_key = f"{self.scenario}_{self.label}_medium"
            p_high_key = f"{self.scenario}_{self.label}_high"
            col1, col2, col3 = container.columns(3)
            with col1:
                low_val = col1.number_input("P10", value=self.p10, key=p_low_key)
            with col2:
                med_val = col2.number_input("P50", value=self.p50, key=p_med_key)
            with col3:
                high_val = col3.number_input("P90", value=self.p90, key=p_high_key)

            self.p10 = low_val
            self.p50 = med_val
            self.p90 = high_val
            # Plotly preview (always show, even when fixed)
            try:
                # Build a tiny metalog using current p10/p50/p90

                if not bool(self.fixed_value):
                    return

                container.plotly_chart(
                    self.dist_plot,
                    use_container_width=True,
                    key=f"{self.scenario}_{self.label}_dist_plot",
                )
            except Exception as e:
                st.error(f"Error plotting distribution: {e}")
        if st.button("Update Data", key=f"{self.scenario}_{self.label}_update_data"):
            self.update_data()
            st.rerun()

    @computed_field
    @property
    def dist(self) -> dict:
        return metalog.fit(
            x=[self.p10, self.p50, self.p90],
            boundedness=self.boundedness,
            bounds=[self.min_value - self.step, self.max_value + self.step],
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )

    @computed_field
    @property
    def dist_plot(self) -> PlotlyFigure:
        """
        Creates a distribution plot (PDF + CDF) from a Metalog object.

        Args:
            m (str): A Metalog distribution dictionary returned by metalog.fit().

        Returns:
            go.Figure: A Plotly figure with two traces: PDF (left Y-axis) and CDF (right Y-axis).
        """

        quantiles = self.dist["M"].iloc[:, 1]
        pdf_values = self.dist["M"].iloc[:, 0]
        cdf_values = self.dist["M"]["y"]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=self.data.flatten(),
                name="Data",
                # histnorm="",
                marker=dict(color="green"),
                yaxis="y3",
                xbins=dict(
                    start=quantiles.iloc[0],
                    end=quantiles.iloc[-1],
                    size=quantiles.iloc[2] - quantiles.iloc[1],
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=pdf_values / sum(pdf_values),
                mode="lines",
                name="PDF",
                line=dict(color="blue"),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=cdf_values,
                mode="lines",
                name="CDF",
                line=dict(color="red", dash="dash"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            xaxis=dict(title="Value"),
            yaxis=dict(title="PDF", title_font_color="blue", tickfont_color="blue"),
            yaxis2=dict(
                title="CDF",
                title_font_color="red",
                tickfont_color="red",
                overlaying="y",
                side="right",
            ),
            yaxis3=dict(
                title="Count",
                title_font_color="green",
                tickfont_color="green",
                overlaying="y",
                side="right",
                anchor="free",
                autoshift=True,
            ),
            legend=dict(x=0, y=1.1, orientation="h"),
            template="plotly",
            hovermode="x unified",
        )
        return fig


class FinancialModel(BaseModel):

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
    def output_names(self) -> List[str]:
        """
        Return a list of all property names whose declared return type is ModelOutput.
        This avoids invoking each property, preventing potential recursion.
        """
        outputs: List[str] = []
        for name, attr in type(self).__dict__.items():
            if isinstance(attr, property):
                fget = attr.fget
                if fget and fget.__annotations__.get("return") is ModelOutput:
                    outputs.append(name)
        return outputs

    @computed_field
    @property
    def calc_names(self) -> List[str]:
        """
        Return a list of all property names whose declared return type is ModelCalculation.
        """
        calcs: List[str] = []
        for name, attr in type(self).__dict__.items():
            if isinstance(attr, property):
                fget = attr.fget
                if fget and fget.__annotations__.get("return") is ModelCalculation:
                    calcs.append(name)
        return calcs

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
    def assumptions(self) -> PandasDataFrame:
        """
        Return a DataFrame of model assumptions with P10, P50, and P90 values.
        This is useful for displaying the model inputs in a structured format.
        """
        return pd.DataFrame(
            {
                field_name: {
                    "p10": (
                        input_model.p10 if input_model.use_fixed is False else np.nan
                    ),
                    "p50": (
                        input_model.p50 if input_model.use_fixed is False else np.nan
                    ),
                    "p90": (
                        input_model.p90 if input_model.use_fixed is False else np.nan
                    ),
                    "is_fixed": input_model.use_fixed,
                    "fixed_value": (
                        input_model.fixed_value if input_model.use_fixed else np.nan
                    ),
                    "units": input_model.units,
                    "description": input_model.description,
                }
                for field_name, input_model in self.inputs_fields.items()
            }
        ).T

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
        inputs = set(self.inputs_names)  # e.g. {'cost', 'capacity', …}
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
            current_input = getattr(self, param)
            baseline_data = current_input.data

            current_input.update_data(
                sensitivity_value=low_val,
            )
            metric_low = np.percentile(self.npv[target_metric], percentile)
            current_input.update_data(
                sensitivity_value=high_val,
            )
            metric_high = np.percentile(self.npv[target_metric], percentile)

            current_input.data = baseline_data  # restore original samples
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

    def layout(self) -> None:

        tabs = st.tabs(
            [
                "Inputs",
                "Assumptions",
                "KPI",
                "Input Sensitivity",
                "Combination Sensitivity",
                "Charts",
                "Iteration Outputs",
            ]
        )

        with tabs[0]:
            value = st.selectbox(
                "Select Input",
                options=self.inputs_names,
                index=None,
                key=f"{self.scenario}-select_config_input",
            )
            if value:
                input: ModelInput = getattr(self, value)
                input.controls

        with tabs[1]:
            st.dataframe(self.assumptions, height=600)

        with tabs[2]:
            st.subheader("KPI")
            st.dataframe(self.kpi.round(3))
            st.dataframe(self.npv_summary.round(0))
        with tabs[3]:
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

        with tabs[4]:
            self.combo_sensitivity_tab()

        with tabs[5]:

            chart_type = st.selectbox(
                "Select Chart Type",
                options=["Timeseries", "Surface", "Histogram"],
                index=1,
                key=f"{self.scenario}-select_chart_type",
            )

            if chart_type == "Histogram":
                names = self.output_names + self.calc_names + self.inputs_names
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

                    response = st.radio("Cumulative",options=['Yes','No'],index=0)

                    if response == 'Yes':
                        cumulative = True
                    else:
                        cumulative = False

                    fig = obj.hist_plot(cumulative=cumulative)

                st.plotly_chart(fig)
        with tabs[6]:
            self.iteration_summary()
