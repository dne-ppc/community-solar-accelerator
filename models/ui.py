from __future__ import annotations
from typing import TypeVar

import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pydantic import  computed_field

import numpy_financial as npf


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")

np.float_ = np.float64


class TensorUI:

    
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
                yaxis=dict(tickmode="linear", dtick=5),
            ),
            height=800,
            margin=dict(l=10, r=10, b=40, t=40),
            scene_camera_eye=dict(x=-1, y=-1, z=0.5),
        )
        return fig

    def hist_plot(self, cumulative=True) -> PlotlyFigure:
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


    

class DistributionUI:
    """
    UI class for distribution-related visualizations.
    """
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
            # st.rerun()

