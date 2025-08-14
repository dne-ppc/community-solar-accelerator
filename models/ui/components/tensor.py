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

    def npv(self, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0], 1)), self.data], axis=1)
        return np.array([npf.npv(rates[i], arr[i, :]) for i in range(self.shape[0])])

    def npv_iter(self, i: int, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0], 1)), self.data], axis=1)

        return np.array([npf.npv(rates[i], arr[i, :])])