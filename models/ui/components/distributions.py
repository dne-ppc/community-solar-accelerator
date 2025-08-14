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

