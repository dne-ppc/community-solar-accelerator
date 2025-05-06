import math
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import Simulation

import pandas as pd
import numpy as np
import numpy_financial as npf   

from typing import List

def plot_summary_grid(simulation: Simulation, cols: int = 2) -> None:
    """
    Plot all key financial metrics from a Simulator instance in a grid of subplots.
    Each metric displays P10-P90 as a filled area and P50 as a line over the 25-year horizon.

    Args:
        simulation: An instance of Simulation with `summary` populated (via monte_carlo_forecast()).
        cols: Number of columns in the subplot grid.
    """
    # Ensure forecast has been run
    if not simulation.summary:
        simulation.monte_carlo_forecast()
    summary = simulation.summary

    metrics = list(simulation.summary.keys())
    n_metrics = len(metrics)
    rows = math.ceil(n_metrics / cols)

    # Create subplots grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=metrics,
        shared_xaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    # Add each metric plot
    for idx, metric in enumerate(metrics):
        row = idx // cols + 1
        col = idx % cols + 1
        entry = summary[metric]

        # Convert dict of percentiles to DataFrame if needed
        if isinstance(entry, dict):
            p10 = entry.get("p10") or entry.get("P10")
            p50 = entry.get("p50") or entry.get("P50")
            p90 = entry.get("p90") or entry.get("P90")
            years = list(range(0, len(p50) ))
            df = pd.DataFrame({"P10": p10, "P50": p50, "P90": p90}, index=years)
        else:
            df = entry.copy()

        years = df.index.tolist()
        p10_vals = df["P10"].tolist()
        p50_vals = df["P50"].tolist()
        p90_vals = df["P90"].tolist()

        # Fill between P10 and P90
        fig.add_trace(
            go.Scatter(
                x=years + years[::-1],
                y=p90_vals + p10_vals[::-1],
                fill="toself",
                fillcolor="rgba(0, 100, 200, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=p90_vals,
                mode="lines",
                line=dict(color="rgba(100, 0, 200, 1)", width=2),
                name=f"P90",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )
        # Plot P50 median line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=p50_vals,
                mode="lines",
                line=dict(color="rgba(0, 100, 200, 1)", width=2),
                name=f"P50",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=p10_vals,
                mode="lines",
                line=dict(color="rgba(200, 100, 0, 1)", width=2),
                name=f"P10",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )



        # Axis labels
        fig.update_xaxes(title_text="Year", row=row, col=col)
        # fig.update_yaxes(title_text=metric, row=row, col=col)

    fig.update_layout(
        height=rows * 350,
        # height=2000,
        # width=cols * 600,
        hovermode="x unified",
        title_text="Simulator Forecast: Key Metrics P10/P50/P90",
        title_x=0.5,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_sensitivity_tornado(
    simulation: Simulation, target_metric: str, year: int
) -> None:
    """
    Plot a tornado chart showing the sensitivity of `target_metric` in `year`
    to each distribution input, using Simulator.sensitivity_analysis.

    Args:
        simulator: Simulator instance with forecast data.
        target_metric: The metric name (e.g., 'Net Income', 'Investor ROE').
        year: The year (1-based) for which sensitivity is calculated.
    """
    # Ensure baseline data
    if not simulation.summary:
        simulation.monte_carlo_forecast()
    df = simulation.sensitivity_analysis(target_metric, year)
    df_sorted = df.sort_values("Range", ascending=True).reset_index(drop=True)

    # Create single-row tornado subplot
    fig = go.Figure()
    # Add traces
    for idx,(_, r) in enumerate(df_sorted.iterrows()):
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
            text=f"{r['LowValue']}",
            xanchor="right",
            yanchor="middle",
            showarrow=False,
        )
        fig.add_annotation(
            x=m_high,
            y=param,
            text=f"{r['HighValue']}",
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

    # Layout tweaks
    fig.update_layout(
        height=400 + 20 * len(df_sorted),
        width=800,
        xaxis_title=target_metric,
        yaxis=dict(title="Parameter", automargin=True),
        title_x=0.5,
        margin=dict(l=200),
        title=f"Sensitivity Analysis: {target_metric} in Year {year}",
    )

    st.plotly_chart(fig, use_container_width=True)


