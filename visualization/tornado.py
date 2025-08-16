from __future__ import annotations

from typing import Iterable

import numpy as np

import plotly.graph_objects as go
import plotly.express as px



def plot(df, target_metric: str) -> go.Figure:

    fig = go.Figure()

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
        title=f"Sensitivity Analysis: {target_metric}",
    )
    return fig