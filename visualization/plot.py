from __future__ import annotations

import numpy as np

import plotly.graph_objects as go
import plotly.express as px


def dist(dist, data) -> go.Figure:
    """
    Creates a distribution plot (PDF + CDF) from a Metalog object.

    Args:
        m (str): A Metalog distribution dictionary returned by metalog.fit().

    Returns:
        go.Figure: A Plotly figure with two traces: PDF (left Y-axis) and CDF (right Y-axis).
    """

    quantiles = dist["M"].iloc[:, 1]
    pdf_values = dist["M"].iloc[:, 0]
    cdf_values = dist["M"]["y"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data.flatten(),
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


def timeseries(data) -> go.Figure:

    years = np.arange(data.shape[1])
    p10_vals = np.percentile(data, 10, axis=0)
    p50_vals = np.percentile(data, 50, axis=0)
    p90_vals = np.percentile(data, 90, axis=0)

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


def surface(
    data: np.ndarray,
    units: str = "",
    percentiles=None,
    title=None,
    min_percentile=None,
    max_percentile=None,
) -> go.Figure:
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

    data = data
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
            zaxis_title=units,
            xaxis=dict(tickmode="linear"),
            yaxis=dict(tickmode="linear", dtick=5),
        ),
        height=800,
        margin=dict(l=10, r=10, b=40, t=40),
        scene_camera_eye=dict(x=-1, y=-1, z=0.5),
    )
    return fig


def hist(label: str, data: np.ndarray, cumulative=True) -> go.Figure:
    """
    Create a histogram plot of the data.
    """
    fig = px.histogram(
        x=data.flatten(),
        nbins=50,
        title=f"Histogram of {label}",
        labels={"x": label},
        # marginal="box",
        cumulative=cumulative,
        histnorm="probability",
    )
    fig.update_layout(
        xaxis_title=label,
        yaxis_title="Count",
        height=600,
        width=800,
    )
    return fig


def tornado(df, target_metric: str) -> go.Figure:

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
