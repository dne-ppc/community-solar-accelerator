import streamlit as st
from models import Simulation
from visualize import (
    plot_summary_grid,
    plot_sensitivity_tornado,
)
from doc import display_latex
import pandas as pd
import plotly.graph_objects as go 


# Initialize or retrieve the Simulation in session state
def get_simulation(config_path="config.yaml") -> Simulation:
    if "sim" not in st.session_state:
        # Default iterations; user can override in sidebar
        st.session_state.sim = Simulation(config_path=config_path, iterations=5000)
    return st.session_state.sim


class Layout:
    @staticmethod
    def controls(sim: Simulation):
        """
        Sidebar controls for running simulations.
        """

        st.header("Simulation Controls")
        cols = st.columns(3)
        # Iterations input
        iters = cols[0].number_input(
            "Monte Carlo Iterations",
            min_value=100,
            max_value=50000,
            value=sim.iterations,
            step=100,
        )
        sim.iterations = int(iters)

        # Years input
        years = cols[1].number_input(
            "Simulation Years",
            min_value=1,
            max_value=50,
            value=sim.years,
            step=1,
        )
        sim.years = int(years)

        return_year = cols[2].number_input(
            "Return Years",
            min_value=1,
            max_value=50,
            value=sim.return_year,
            step=1,
        )
        sim.return_year = int(return_year)

        # Button to run forecast
        if st.button("Run Simulation", use_container_width=True, key="run_simulation"):
            # with st.spinner("Running simulation..."):
            sim.monte_carlo_forecast()
            st.success("Simulation completed!")

        st.divider()

        tabs = st.tabs(sim.distributions.keys())
        for i, label in enumerate(sim.distributions.keys()):
            dist = sim.distributions[label]
            with tabs[i]:

                dist.create_controls()

    @staticmethod
    def latex_model():
        """
        Display the LaTeX model specification.
        """
        st.header("Model Specification")
        display_latex()

    @staticmethod
    def financial_statements(sim: Simulation):
        st.header("GAAP-Style Financial Statements & Key Metrics")
        model = sim.model

        if not model.summary:
            st.info("Run the simulation first in the sidebar to view statements.")
            return

        # Select percentile
        p = st.selectbox("Select percentile", [10, 50, 90], index=1)

        # Pull the GAAP report
        report = model.get_gaap_report(percentile=p)

        # Key Metrics
        st.subheader("Key Performance Metrics")
        km = report["Key Metrics"].copy()
        # Format boolean nicely
        # km["Self-sustaining by Year 25"] = km["Self-sustaining by Year 25"].map({True: "Yes", False: "No"})
        st.dataframe(km.to_frame("Value"))

        # # Income Statement
        # st.subheader(f"Income Statement (P{p})")
        # st.dataframe(report["Income Statement"].style.format("{:,.0f}"))

        # # Balance Sheet
        # st.subheader(f"Balance Sheet (P{p})")
        # st.dataframe(report["Balance Sheet"].style.format("{:,.0f}"))

        # # Cash Flow Statement
        # st.subheader(f"Cash Flow Statement (P{p})")
        # st.dataframe(report["Cash Flow Statement"].style.format("{:,.0f}"))

    @staticmethod
    def forecast_plots(sim: Simulation):
        """
        Show time-series area plots of P10–P90 and P50 line for each metric.
        """
        st.header("Forecast Plots: Key Metrics Over Time")
        if not sim.model.summary:
            st.info("Run the simulation first in the sidebar to view plots.")
            return
        # cols = st.slider("Number of plot columns", min_value=1, max_value=4, value=2)
        plot_summary_grid(sim, 3)

    @staticmethod
    def sensitivity_analysis(sim: Simulation):
        """
        Render a tornado chart showing sensitivity of a selected metric and year.
        """
        st.header("Sensitivity Analysis: Tornado Chart")
        if not sim.summary:
            st.info("Run the simulation first in the sidebar to view sensitivity.")
            return
        metric = st.selectbox(
            "Select metric for sensitivity analysis", list(sim.summary.keys())
        )
        year = st.slider(
            "Select year (1-based)", min_value=1, max_value=sim.years, value=sim.years
        )
        plot_sensitivity_tornado(sim, metric, int(year))

    @staticmethod
    def investor_benefits(sim: Simulation):
        """
        Display benefits specific to private investors.
        """
        st.header("Investor Benefits")
        st.markdown(
            """
- **Steady Cash Flows & Diversification:** Long-term PPAs and a portfolio of projects ensure predictable, inflation‑protected returns and spread local variability.
- **Government Incentives:** 30% Alberta Investor Tax Credit on equity and potential federal benefits boost net yields.
- **Regulated & Protected:** Fully compliant with Alberta Securities Act, with formal offering documents, transparent governance, and investor safeguards.
- **Social/ESG Impact:** Directly fund community solar projects, quantified by kW installed, MWh generated, and CO₂ avoided.

*See below: projected IRR vs. exit year and key-year IRR bar chart.*
            """
        )
        if not sim.summary:
            st.info("Run the simulation first in the sidebar to view metrics.")
            return

        # Let user pick percentile
        pct = st.selectbox(
            "Select percentile for investor metrics", [10, 50, 90], index=1
        )

        # Get the table and show it
        report = sim.model.get_gaap_report(
            percentile=pct
        )  # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        metrics = report["Investor Metrics"]
        # Convert Series to DataFrame, rename column
        df = metrics.to_frame(name=f"P{pct}").rename_axis("Metric")
        st.dataframe(df.style.format("{:,.2f}"),use_container_width=False)

    @staticmethod
    def community_benefits(sim: Simulation):
        """
        Display benefits to community organizations.
        """
        st.header("Community Benefits")
        st.markdown(
            """
- **No Upfront Cost:** Solar installations require zero capital from hosts; the accelerator funds build and O&M.
- **Lower Energy Bills:** PPA rates several cents below retail lock in savings from day one, freeing budgets for core programs.
- **Hassle-Free Operation:** O&M, insurance, and performance guarantees managed by the accelerator.
- **Local Leadership & Green Impact:** Visible clean energy on community sites enhances reputation and helps meet sustainability goals.

*See below: projected cumulative utility cost savings over time.*
            """
        )

        # Plot community savings using Plotly
        summary = sim.summary.get("Community Savings (CAD)", {})
        if summary:
            years = list(range(len(summary.get("P50", []))))
            fig = go.Figure()
            for p in [10, 50, 90]:
                vals = summary.get(f"P{p}", [])
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=vals,
                        mode='lines',
                        name=f'P{p}'
                    )
                )
            fig.update_layout(
                title="Community Savings (CAD) Over Time",
                xaxis_title="Year",
                yaxis_title="Savings (CAD)",
                legend_title="Percentile"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the simulation first to view community savings.")

    @staticmethod
    def government_benefits(sim: Simulation):
        """
        Display benefits to government funders.
        """
        st.header("Government Benefits")
        st.markdown(
            """
- **High Leverage:** Each $1M of public seed grants ~5× total project funding through co-investment and revolving capital.
- **Budgetary Relief:** Reduced operating costs for public/nonprofit buildings lessen subsidy needs.
- **Climate & Energy Goals:** Direct contributions to Alberta’s renewable targets and GHG reductions.
- **Economic Diversification:** Stimulates local green jobs in installation and maintenance, aligning with diversification mandates.

*See below: simulated seed fund leverage and cumulative impacts.*
            """
        )
        if not sim.summary:
            st.info("Run the simulation first in the sidebar to view metrics.")
            return

        # Let user pick percentile
        pct = st.selectbox(
            "Select percentile for government metrics", [10, 50, 90], index=1
        )

        # Get the table and show it
        report = sim.model.get_gaap_report(
            percentile=pct
        )  # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        metrics = report["Government Metrics"]
        # Convert Series to DataFrame, rename column
        df = metrics.to_frame(name=f"P{pct}").rename_axis("Metric")
        st.dataframe(df.style.format("{:,.2f}"),use_container_width=False)


    @staticmethod
    def create_tabs():
        """
        Main layout: create tabs for all sections.
        """
        sim = get_simulation()
        tabs = st.tabs(
            [
                "Formula",
                "Inputs",
                "Forecasts",
                "Sensitivity",
                "Investor Benefits",
                "Community Benefits",
                "Government Benefits",
            ]
        )
        with tabs[0]:
            Layout.latex_model()
        with tabs[1]:
            Layout.controls(sim)
        with tabs[2]:
            Layout.forecast_plots(sim)
        with tabs[3]:
            Layout.sensitivity_analysis(sim)

        with tabs[4]:
            Layout.investor_benefits(sim)
        with tabs[5]:
            Layout.community_benefits(sim)
        with tabs[6]:
            Layout.government_benefits(sim)
