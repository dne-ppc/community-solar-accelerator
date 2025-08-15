import streamlit as st
from visualization import tables
from visualization import plot

from projects.solar import SolarProject
from models.tensor import Tensor
from ui.financial import configure_project
from analysis.tornado import sensitivity_analysis

from utils import get_config_paths


def charts_tab(project: SolarProject) -> None:

    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Timeseries", "Surface", "Histogram"],
        index=1,
        key=f"{project.scenario}-select_chart_type",
    )

    if chart_type == "Histogram":
        names = project.output_names + project.calc_names + project.input_names
    else:
        names = project.output_names + project.calc_names

    metric = st.selectbox(
        "Select Metric",
        options=names,
        key=f"{project.scenario}-select_metric",
        index=None,
    )
    if metric:

        tensor: Tensor = getattr(project, metric)

        if chart_type == "Timeseries":
            fig = plot.timeseries(tensor.data)

        elif chart_type == "Surface":

            col1, col2 = st.columns(2)
            min_percentile = col1.number_input(
                "Select Minimum Percentile",
                key=f"{project.scenario}-select_min",
                value=10,
                min_value=1,
                max_value=50,
            )

            max_percentile = col2.number_input(
                "Select Maximum Percentile",
                key=f"{project.scenario}-select_max",
                value=90,
                min_value=51,
                max_value=100,
            )
            fig = plot.surface(
                tensor.data,
                tensor.units,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
            )

        elif chart_type == "Histogram":

            response = st.radio("Cumulative", options=["Yes", "No"], index=0)

            if response == "Yes":
                cumulative = True
            else:
                cumulative = False

            fig = plot.hist(tensor.label, tensor.data, cumulative=cumulative)

        st.plotly_chart(fig)


def summarize_iteration(project: SolarProject) -> None:
    iteration = st.number_input(
        "Iteration",
        min_value=0,
        max_value=project.iterations - 1,
        value=0,
        step=1,
        help="Select the iteration to view detailed financials.",
    )

    investment = tables.investment_iteration(project, iteration)
    system = tables.system_iteration(project, iteration)
    annual = tables.annual_iteration(project, iteration)
    npv = tables.npv_iteration(project, iteration)
    costs = tables.costs_iteration(project, iteration)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Investment Assumptions")
        st.dataframe(investment.round(2))

    with col2:
        st.subheader("System Assumptions")
        st.dataframe(system.round(2))
    with col3:
        st.subheader("Cost Assumptions")
        st.dataframe(costs.round(2))

    with col4:

        st.subheader("Net Present Value")
        st.dataframe(npv.round(0))

    st.subheader("Annual Financials")
    st.dataframe(annual.round(0), height=500)


def layout(project: SolarProject) -> None:

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
        configure_project(project)

    with tabs[1]:
        st.subheader("KPI")
        st.dataframe(tables.kpi(project))
        st.dataframe(tables.npv_summary(project))

    with tabs[2]:
        metric = st.selectbox(
            "Select Metric",
            options=project.output_names,
            key=f"{project.scenario}-select_sensitivity_output",
            index=None,
        )
        if metric:
            variables = project.assumptions[project.assumptions.is_fixed == False]
            if variables.empty:
                st.info("No unfixed inputs available for sensitivity analysis.")
            else:
                df = sensitivity_analysis(project,metric)
                fig = plot.tornado(df,metric)
                st.plotly_chart(fig)

    with tabs[3]:
        charts_tab(project)
    with tabs[4]:
        summarize_iteration(project)
