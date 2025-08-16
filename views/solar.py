import streamlit as st
from visualization import tables

from projects.solar import SolarProject
from views.project import configure
from views import risk, tensor, tornado, financial, solver


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
            "Tensors",
            "Financial",
            "Risk",
            "Solver",
            "Iteration Outputs",
        ]
    )

    with tabs[0]:
        configure(project)

    with tabs[1]:
        st.subheader("KPI")
        st.dataframe(tables.kpi(project))
        st.dataframe(tables.npv_summary(project))

    with tabs[2]:
        tornado.tab(project)

    with tabs[3]:
        tensor.tab(project)

    with tabs[4]:
        financial.tab(project)
    with tabs[5]:
        risk.tab(project)

    with tabs[6]:
        solver.tab(project)

    with tabs[7]:
        summarize_iteration(project)
