import streamlit as st
from visualization import financial as plot

def tab(project):
    st.subheader("Financial")
    viz = st.selectbox(
        "Visualization",
        options=["NPV Decomposition", "Payback CDF", "IRR Histogram", "DSCR Heatmap"],
        key=f"{project.scenario}-select_fin_viz",
    )

    if viz == "NPV Decomposition":
        how = st.selectbox("Statistic", options=["P10", "P50", "P90", "Mean"], index=1)
        fig = plot.npv_decomposition_waterfall(project, how=how)
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Payback CDF":
        which = st.radio("Cashflow Basis", options=["project", "equity"], index=0, horizontal=True)
        discounted = st.checkbox("Discounted", value=True)
        fig = plot.payback_cdf(project, discounted=discounted, which=which)
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "IRR Histogram":
        which = st.radio("IRR Type", options=["project", "equity"], index=0, horizontal=True)
        bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5)
        fig = plot.irr_histogram(project, which=which, bins=bins)
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "DSCR Heatmap":
        psel = st.multiselect("Percentiles", options=[10, 25, 50, 75, 90], default=[10, 50, 90])
        fig = plot.dscr_heatmap(project, percentiles=psel if psel else (10, 50, 90))
        st.plotly_chart(fig, use_container_width=True)