import streamlit as st

from projects.solar import SolarProject
from analysis.tornado import sensitivity_analysis
from visualization import tornado


def tab(project: SolarProject) -> None:

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
            df = sensitivity_analysis(project, metric)
            fig = tornado.plot(df, metric)
            st.plotly_chart(fig)
