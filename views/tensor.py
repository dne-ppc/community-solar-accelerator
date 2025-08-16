import streamlit as st

from visualization import tensor as plot

from models.tensor import Tensor



def tab(project) -> None:

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
