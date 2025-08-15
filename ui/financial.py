import streamlit as st
from models.types import Input
from utils import get_config_paths
from ui.distributions import controls



def configure_project(project) -> None:

    if f"{project.scenario}-input_selection_row" not in st.session_state:
        st.session_state[f"{project.scenario}-input_selection_row"] = 0

    st.subheader("Configure Financial Model")
    cols = st.columns(
        [
            0.10,
            0.10,
            0.30,
            0.10,
            0.30,
            0.10,
        ],
        vertical_alignment="bottom",
    )
    project.years = cols[0].number_input(
        "Years",
        min_value=1,
        value=project.years,
        step=1,
        key="model_years",
    )
    project.iterations = cols[1].number_input(
        "Iterations",
        min_value=1,
        value=project.iterations,
        step=1,
        key="model_iters",
    )

    options = get_config_paths()
    config_path = cols[2].selectbox(
        "Input Config File",
        options=options,
        index=options.index("inputs/Base.yaml"),
        key=f"{project.scenario}_model_config_path",
    )
    if cols[3].button("Load Model", key="load_model", use_container_width=True):
        project.update_inputs(config_path)

    project.scenario = cols[4].text_input(
        "Scenario Name",
        value=project.scenario,
        key="model_scenario",
    )

    if cols[5].button(
        "Save", key=f"{project.scenario}_save_config", use_container_width=True
    ):
        project.save_model()

    col1, col2 = st.columns([0.50, 0.50])

    with col1:
        st.subheader("Assumptions")
    selection = col1.dataframe(
        project.assumptions, height=800, selection_mode="single-row", on_select="rerun"
    )["selection"]
    if selection["rows"]:
        st.session_state[f"{project.scenario}-input_selection_row"] = selection["rows"][0]

    with col2:
        st.subheader("Inputs")
        name = st.selectbox(
            "Select Input",
            options=project.input_names,
            index=(
                selection["rows"][0]
                if selection["rows"]
                else st.session_state[f"{project.scenario}-input_selection_row"]
            ),
            key=f"{project.scenario}-select_config_input",
            label_visibility="collapsed",
        )
        if name:
            input: Input = getattr(project, name)

            controls(input)