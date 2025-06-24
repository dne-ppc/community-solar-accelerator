import streamlit as st
# from layout import Layout
from models.solar import SolarSimulator


st.set_page_config(
    page_title="Community Solar Accelerator - Financial Model", layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 500px;
        max-width: 700px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "simulation" not in st.session_state:
    st.session_state.simulation = SolarSimulator()


st.session_state.simulation.sidenav()
st.session_state.simulation.tabs()

