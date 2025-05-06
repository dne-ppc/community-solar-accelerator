import streamlit as st
from layout import Layout
from models import Simulation


st.set_page_config(
    page_title="Community Solar Accelerator - Financial Model", layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "simulation" not in st.session_state:
    st.session_state.simulation = Simulation()


Layout.create_tabs()
