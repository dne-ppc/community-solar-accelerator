import streamlit as st

# from layout import Layout
# from models.portfolio import CommunityPortfolio
from projects.solar import SolarProject
from views.solar import layout


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


# if "portfolio" not in st.session_state:
#     st.session_state.portfolio = CommunityPortfolio()


# # st.session_state.simulation.sidenav()
# st.session_state.portfolio.layout()

if "project" not in st.session_state:
    st.session_state.project = SolarProject()


# st.session_state.simulation.sidenav()
layout(st.session_state.project)
