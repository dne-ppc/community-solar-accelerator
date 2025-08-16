from __future__ import annotations


import streamlit as st
from visualization import distributions

def controls(tensor) -> None:
    """
    Streamlit UI: lets the user toggle between a fixed value or editable
    P10/P50/P90 sliders.
    """
    container = st.container(border=True)
    container.write(f"### {tensor.label}")

    # Fixed‐value toggle
    fixed_key = f"{tensor.scenario}_{tensor.label}_use_fixed"
    tensor.use_fixed = container.checkbox(
        "Use fixed value", value=tensor.use_fixed, key=fixed_key
    )

    if tensor.use_fixed:
        # Ask for the constant to use
        val_key = f"{tensor.scenario}_{tensor.label}_fixed_value"
        default = tensor.fixed_value if tensor.fixed_value is not None else tensor.p50
        tensor.fixed_value = container.number_input(
            "Fixed value",
            value=default,
            step=tensor.step,
            key=val_key,
        )
    else:
        # Original P10/P50/P90 controls
        p_low_key = f"{tensor.scenario}_{tensor.label}_low"
        p_med_key = f"{tensor.scenario}_{tensor.label}_medium"
        p_high_key = f"{tensor.scenario}_{tensor.label}_high"
        col1, col2, col3 = container.columns(3)
        with col1:
            low_val = col1.number_input("P10", value=tensor.p10, key=p_low_key)
        with col2:
            med_val = col2.number_input("P50", value=tensor.p50, key=p_med_key)
        with col3:
            high_val = col3.number_input("P90", value=tensor.p90, key=p_high_key)

        tensor.p10 = low_val
        tensor.p50 = med_val
        tensor.p90 = high_val
        # Plotly preview (always show, even when fixed)
        try:
            # Build a tiny metalog using current p10/p50/p90

            if not bool(tensor.fixed_value):
                return

            container.plotly_chart(
                distributions.plot(tensor.dist, tensor.data),
                use_container_width=True,
                key=f"{tensor.scenario}_{tensor.label}_dist_plot",
            )
        except Exception as e:
            st.error(f"Error plotting distribution: {e}")
    if st.button("Update Data", key=f"{tensor.scenario}_{tensor.label}_update_data"):
        tensor.update_data()
        # st.rerun()

