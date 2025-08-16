import streamlit as st
from visualization import risk


def tab(project) -> None:
    st.subheader("Risk")

    col1, col2 = st.columns([2, 1])

    with col1:
        which_plot = st.selectbox(
            "Risk visualization",
            ["VaR/CVaR", "Distribution", "NPV Profile", "Drawdown Histogram"],
            key="risk_plot_kind",
        )

        if which_plot == "VaR/CVaR":
            target = st.selectbox("Target", ["npv_total", "project_irr", "equity_irr", "dscr"])
            alpha = st.slider("Alpha (tail probability)", 0.001, 0.20, 0.05, step=0.001)
            side = st.radio("Tail", ["lower", "upper"], horizontal=True, index=0)
            fig = risk.var_cvar(project, target, alpha=alpha, side=side)
            st.plotly_chart(fig, use_container_width=True)

        elif which_plot == "Distribution":
            target = st.selectbox("Target", ["npv_total", "project_irr", "equity_irr", "dscr"])
            bins = st.slider("Bins", 10, 120, 40, step=5)
            fig = risk.distribution(project, target, bins=bins)
            st.plotly_chart(fig, use_container_width=True)

        elif which_plot == "NPV Profile":
            basis = st.radio("Cashflow basis", ["project_cashflow", "equity_cashflow"], horizontal=True)
            fig = risk.npv_profile(project, cashflow=basis)
            st.plotly_chart(fig, use_container_width=True)

        elif which_plot == "Drawdown Histogram":
            bins = st.slider("Bins", 10, 120, 40, step=5)
            fig = risk.drawdown_histogram(project, bins=bins)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Tail Risk (lower tail)**")
        tail_target = st.selectbox("Tail target", ["npv_total", "project_irr", "equity_irr"], key="tail_target")
        st.dataframe(risk.tail_risk_table(project, tail_target, alphas=(0.01, 0.05, 0.10), side="lower"),
                    use_container_width=True)

        st.markdown("**Summary**")
        st.dataframe(risk.summary_table(project, risk.DEFAULT_RISK_ITEMS), use_container_width=True)
