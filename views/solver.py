import streamlit as st
import numpy as np
from visualization.solver import run_solver, result_df, fig_response_curve


def tab(project) -> None:

    st.subheader("Solvers")
    solver = st.selectbox(
        "Choose solver",
        [
            "CAPEX for NPV=0",
            "Price for target IRR%",
            "Price for min DSCR",
            "Solve input for metric",
            "Breakeven (metric=0)",
        ],
        index=0,
    )
    p = st.slider("Percentile (P)", 1.0, 99.0, 50.0, step=1.0)

    input_names = getattr(project, "input_names", [])
    if solver == "CAPEX for NPV=0":
        input_name = "capex"
        st.text_input("Input name", value="capex", disabled=True)
    else:
        input_name = st.selectbox("Input name", options=input_names or [""])

    def _bounds_default(inp: str) -> tuple[float, float]:
        try:
            data = getattr(project, inp).data
            base = float(np.nanmedian(data[:, 0] if data.ndim == 2 else data))
            lo = base * 0.5 if base > 0 else base * 1.5
            hi = base * 1.5 if base > 0 else base * 0.5
            if lo > hi:
                lo, hi = hi, lo
            return (float(lo), float(hi))
        except Exception:
            return (0.0, 1.0)

    default_lo, default_hi = _bounds_default(input_name)
    colA, colB = st.columns(2)
    with colA:
        lo = st.number_input("Lower bound", value=float(default_lo), format="%.6f")
    with colB:
        hi = st.number_input("Upper bound", value=float(default_hi), format="%.6f")
    bounds = (float(lo), float(hi))

    target_metric = target_value = min_dscr_target = target_irr_percent = None
    key = None
    year = "final"

    if solver == "Solve input for metric":
        target_metric = st.selectbox(
            "Target metric", ["npv_total", "project_irr", "equity_irr", "dscr", "roi"]
        )
        target_value = st.number_input("Target value", value=0.0, format="%.6f")
        year = st.selectbox(
            "Timeseries aggregation (year)", ["final", "first", "min", "max", "mean"]
        )
    elif solver == "Breakeven (metric=0)":
        target_metric = st.selectbox(
            "Target metric", ["npv_total", "project_irr", "equity_irr", "dscr", "roi"]
        )
        year = st.selectbox(
            "Timeseries aggregation (year)", ["final", "first", "min", "max", "mean"]
        )
        target_value = 0.0
    elif solver == "Price for target IRR%":
        target_irr_percent = st.number_input(
            "Target equity IRR (%)", value=8.0, step=0.25, format="%.3f"
        )
    elif solver == "Price for min DSCR":
        min_dscr_target = st.number_input(
            "Minimum DSCR target", value=1.3, step=0.05, format="%.3f"
        )

    if st.button("Solve", type="primary"):
        try:
            res = run_solver(
                project,
                solver,
                input_name=input_name,
                target_metric=target_metric,
                target_value=target_value,
                key=key,
                year=year,
                bounds=bounds,
                p=p,
                min_dscr_target=min_dscr_target,
                target_irr_percent=target_irr_percent,
            )
            st.success(
                f"Solved {res.input_name} = {res.value:.6g} → {res.metric}≈{res.achieved:.6g} (iters: {res.iterations}, converged: {res.converged})"
            )
            st.dataframe(result_df(res), use_container_width=True)

            metric = (
                "npv_total"
                if solver == "CAPEX for NPV=0"
                else (
                    "equity_irr"
                    if solver == "Price for target IRR%"
                    else "dscr" if solver == "Price for min DSCR" else target_metric
                )
            )
            target_line = (
                0.0
                if solver in ("CAPEX for NPV=0", "Breakeven (metric=0)")
                else (
                    target_irr_percent
                    if solver == "Price for target IRR%"
                    else (
                        min_dscr_target
                        if solver == "Price for min DSCR"
                        else target_value
                    )
                )
            )
            year_sel = "min" if solver == "Price for min DSCR" else "final"
            st.plotly_chart(
                fig_response_curve(
                    project,
                    input_name,
                    metric,
                    bounds=bounds,
                    p=p,
                    year=year_sel,
                    solver_result=res,
                    target_value=target_line,
                    title="Response curve (percentile metric vs input)",
                ),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Solver failed: {e}")
