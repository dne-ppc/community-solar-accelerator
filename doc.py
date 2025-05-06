import streamlit as st

def display_latex():
    """
    Render the key equations and parameters of the Community Solar Accelerator model inline with st.latex.
    """
    st.header("Financial Model Equations")

    # Key Parameters
    st.subheader("Key Parameters")
    params = [
        ("k_t", "Project size (kW)"),
        ("c_t", "Installed cost per watt (CAD/W)"),
        ("d_t", "Debt financing fraction in year t"),
        ("r_t", "Debt interest rate in year t"),
        ("i_{cash,t}", "Cash interest rate in year t"),
        ("N_t", "Desired projects in year t"),
        ("phi_t", "Capacity factor"),
        ("p_t", "Price spread (CAD/kWh)"),
        ("o_t", "Opex per project (CAD/project)"),
        ("tau_t", "Corporate tax rate"),
        ("UsefulLife", "Depreciation life (years)"),
        ("debtTerm", "Debt amortization term (years)"),
        ("PrivateInit", "Initial private seed capital (CAD)"),
        ("ReturnYear", "Year to return private equity"),
        ("B^{pub}_0", "Initial public seed capital (CAD)"),
        ("B^{priv}_0", "Initial private seed capital (CAD)"),
    ]
    cols = st.columns(3)
    for idx, (sym, desc) in enumerate(params):
        col = cols[idx % len(cols)]
        # desc = '\mbox{' + desc + '}'
        inner_col1, inner_col2 = col.columns([1, 3],vertical_alignment="center")
        inner_col1.latex(sym)
        inner_col2.write(desc)

    # 1) Deployment & Financing
    st.subheader("Deployment & Financing")
    st.latex(r"cost_t = k_t \times 1000 \times c_t")
    st.latex(r"debtCost_t = d_t \times cost_t, \quad equityCost_t = (1 - d_t) \times cost_t")
    st.latex(r"maxPubProj_t = \left\lfloor \frac{B^{pub}_{t-1}}{equityCost_t} \right\rfloor, \quad pubProj_t = \min(N_t, maxPubProj_t)")
    st.latex(r"maxPrivProj_t = \left\lfloor \frac{B^{priv}_{t-1} - PrivateInit}{equityCost_t} \right\rfloor, \quad privProj_t = \min(N_t - pubProj_t, maxPrivProj_t)")
    st.latex(r"debtProj_t = \min\bigl(N_t - pubProj_t - privProj_t, \left\lfloor \frac{(B^{pub}_{t-1}+B^{priv}_{t-1}) \times d_t}{equityCost_t} \right\rfloor\bigr)")
    st.latex(r"CAPEX_t = cost_t \times (pubProj_t + privProj_t + debtProj_t)")

    # 2) Equity Draw Logic
    st.subheader("Equity Draws")
    st.latex(r"draw^{pub}_t = \min\bigl(B^{pub}_{t-1}, pubProj_t \times equityCost_t\bigr)")
    st.latex(r"draw^{priv}_t = \min\bigl(\max(B^{priv}_{t-1} - PrivateInit, 0), privProj_t \times equityCost_t\bigr)")

    # 3) Debt & Repayment
    st.subheader("Debt & Repayment")
    st.latex(r"D_t = D_{t-1} + debtProj_t \times d_t \times cost_t - repay_t")
    st.latex(r"repay_t = \sum_{s=1}^{t} PPMT(r_t, s, debtTerm, newDebt_s)")
    st.latex(r"I^{exp}_t = r_t \times D_{t-1}")

    # 4) Deferred Tax Liability & Amortization
    st.subheader("Deferred Tax")
    st.latex(r"DTL_t = \sum_{i=0}^{t} \bigl(taxDepr_{i,t} - bookDepr_{i,t}\bigr) \times \tau_t")
    st.latex(r"Amort_t = DTL_{t-1} - DTL_t")

    # 5) Interest Income on Cash
    st.subheader("Interest Income on Cash")
    st.latex(r"I^{inc}_t = (B^{pub}_{t-1} + B^{priv}_{t-1}) \times i_{cash,t}")

    # 6) Energy Production & Revenue
    st.subheader("Energy Production & Revenue")
    st.latex(r"K^{cum}_{t-1} = \sum_{s=0}^{t-1} N_s \times k_s")
    st.latex(r"E_t = K^{cum}_{t-1} \times 1000 \times 8760 \times \phi_t")
    st.latex(r"R_t = E_t \times p_t + I^{inc}_t")

    # 7) Expenses & Depreciation
    st.subheader("Expenses & Depreciation")
    st.latex(r"OPEX_t = o_t \times \sum_{s=0}^{t} totalProjects_s")
    st.latex(r"Dep_t = \frac{\sum_{s=0}^{t-1} CAPEX_s}{UsefulLife}")

    # 8) Taxes & Net Income
    st.subheader("Taxes & Net Income")
    st.latex(r"TaxExp_t = \tau_t \times \max(0, R_t - OPEX_t - Dep_t - I^{exp}_t)")
    st.latex(r"NetInc_t = R_t - OPEX_t - Dep_t - I^{exp}_t - TaxExp_t")

    # 9) Returns to Private Investors
    st.subheader("Investor Returns")
    st.latex(r"RetPriv_t = \begin{cases} PrivateInit \times \theta, & t < ReturnYear \\ PrivateInit \times \theta + \max(B^{priv}_{t-1},0), & t = ReturnYear \\ 0, & t > ReturnYear \end{cases}")

    # 10) Cash Flow & Balance Sheet
    st.subheader("Cash Flow Statement")
    st.latex(r"CF_{op,t} = NetInc_t + Dep_t + Amort_t + I^{exp}_t")
    st.latex(r"CF_{inv,t} = -CAPEX_t")
    st.latex(r"CF_{fin,t} = debtProj_t \times d_t \times cost_t - repay_t - RetPriv_t")
    st.latex(r"Cash_t = Cash_{t-1} + CF_{op,t} + CF_{inv,t} + CF_{fin,t}")

    st.subheader("Balance Sheet")
    st.latex(r"Assets_t = Cash_t + \sum_{s=0}^{t} CAPEX_s - \sum_{u=0}^{t} Dep_u")
    st.latex(r"Liabilities_t = D_t + DTL_t")
    st.latex(r"Equity_t = B^{pub}_t + B^{priv}_t")
