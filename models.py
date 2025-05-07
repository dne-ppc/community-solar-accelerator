# models.py
from typing import Dict, List, Optional, TypeVar, Any, Tuple
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator
from metalog import metalog
import yaml
import numpy_financial as npf
# from numpy_financial import ppmt, ipmt



PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")

np.float_ = np.float64


class Distribution(BaseModel):
    """
    Stores the basic configuration for a single parameter's Metalog distribution.
    For example: Price of Gasoline ($/L), with min_value, max_value, p10, p50, p90, etc.
    """

    label: str
    min_value: float | int
    max_value: float | int
    p10: float | int
    p50: float | int
    p90: float | int
    step: float | int
    boundedness: str = "b"  # 'b' for two-sided bounding in metalog

    use_fixed: bool = Field(default=False, description="Override to a fixed constant?")
    fixed_value: bool | int | float = Field(
        default=False,
        description="If use_fixed is True, draw every sample from this value",
    )

    def create_data(
        self,
        iterations: int,
        years: int,
        sensitivity_param: Optional[str] = None,
        sensitivity_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draws samples for `iterations` × (`years` + 1).
        Priority:
          1) If sensitivity override matches label → full constant array of sensitivity_value.
          2) Else if use_fixed=True and fixed_value is set → full constant array of fixed_value.
          3) Otherwise sample from Metalog as before.
        """
        shape = (iterations, years + 1)

        # 1) Sensitivity‐analysis override
        if sensitivity_param == self.label and sensitivity_value is not None:
            return np.full(shape, sensitivity_value, dtype=float)

        # 2) Fixed‐value override
        if self.use_fixed:
            # Fallback if fixed_value is None: use median
            val = self.fixed_value if self.fixed_value is not None else self.p50
            return np.full(shape, val, dtype=float)

        # 3) Metalog sampling
        if not metalog:
            raise RuntimeError("metalog library not available")

        dist = metalog.fit(
            x=[self.p10, self.p50, self.p90],
            boundedness=self.boundedness,
            bounds=[self.min_value - self.step, self.max_value + self.step],
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )
        values = metalog.r(dist, n=iterations * (years + 1))
        return values.reshape(shape)

    def create_controls(self) -> None:
        """
        Streamlit UI: lets the user toggle between a fixed value or editable
        P10/P50/P90 sliders.
        """
        container = st.container(border=True)
        container.write(f"### {self.label}")

        # Fixed‐value toggle
        fixed_key = f"{self.label}_use_fixed"
        self.use_fixed = container.checkbox(
            "Use fixed value", value=self.use_fixed, key=fixed_key
        )

        if self.use_fixed:
            # Ask for the constant to use
            val_key = f"{self.label}_fixed_value"
            default = self.fixed_value if self.fixed_value is not None else self.p50
            self.fixed_value = container.number_input(
                "Fixed value",
                value=default,
                step=self.step,
                key=val_key,
            )
        else:
            # Original P10/P50/P90 controls
            p10_key = f"{self.label}_low"
            p50_key = f"{self.label}_medium"
            p90_key = f"{self.label}_high"
            col1, col2, col3 = container.columns(3)
            with col1:
                low_val = col1.number_input("P10", value=self.p10, key=p10_key)
            with col2:
                med_val = col2.number_input("P50", value=self.p50, key=p50_key)
            with col3:
                high_val = col3.number_input("P90", value=self.p90, key=p90_key)

            self.p10 = low_val
            self.p50 = med_val
            self.p90 = high_val
            # Plotly preview (always show, even when fixed)
            try:
                # Build a tiny metalog using current p10/p50/p90
                dist = metalog.fit(
                    x=[self.p10, self.p50, self.p90],
                    boundedness=self.boundedness,
                    bounds=[self.min_value - self.step, self.max_value + self.step],
                    term_limit=3,
                    probs=[0.1, 0.5, 0.9],
                )
                if not bool(self.fixed_value):
                    return
                fig = self.create_dist_plot(dist)
                container.plotly_chart(
                    fig, use_container_width=True, key=f"dist_{self.label}"
                )
            except Exception as e:
                st.error(f"Error plotting distribution: {e}")

    def create_dist_plot(self, m: str):
        """
        Creates a distribution plot (PDF + CDF) from a Metalog object.

        Args:
            m (str): A Metalog distribution dictionary returned by metalog.fit().

        Returns:
            go.Figure: A Plotly figure with two traces: PDF (left Y-axis) and CDF (right Y-axis).
        """

        quantiles = m["M"].iloc[:, 1]
        pdf_values = m["M"].iloc[:, 0]
        cdf_values = m["M"]["y"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=pdf_values / sum(pdf_values),
                mode="lines",
                name="PDF",
                line=dict(color="blue"),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=cdf_values,
                mode="lines",
                name="CDF",
                line=dict(color="red", dash="dash"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Value"),
            yaxis=dict(title="PDF", title_font_color="blue", tickfont_color="blue"),
            yaxis2=dict(
                title="CDF",
                title_font_color="red",
                tickfont_color="red",
                overlaying="y",
                side="right",
            ),
            legend=dict(x=0, y=1.1, orientation="h"),
            template="plotly",
            hovermode="x unified",
        )
        return fig


class FinancialModel(BaseModel):

    years: int = Field(default=25, description="Number of years to simulate")
    return_year: int = Field(default=10, description="Year to return private equity")

    revenue: NdArray = None
    opex: NdArray = None
    capex: NdArray = None
    interest_expense: NdArray = None
    depreciation_expense: NdArray = None
    deferred_tax_amortization: NdArray = None
    tax_expense: NdArray = None
    returns_to_private: NdArray = None
    public_equity_balance: NdArray = None
    private_equity_balance: NdArray = None
    total_projects: NdArray = None
    total_capacity: NdArray = None
    assets: NdArray = None
    interest_income: NdArray = None
    principal_repayment: NdArray = None
    debt_balance: NdArray = None
    deferred_tax_liability: NdArray = None
    initial_private: NdArray = None
    initial_public: NdArray = None
    community_savings: NdArray = None

    summary: Dict[str, Dict[str, NdArray]] = Field(default_factory=dict)

    useful_life_years: int = Field(
        default=25, description="Useful life for straight-line depreciation"
    )
    debt_term_years: int = Field(
        default=10, description="Term (years) for loan amortization"
    )



    def run_simulation(self, inputs: Dict[str, np.ndarray]):
        n_iter = inputs["Public Seed Grant (CAD)"].shape[0]
        Y = self.years

        # Initialize GAAP arrays
        revenue = np.zeros((n_iter, Y))
        opex = np.zeros((n_iter, Y))
        capex = np.zeros((n_iter, Y))
        depreciation_expense = np.zeros((n_iter, Y))
        interest_expense = np.zeros((n_iter, Y))
        principal_repayment = np.zeros((n_iter, Y))
        debt_balance = np.zeros((n_iter, Y))
        deferred_tax_liability = np.zeros((n_iter, Y))
        deferred_tax_amortization = np.zeros((n_iter, Y))
        tax_expense = np.zeros((n_iter, Y))
        returns_to_private = np.zeros((n_iter, Y))
        public_equity_balance = np.zeros((n_iter, Y))
        private_equity_balance = np.zeros((n_iter, Y))
        total_projects = np.zeros((n_iter, Y))
        total_capacity = np.zeros((n_iter, Y))
        assets = np.zeros((n_iter, Y))
        interest_income = np.zeros((n_iter, Y))
        community_savings = np.zeros((n_iter, Y))
        cash = (
            inputs["Public Seed Grant (CAD)"].copy()[:, 0]
            + inputs["Private Investment (CAD)"].copy()[:, 0]
        )
        cash_rate_arr = inputs.get("Cash Interest Rate (%)") / 100.0

        # Track capex and dtc history for amortization
        capex_history = np.zeros((n_iter, Y, Y))

        # Initial balances
        public_balance = inputs["Public Seed Grant (CAD)"].copy()[:, 0]
        private_balance = inputs["Private Investment (CAD)"].copy()[:, 0]
        initial_private = private_balance.copy()
        initial_public = public_balance.copy()

        # Simulation loop
        for t in range(Y):

            interest_income_t = (
                (public_balance + private_balance) * cash_rate_arr[:, t] if t > 0 else 0
            )
            interest_income[:, t] = interest_income_t
            cash += interest_income_t
            # --- Project and financing ---
            desired = inputs["Projects per Year"][:, t].astype(int)
            size_t = inputs["Project Size (kW)"][:, t] 
            cost_kw = inputs["Installed Cost (CAD/W)"][:, t] *1000
            df_t = inputs.get("Debt Financing (%)")[:, t] / 100
            int_rate = inputs.get("Debt Interest Rate (%)")[:, t] / 100
            payout_rate = inputs.get("Target Investor Return (%)")[:, t] / 100
            tax_rate = inputs.get("Corporate Tax Rate (%)")[:, t] / 100
            opex_proj = inputs.get("Opex per Project (CAD/project)")[:, t]
            cf = inputs.get("Capacity Factor (%)")[:, t] / 100
            retail_price = inputs.get("Retail Price (CAD/kWh)")[:, t]
            ppa_price = inputs.get("PPA Price (CAD/kWh)")[:, t]

            # 1) per‐project costs
            portfolio_cost = size_t * cost_kw  # total cost per project
            debt_cost = portfolio_cost * df_t  # portion financed by debt
            equity_cost = portfolio_cost - debt_cost  # portion financed by equity

            # 2) how many projects do we want?
            #    (you already have `desired` = number of projects you'd like to build)

            # 3) equity‐finance with public funds first
            max_pub_projects = (
                np.floor(public_balance / equity_cost).astype(int).clip(0, None)
            )
            pub_projects = np.minimum(desired, max_pub_projects)
            public_balance -= pub_projects * equity_cost

            # 4) equity‐finance remaining with private funds
            remaining = desired - pub_projects
            max_priv_projects = (
                np.floor(private_balance / equity_cost).astype(int).clip(0, None)
            )
            priv_projects = np.minimum(remaining, max_priv_projects)
            private_balance -= priv_projects * equity_cost

            eq_projects = pub_projects + priv_projects
            eq_projects = np.minimum(eq_projects, desired)

            # 5) whatever’s still “desired” gets debt‐financed
            equity_available = public_balance + private_balance  # shape (n_iter,)

            # maximum cash you’re willing to put up for equity share of debt projects
            # (e.g. if df_t = 0.01, only 1% of your equity can be used to support debt)
            debt_capacity_cash = equity_available * df_t          # shape (n_iter,)

            # cost per project that *equity* must cover
            # (since debt_cost = portfolio_cost * df_t,
            #  equity_cost = portfolio_cost - debt_cost)
            # already computed above as equity_cost

            # how many projects you can debt‐finance given that capacity
            max_debt_projects = np.floor(debt_capacity_cash / equity_cost).astype(int)
            max_debt_projects = np.clip(max_debt_projects, 0, None)   # no negatives

            # number of projects still “desired” after pure equity financing
            remaining_projects = (desired - eq_projects).clip(0, None)

            # final count of debt‐financed projects
            debt_projects = np.minimum(remaining_projects, max_debt_projects)

            # now draw the debt and add to cash
            new_debt = debt_projects * debt_cost
            public_balance += new_debt

            # 6) record total new projects and cumulative
            new_projects = eq_projects + debt_projects
            total_projects[:, t] = new_projects + (
                total_projects[:, t - 1] if t > 0 else 0
            )

            # Record CAPEX & history
            capex_t = portfolio_cost * new_projects
            capex[:, t] = capex_t
            capex_history[:, :, t] = np.eye(Y)[t] * capex_t[:, None]

            # Draw from public equity
            draw_pub = np.minimum(public_balance, portfolio_cost)
            public_balance -= draw_pub
            remaining = portfolio_cost - draw_pub

            # Draw private equity only if above initial contribution
            private_avail = np.maximum(private_balance - initial_private, 0)
            draw_priv = np.minimum(private_avail, remaining)
            private_balance -= draw_priv
            remaining -= draw_priv


            for project_year in range(t + 1):
                if project_year <= self.debt_term_years:
                    principal = npf.ppmt(
                        rate=int_rate,
                        per=project_year + 1,
                        nper=self.debt_term_years,
                        pv=new_debt,
                    )
                    principal_repayment[:, t] += principal


            # Update debt balance
            debt_balance[:, t] = (
                (debt_balance[:, t - 1] if t > 0 else 0)
                + new_debt
                - principal_repayment[:, t]
            )

            # Cash updates: draw debt, pay capex, repay principal

            public_balance -= capex_t
            public_balance -= principal_repayment[:, t]

            # --- Operations & revenue ---
            prev_cap = total_capacity[:, t - 1]
            total_capacity[:, t] = prev_cap + new_projects * size_t
            energy = prev_cap * cf * 8760
            revenue[:, t] = energy * ppa_price + interest_income_t
            opex[:, t] = opex_proj * total_projects[:, t]

            community_savings[:, t] = retail_price * energy - ppa_price * energy

            # Depreciation: straight-line on cumulative CAPEX
            cumulative_capex = np.sum(capex[:, :t], axis=1)
            depreciation_expense[:, t] = cumulative_capex / self.useful_life_years

            

            dtl_t = np.zeros_like(deferred_tax_liability[:, t])
            for i in range(t + 1):
                # capex of vintage i
                cap_i = capex_history[:, i, i]
                # book depreciation on that vintage through t
                years_depr = min(t - i + 1, self.useful_life_years)
                book_depr_cum = cap_i * (years_depr / self.useful_life_years)
                # tax depreciation cumulative (assume 30% declining balance)
                tax_depr_cum = cap_i * (1 - (1 - 0.3) ** years_depr)

                dtl_t += (tax_depr_cum - book_depr_cum) * tax_rate

            deferred_tax_liability[:, t] = dtl_t


            # 2) deferred‑tax amortization (the “deferred” portion of tax expense) is just the reversal of DTL:
            if t == 0:
                deferred_tax_amortization = np.zeros_like(dtl_t)
            else:
                # reduction in liability releases into income
                deferred_tax_amortization = (
                    deferred_tax_liability[:, t - 1] - deferred_tax_liability[:, t]
                )

            # Interest expense on outstanding debt
            interest_expense[:, t] = (debt_balance[:, t - 1] if t > 0 else 0) * int_rate

            # Tax expense: current tax plus deferred amortization
            taxable = (
                revenue[:, t]
                - opex[:, t]
                - interest_expense[:, t]
                - depreciation_expense[:, t]
            )
            current_tax = tax_rate * np.maximum(taxable, 0)
            tax_expense[:, t] = current_tax + deferred_tax_amortization

            # Net income
            net_inc = (
                revenue[:, t]
                - opex[:, t]
                - depreciation_expense[:, t]
                - interest_expense[:, t]
                - tax_expense[:, t]
            )

            # Investor returns and principal return
            if t < self.return_year:
                payout = initial_private * payout_rate
            elif t == self.return_year:
                payout = initial_private * payout_rate + np.maximum(private_balance, 0)
                private_balance[:] = 0
            else:
                payout = np.zeros(n_iter)
            returns_to_private[:, t] = payout

            public_balance += net_inc - payout

            # Assets = net book value of PP&E + cash
            assets[:, t] = (
                cumulative_capex - np.cumsum(depreciation_expense, axis=1)[:, t] + cash
            )
            public_equity_balance[:, t] = public_balance
            private_equity_balance[:, t] = private_balance

        # Assign to self
        self.revenue = revenue
        self.opex = opex
        self.capex = capex
        self.depreciation_expense = depreciation_expense
        self.interest_expense = interest_expense
        self.principal_repayment = principal_repayment
        self.debt_balance = debt_balance
        self.deferred_tax_liability = deferred_tax_liability
        self.tax_expense = tax_expense
        self.returns_to_private = returns_to_private
        self.public_equity_balance = public_equity_balance
        self.private_equity_balance = private_equity_balance
        self.total_projects = total_projects
        self.total_capacity = total_capacity
        self.assets = assets
        self.interest_income = interest_income
        self.initial_private = initial_private
        self.initial_public = initial_public
        self.deferred_tax_amortization = deferred_tax_amortization
        self.community_savings = community_savings

        self.summary = self.summarize_percentiles()

    def summarize_percentiles(
        self, percentiles: List[float] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute P10/P50/P90 for each raw series *before* performing any combinations,
        then build derived metrics (net income, expenses, equity, liabilities) from
        those percentile series. Returns a dict of {metric: {P10: array, …}}.
        """
        if percentiles is None:
            percentiles = [10, 50, 90]
        names = [f"P{p}" for p in percentiles]

        # 1) Precompute percentiles of each *raw* series:
        pct = lambda arr: {
            f"P{p}": np.percentile(arr, [p], axis=0)[0] for p in percentiles
        }

        rev = self.revenue
        opex = self.opex
        depr = self.depreciation_expense
        intl = self.interest_expense
        tax = self.tax_expense
        net = rev - opex - depr - intl - tax

        expenses = opex + depr + intl + tax
        assets = self.assets
        

        equity = self.public_equity_balance + self.private_equity_balance

        liab = assets - equity


        # 2) Build composite percentiles:
        #    Note: we *subtract/add* the matching percentile arrays.


        # 3) Assemble into the same summary format:
        summary = {
            "Assets (CAD)": pct(assets),
            "Liabilities (CAD)": pct(liab),
            "Equity (CAD)": pct(equity),
            "Revenue (CAD)": pct(rev),
            "Expenses (CAD)": pct(expenses),
            "Net Income (CAD)": pct(net),
            "Total Projects": pct(self.total_projects),
            "Total Capacity (kW)": pct(self.total_capacity),
            "Capital Expenditures (CAD)": pct(self.capex),
            "Community Savings (CAD)": pct(self.community_savings),
            "Returns to Private (CAD)": pct(self.returns_to_private),
        }

        return summary
    
    def get_investor_dashboard(self, percentile: int = 50) -> Dict[str, Any]:
        """Enhanced investor-focused metrics and visualizations"""
        def pct(arr):
            return np.percentile(arr, [percentile], axis=0)[0]
        
        # Cash flow analysis
        initial_investment = pct(self.initial_private)
        print(f"Initial Investment (P{percentile}): {initial_investment}")
        returns = pct(self.returns_to_private[:,:self.return_year+1])
        cumulative_returns = np.cumsum(returns)
        
        # Create cash flow chart
        fig_cashflow = go.Figure()
        fig_cashflow.add_trace(go.Bar(
            x=list(range(self.return_year+1)),
            y=returns,
            name='Annual Returns'
        ))
        fig_cashflow.add_trace(go.Scatter(
            x=list(range(self.return_year+1)),
            y=cumulative_returns,
            name='Cumulative Returns',
            line=dict(color='red', width=2)
        ))
        fig_cashflow.update_layout(
            title=f'Investor Cash Flows (P{percentile})',
            xaxis_title='Year',
            yaxis_title='CAD',
            hovermode='x unified'
        )
        st.plotly_chart(fig_cashflow, use_container_width=True)
        
        # Risk metrics
        total_return = cumulative_returns[-1]
        irr = npf.irr([-initial_investment] + list(returns)) * 100
        multiple = total_return / initial_investment

        metrics= pd.Series({
                'Initial Investment (CAD)': initial_investment,
                'Total Return (CAD)': total_return,
                'Multiple (x)': multiple,
                'IRR (%)': irr,
                'Payback Period (years)': np.argmax(cumulative_returns >= initial_investment) + 1
            }
        )

        df = metrics.to_frame(name=f"P{pct}").rename_axis("Metric")
        st.dataframe(df.style.format("{:,.2f}"), use_container_width=False)


    def get_government_dashboard(self, percentile: int = 50) -> Dict[str, Any]:
        """Enhanced government-focused metrics and visualizations"""
        def pct(arr):
            return np.percentile(arr, [percentile], axis=0)[0]
        
        public_grant = pct(self.initial_public)
        total_capacity = pct(self.total_capacity)[-1]
        total_projects = pct(self.total_projects)[-1]
        community_savings = pct(self.community_savings)[-1]
        
        # Capacity deployment chart
        fig_capacity = go.Figure()
        fig_capacity.add_trace(go.Scatter(
            x=list(range(self.years)),
            y=pct(self.total_capacity),
            name='Cumulative Capacity',
            line=dict(color='green', width=2)
        ))
        
        # Leverage metrics
        total_capex = pct(self.capex).sum()
        leverage_multiple = total_capex / public_grant
        

        metrics= pd.Series({
                'Public Grant (CAD)': public_grant,
                'Total Capacity Deployed (kW)': total_capacity,
                'Total Projects Built': total_projects,
                'Total Community Savings (CAD)': community_savings,
                'Capex Leverage Multiple (x)': leverage_multiple,
                'kW per $ Public Grant': total_capacity / public_grant * 1000,
                # 'Estimated Jobs Created': total_projects * 2.5  # Example job multiplier
            }
        )

        df = metrics.to_frame(name=f"P{pct}").rename_axis("Metric")
        st.dataframe(df.style.format("{:,.2f}"), use_container_width=False)


class Simulation(BaseModel):
    years: int = Field(default=25, description="Number of years to simulate")
    return_year: int = Field(default=10, description="Years to return private equity")
    iterations: int = Field(default=1000, description="Monte Carlo iterations per run")
    distributions: Dict[str, Distribution] = Field(default_factory=dict)
    samples: Dict[str, NdArray] = Field(default_factory=dict)
    model: FinancialModel = Field(None)
    summary: Dict[str, PandasDataFrame] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            NdArray: lambda arr: arr.tolist(),
            PandasDataFrame: lambda df: df.to_dict(orient="records"),
        }

    def __init__(self, config_path: str = "config.yaml", **data: Any):
        super().__init__(**data)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.years = cfg.get("years", self.years)
        for dist_cfg in cfg.get("distributions", []):
            dist = Distribution(**dist_cfg)
            self.distributions[dist.label] = dist
        self.model = FinancialModel()

    def sample_distributions(self) -> None:
        """Draw Monte Carlo samples for each distribution input."""
        self.samples.clear()
        for label, dist in self.distributions.items():
            data = dist.create_data(self.iterations, self.years)
            self.samples[label] = data

    def monte_carlo_forecast(self) -> Dict[str, pd.DataFrame]:
        """
        Run Monte Carlo forecast using the updated FinancialModel.
        """
        # self.model = FinancialModel(years=self.years, return_year=self.return_year)
        # 1) Sample all inputs
        self.sample_distributions()
        # 2) Run the financial simulation
        self.model.run_simulation(self.samples)
        # 3) Summarize key percentiles
        self.summary = self.model.summarize_percentiles()
        return self.summary

    def sensitivity_analysis(self, target_metric: str, year: int) -> pd.DataFrame:
        """
        Compute sensitivity of `target_metric` at given 1-based `year`
        to each input distribution. Returns a DataFrame with columns:
        ['Parameter','LowValue','HighValue','MetricLow','MetricHigh','Baseline','Range']
        """
        # Ensure baseline forecast exists
        if not self.summary:
            self.monte_carlo_forecast()

        # convert to 0-based index
        idx = year - 1

        # Baseline median for target metric
        # summary[target_metric] is a dict {'p10':array, 'p50':array, 'p90':array}
        baseline_val = self.summary[target_metric]["P50"][idx]

        records = []
        # Iterate each distribution parameter
        for label, dist in self.distributions.items():
            if dist.use_fixed:
                continue
            orig = self.samples[label].copy()

            # Low‐shock: set all samples to p10
            self.samples[label] = np.full(orig.shape, dist.p10)
            self.model.run_simulation(self.samples)
            summary_low = self.model.summarize_percentiles()
            low_val = summary_low[target_metric]["P50"][idx]

            # High‐shock: set all samples to p90
            self.samples[label] = np.full(orig.shape, dist.p90)
            self.model.run_simulation(self.samples)
            summary_high = self.model.summarize_percentiles()
            high_val = summary_high[target_metric]["P50"][idx]

            # restore original samples
            self.samples[label] = orig

            records.append(
                {
                    "Parameter": label,
                    "LowValue": dist.p10,
                    "HighValue": dist.p90,
                    "MetricLow": low_val,
                    "MetricHigh": high_val,
                    "Baseline": baseline_val,
                    "Range": abs(
                        high_val - low_val,
                    ),
                }
            )

        return pd.DataFrame(records)
