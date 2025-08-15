# models/visualization/tables.py
import pandas as pd
import numpy as np
from projects.solar import SolarProject


def npv_summary(model, accuracy: int = 0) -> pd.DataFrame:
    """
    Return a DataFrame summarizing the NPV of each output property.
    The DataFrame has columns for each output name and rows for percentiles (10th, 50th, 90th).
    """
    summary_data = {}
    for name, values in model.npv.items():
        # Convert each output's NPV array to a DataFrame with percentiles as rows
        summary_data[name] = np.percentile(values, [10, 50, 90])

    df = pd.DataFrame(summary_data, index=["P10", "P50", "P90"])
    df.index.name = "Metric"
    return df.T.round(accuracy)


def kpi(model, accuracy: int = 3) -> pd.DataFrame:
    """
    Gather every property on this model whose name ends with "_percentiles",
    call it, and stack the results into a single DataFrame.

    Each individual `<something>_percentiles` is assumed to return a 1×3 DataFrame
    whose index is [P10, P50, P90] (transposed, so after .T its index is the metric name).
    This method simply concatenates them so that each row is one metric and the columns
    are ["P10","P50","P90"].

    Example output:

                        P10     P50     P90
    cash_depletion_year  5.20    8.00   12.40
    private_investor_irr 2.15    7.80   15.30
    project_irr          1.75    5.50   11.10
    """
    dfs: list[pd.DataFrame] = []

    # Loop over all attributes; pick those whose name ends with "_percentiles".
    for attr_name in dir(model):
        if not attr_name.endswith("_percentiles"):
            continue

        # Avoid recursively calling this property itself
        if attr_name == "all_percentiles":
            continue

        candidate = getattr(model, attr_name, None)
        # Only keep it if it returned a DataFrame
        if isinstance(candidate, pd.DataFrame):
            dfs.append(candidate)

    if len(dfs) == 0:
        # If no _percentiles properties exist, return empty DataFrame
        return pd.DataFrame(columns=["P10", "P50", "P90"])

    # Concatenate so that each DataFrame’s index (its single metric) becomes a row in the final table
    result = pd.concat(dfs, axis=0)

    # Optional: sort by index (i.e. metric name)
    result = result.sort_index()

    for col in ["P10", "P50", "P90"]:
        result[col] = result[col].astype(float)

    return result.round(accuracy)


def investment_iteration(project: SolarProject, iteration: int) -> None:

    return pd.DataFrame(
        data=[
            project.system_output[iteration],
            project.private_investment[iteration],
            project.public_investment[iteration],
            project.seed_capital[iteration],
            project.dividend_rate[iteration],
            project.dividend_start_year[iteration],
            project.capital_return_year[iteration],
            project.return_period[iteration],
            project.discount_rate[iteration],
        ],
        index=[
            "System Output",
            "Private Investment",
            "Public Investment",
            "Seed Capital",
            "Dividend Rate",
            "Dividend Start Year",
            "Capital Return Start Year",
            "Return Periods",
            "Discount Rate",
        ],
        columns=[""],
    ).round(2)


def system_iteration(project: SolarProject, iteration: int) -> None:
    return pd.DataFrame(
        data=[
            project.system_output[iteration],
            project.degradation_rate[iteration],
        ],
        index=[
            "System Output",
            "Panel Degradation Rate",
        ],
        columns=[""],
    ).round(2)


def costs_iteration(project: SolarProject, iteration: int) -> None:
    return pd.DataFrame(
        data=[
            project.maintenance_rate[iteration],
            project.admin_rate[iteration],
            project.insurance_rate[iteration],
            project.inflation_rate[iteration],
        ],
        index=[
            "Maintenance Cost",
            "Administrative Cost",
            "Insurance Cost",
            "Inflation Rate",
        ],
        columns=[""],
    ).round(2)


def annual_iteration(project: SolarProject, iteration: int) -> None:
    annual = pd.DataFrame(
        data=[
            project.production[iteration],
            project.maintenance_cost[iteration],
            project.admin_cost[iteration],
            project.insurance_cost[iteration],
            project.opex[iteration],
            project.tax_revenue[iteration],
            project.revenue[iteration],
            project.operating_margin[iteration],
            project.investor_dividends[iteration],
            project.capital_returned[iteration],
            project.finance_costs[iteration],
            project.remaining_private_investment[iteration],
            project.retained_earnings[iteration],
        ],
        index=[
            "Energy Production (kWh)",
            "Maintenance Cost",
            "Admin Cost",
            "Insurance Cost",
            "OPEX",
            "Tax Revenue",
            "Revenue",
            "Operating Margin",
            "Investor Dividends",
            "Capital Returned",
            "Finance Costs",
            "Remaining Private Investment",
            "Retained Earnings",
        ],
        columns=range(2026, 2026 + project.years),
    )

    cols = annual.columns

    annual["Total"] = annual.sum(axis=1)

    annual = annual[["Total"] + list(cols)]

    return annual.round(0)


def npv_iteration(project: SolarProject, iteration: int) -> None:
    revenue = project.revenue.npv_iter(iteration, project.discount_rate / 100)
    opex = project.opex.npv_iter(iteration, project.discount_rate / 100)
    captial = project.private_investment[iteration]
    finance = project.investor_dividends.npv_iter(
        iteration, project.discount_rate / 100
    )
    total = revenue - opex - captial - finance

    return pd.DataFrame(
        data=[revenue, opex, captial, finance, total],
        index=["NPV Revenue", "NPV OPEX", "NPV Capital", "NPV Financing", "NPV Total"],
        columns=[""],
    ).round(0)
