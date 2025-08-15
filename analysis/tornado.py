from contextlib import contextmanager
import itertools
import numpy as np
import pandas as pd

@contextmanager
def patched_inputs(model, **overrides):
    saved = {k: getattr(model, k).data.copy() for k in overrides}
    try:
        for k, v in overrides.items():
            getattr(model, k).update_data(sensitivity_value=v)
        yield
    finally:
        for k, data in saved.items():
            getattr(model, k).data = data


def sensitivity_analysis(
        model, target_metric: str, percentile: int = 50
    ) -> pd.DataFrame:
        """
        Compute sensitivity of `target_metric` at a given 1-based `year`
        to each input assumption that is not fixed. Returns a DataFrame
        with columns: ['Parameter','LowValue','HighValue','MetricLow','MetricHigh','Baseline','Range'].

        For each input parameter:
          - `LowValue` is its p10,
          - `HighValue` is its p90,
          - `MetricLow` is the 50th-percentile of target_metric (at `year`)
             when that parameter is held constant at p10 (and all other inputs unchanged),
          - `MetricHigh` is the 50th-percentile when that parameter is held constant at p90,
          - `Baseline` is the 50th-percentile of the current (unshocked) model,
          - `Range` = abs(MetricHigh – MetricLow).

        This method creates two new FinancialModel instances for each non-fixed input:
        one with the input`s samples forced to p10, one to p90. It does NOT mutate
        the original (self) model. It then reads off the median value of
        `target_metric` at the given year for each shock.
        """

        # 1) Baseline median for target_metric
        metric_baseline = np.percentile(model.npv[target_metric], percentile)

        records = []

        variables = model.assumptions[model.assumptions.is_fixed == False]
        # 2) Loop over each assumption (rows of the DataFrame)

        for param, row in variables.iterrows():

            low_val = row["p10"]
            high_val = row["p90"]

            with patched_inputs(model, **{param: low_val}):
                metric_low = np.percentile(model.npv[target_metric], percentile)
            with patched_inputs(model, **{param: high_val}):
                metric_high = np.percentile(model.npv[target_metric], percentile)
            # 5) Record everything
            records.append(
                {
                    "Parameter": param,
                    "LowValue": low_val,
                    "HighValue": high_val,
                    "MetricLow": metric_low,
                    "MetricHigh": metric_high,
                    "BaselineValue": row["p50"],
                    "Baseline": metric_baseline,
                    "Range": abs(metric_high - metric_low),
                }
            )

        return (
            pd.DataFrame(records)
            .sort_values("Range", ascending=True)
            .reset_index(drop=True)
        )

def sensitivity_analysis_combo(
        model,
        target_metric: str,
        percentiles: list = [10, 90],
        top_n: int = 10,
        metric_percentile=50,
        n=3,
    ) -> pd.DataFrame:
        """
        For every combination of 3 non-fixed inputs, evaluate all combos of pX for those three.
        Returns a DataFrame with each row = (triplet, setting, metric value at that setting).
        Also returns the top N triplets by max-min range across the grid.
        """
        ots = model.sensitivity_analysis(target_metric, metric_percentile)
        variables = model.assumptions[model.assumptions.index.isin(ots.Parameter)]
        names = list(variables.index)
        if len(names) < 3:
            raise ValueError("Fewer than three unfixed inputs available.")

        base = np.percentile(model.npv["retained_earnings"], 50)

        percentile_names = [f"p{p}" for p in percentiles]
        percentile_combos = list(itertools.product(percentile_names, repeat=n))
        names = list(variables.index)
        input_combos = list(itertools.combinations(names, n))
        combos = list(itertools.product(percentile_combos, input_combos))

        dfs = []

        summaries = []
        for percentiles, inputs in combos:

            saved = {k: getattr(model, k).data.copy() for k in inputs}
            labels = []

            rows = []

            for percentile, input in zip(percentiles, inputs):
                v = variables.loc[input, percentile]
                getattr(model, input).update_data(sensitivity_value=v)

                if percentile == "p10":
                    col = "MetricLow"
                else:
                    col = "MetricHigh"

                label = f"{input}_{percentile}"
                labels.append(label)
                rows.append(
                    {
                        "percentile": percentile,
                        "input": input,
                        "individual_value": ots.loc[
                            ots.Parameter == input, col
                        ].squeeze()
                        - base,
                    }
                )

            df = pd.DataFrame(rows)
            combo = ",".join(labels)
            df["combo"] = combo
            df["total_value"] = np.percentile(model.npv["retained_earnings"], 50) - base
            dfs.append(df)
            summaries.append(
                {
                    "combo": combo,
                    "total_value": df.loc[0, "total_value"].squeeze(),
                }
            )
            for k, v in saved.items():
                getattr(model, k).data = v
        df = pd.concat(dfs, axis=0)

        df.sort_values(
            [
                "total_value",
                "combo",
            ],
            ascending=False,
            inplace=True,
        )
        mask = (df.combo != df.combo.shift(periods=1)).cumsum() < top_n
        df = df[mask].reset_index(drop=True)
        summary_df = pd.DataFrame(summaries)
        summary_df.sort_values("total_value", ascending=False, inplace=True)
        summary_df = summary_df.head(top_n)

        return df, summary_df