# common_model.py
from typing import Dict, List, Iterable, Any
import os
import yaml

import numpy as np
import pandas as pd
from pydantic import computed_field
from models.core.types import (
    RoleRegistry,
    ModelInput,
    ModelCalculation,
    ModelOutput,
    PandasDataFrame,
    PlotlyFigure,
)
import plotly.graph_objects as go


class CommonModelMixin:
    """
    Mix-in capturing:
      • inputs_names, inputs_fields, assumptions
      • output_names
      • update_inputs()
    for any Pydantic model that uses ModelInput/ModelOutput.
    """

    scenario: str
    years: int
    iterations: int
    # --------------- Role matching helpers ---------------------------------
    @staticmethod
    def _ann_matches_role(ann: Any, role_types: tuple[type, ...], type_names: set[str]) -> bool:
        """True if 'ann' matches any registered class for the role."""
        # forward-ref string
        if isinstance(ann, str):
            return ann in type_names
        # direct or subclass match
        if isinstance(ann, type):
            return any(ann is t or issubclass(ann, t) for t in role_types)
        return False

    @classmethod
    def _iter_role_names_cls(cls, role: str) -> Iterable[str]:
        """
        Yield names of fields (declared + computed) whose type/return_type
        matches any class registered for 'role'.
        """
        role_types = tuple(RoleRegistry.classes_for(role))
        if not role_types:
            return  # nothing registered for this role

        type_names = {t.__name__ for t in role_types}

        # 1) Declared (non-computed) fields
        for name, finfo in getattr(cls, "model_fields", {}).items():
            ann = getattr(finfo, "annotation", None)
            if cls._ann_matches_role(ann, role_types, type_names):
                yield name

        # 2) Computed fields
        for name, cinfo in getattr(cls, "model_computed_fields", {}).items():
            ann = getattr(cinfo, "return_type", None)
            if cls._ann_matches_role(ann, role_types, type_names):
                yield name

    # --------------- Public API --------------------------------------------
    def role_names(self, role: str) -> List[str]:
        """Class-driven discovery of names (safe during __init__)."""
        return list(type(self)._iter_role_names_cls(role) or [])

    def role_fields(self, role: str) -> Dict[str, Any]:
        """
        Instance values for a role. Accessing computed fields will evaluate them.
        Filters out None to avoid half-initialized values.
        """
        out: Dict[str, Any] = {}
        for name in self.role_names(role):
            val = getattr(self, name, None)
            if val is not None:
                out[name] = val
        return out

    # --------------- Conveniences for common roles --------------------------
    @computed_field
    @property
    def inputs_names(self) -> List[str]:
        return self.role_names("input")
    

    @computed_field
    @property
    def calc_names(self) -> List[str]:
        return self.role_names("calc")

    @computed_field
    @property
    def output_names(self) -> List[str]:
        return self.role_names("output")

    # --------------- Dynamic accessors for arbitrary roles ------------------
    def __getattr__(self, name: str):
        """
        Support <role>_names / <role>_fields for any registered role.
        Example: self.x_names, self.x_fields (for role 'x').
        """
        if name.endswith("_names"):
            role = name[:-6]
            if RoleRegistry.classes_for(role):
                return self.role_names(role)
        if name.endswith("_fields"):
            role = name[:-7]
            if RoleRegistry.classes_for(role):
                return self.role_fields(role)
        raise AttributeError(name)
    
    # ---------------- Inputs and assumptions ---------------------------------

    @computed_field
    @property
    def assumptions(self) -> PandasDataFrame:
        # assemble p10/p50/p90, fixed flags, etc.
        return pd.DataFrame(
            {
                name: {
                    "p10": inp.p10 if not inp.use_fixed else np.nan,
                    "p50": inp.p50 if not inp.use_fixed else np.nan,
                    "p90": inp.p90 if not inp.use_fixed else np.nan,
                    "is_fixed": inp.use_fixed,
                    "fixed_value": inp.fixed_value if inp.use_fixed else np.nan,
                    "units": inp.units,
                    "description": inp.description,
                }
                for name, inp in self.input_fields.items()
            }
        ).T

    def update_inputs(self, config_path: str) -> None:
        """
        Reload every ModelInput from the given YAML.
        """
        for name in type(self).role_names_cls("input"):

            mi = ModelInput.from_config(
                self.scenario,
                name,
                years=self.years,
                iterations=self.iterations,
                config_path=config_path,
            )
            setattr(self, name, mi)

    def save_model(self) -> None:

        fname = f"{self.scenario}.yaml"
        save_path = os.path.join("inputs", fname)
        data = {}
        exclude = [
            "data",
            "dist",
            "dist_plot",
            "hist_plot",
            "controls",
            "surface_plot",
            "timeseries_plot",
            "boundedness",
            "iterations",
        ]
        for name, input in self.input_fields.items():
            data[name] = input.model_dump(exclude=exclude)
        with open(save_path, "w") as f:
            yaml.safe_dump(data, f)

    @computed_field
    @property
    def npv_summary(self) -> PandasDataFrame:
        """
        Summarize each dollar‐metric NPV at P10/P50/P90 into a DataFrame.
        """
        data = {}
        for name, arr in self.npv.items():
            data[name] = np.percentile(arr, [10, 50, 90])
        df = pd.DataFrame(data, index=["P10", "P50", "P90"]).T
        df.index.name = "Metric"
        return df

    def sensitivity_analysis(
        self, target_metric: str, percentile: int = 50
    ) -> pd.DataFrame:
        """
        For each non‐fixed portfolio input, shock it to P10 and P90 and
        record the change in the percentile‐NPV of `target_metric`.
        Returns a DataFrame with columns:
        ['Parameter','LowValue','HighValue','MetricLow','MetricHigh','Baseline','Range'].
        """
        # baseline
        base_vals = self.npv[target_metric]
        baseline = float(np.percentile(base_vals, percentile))

        records = []
        vars_df = self.assumptions[self.assumptions.is_fixed == False]

        for param, row in vars_df.iterrows():
            inp = getattr(self, param)
            orig = inp.data.copy()

            low, high = row["p10"], row["p90"]

            # shock low
            inp.update_data(sensitivity_value=low)
            low_val = float(np.percentile(self.npv[target_metric], percentile))

            # shock high
            inp.update_data(sensitivity_value=high)
            high_val = float(np.percentile(self.npv[target_metric], percentile))

            # restore
            inp.data = orig

            records.append(
                {
                    "Parameter": param,
                    "LowValue": low,
                    "HighValue": high,
                    "MetricLow": low_val,
                    "MetricHigh": high_val,
                    "Baseline": baseline,
                    "Range": abs(high_val - low_val),
                }
            )

        return (
            pd.DataFrame(records)
            .sort_values("Range", ascending=False)
            .reset_index(drop=True)
        )

    def sensitivity_plot(
        self, target_metric: str, percentile: int = 50
    ) -> PlotlyFigure:

        fig = go.Figure()

        df = self.sensitivity_analysis(target_metric, percentile)

        # Add traces
        for idx, (_, r) in enumerate(df.iterrows()):
            param = r["Parameter"]
            m_low = r["MetricLow"]
            base = r["Baseline"]
            m_high = r["MetricHigh"]

            # Low to base (red)
            fig.add_trace(
                go.Scatter(
                    x=[m_low, base],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="orange", width=10),
                    showlegend=True if idx == 0 else False,
                    name=f"P10",
                ),
            )
            # Base to high (green)
            fig.add_trace(
                go.Scatter(
                    x=[base, m_high],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="blue", width=10),
                    showlegend=True if idx == 0 else False,
                    name=f"P90",
                ),
            )
            # Annotate parameter bounds
            fig.add_annotation(
                x=m_low,
                y=param,
                text=f"{r['LowValue']}\t",
                xanchor="right",
                yanchor="middle",
                showarrow=False,
            )
            fig.add_annotation(
                x=m_high,
                y=param,
                text=f"\t{r['HighValue']}",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
            )
            # Baseline marker
            fig.add_trace(
                go.Scatter(
                    x=[base],
                    y=[param],
                    mode="markers",
                    marker=dict(color="black", symbol="circle", size=8),
                    showlegend=False,
                ),
            )

            fig.add_annotation(
                x=base,
                y=param,
                text=f"\t{r['Baseline']}",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
            )

        # Layout tweaks
        fig.update_layout(
            height=400 + 20 * len(df),
            width=800,
            # xaxis_title=target_metric,
            # yaxis=dict(title="Parameter", automargin=True),
            title_x=0.5,
            margin=dict(l=200),
            title=f"Sensitivity Analysis: {getattr(self,target_metric).label}",
        )
        return fig
