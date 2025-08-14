# models.py
from __future__ import annotations
from typing import Dict, Optional, TypeVar, Any, Optional, Any, Dict, Iterable, Callable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pydantic import Field, computed_field

from metalog import metalog
import yaml
import numpy_financial as npf

from models.core.registry import RoleRegistry, register_role
from models.core.tensor import TensorCore
from models.ui import TensorUI, DistributionUI

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NdArray = TypeVar("numpy.ndarray")
PlotlyFigure = TypeVar("plotly.graph_objs._figure.Figure")

np.float_ = np.float64


@register_role("calc")
class ModelCalculation(TensorCore, TensorUI):
    pass


@register_role("output")
class ModelOutput(TensorCore, TensorUI):
    pass


@register_role("input")
class ModelInput(TensorCore, TensorUI, DistributionUI):
    """
    Stores the basic configuration for a single parameter's Metalog distribution.
    For example: Price of Gasoline ($/L), with min_value, max_value, p10, p50, p90, etc.
    """

    min_value: float | int
    max_value: float | int
    p10: float | int
    p50: float | int
    p90: float | int
    step: float | int = Field(
        default=0.01,
        description="Step size for sliders in Streamlit UI",
    )
    boundedness: str = "b"  # 'b' for two-sided bounding in metalog

    use_fixed: bool = Field(default=False, description="Override to a fixed constant?")
    fixed_value: bool | int | float = Field(
        default=False,
        description="If use_fixed is True, draw every sample from this value",
    )

    def __init__(
        self,
        sensitivity_value: Optional[float] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)

        if "data" not in data:
            self.update_data(
                sensitivity_value=sensitivity_value,
            )

    @computed_field
    @property
    def dist(self) -> dict:
        return metalog.fit(
            x=[self.p10, self.p50, self.p90],
            boundedness=self.boundedness,
            bounds=[self.min_value - self.step, self.max_value + self.step],
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )


    @classmethod
    def from_config(cls, scenario, name, config_path: str = "inputs/Base.yaml", **data):
        with open(config_path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        settings: Dict[str, Any] = cfg.get(name)
        settings["scenario"] = scenario
        if "years" in settings:
            data.pop("years", None)
        settings.update(data)
        return cls(**settings)

    def update_data(
        self,
        sensitivity_value: float | int | None = None,
    ):
        """
        Draws samples for `iterations` * `years`.
        Priority:
          1) If sensitivity override matches label → full constant array of sensitivity_value.
          2) Else if use_fixed=True and fixed_value is set → full constant array of fixed_value.
          3) Otherwise sample from Metalog as before.
        """
        shape = (self.iterations, self.years)

        # 1) Sensitivity‐analysis override
        if sensitivity_value is not None:
            self.data = np.full(shape, sensitivity_value, dtype=float)
            return

        # 2) Fixed‐value override
        if self.use_fixed:
            # Fallback if fixed_value is None: use median
            val = self.fixed_value if self.fixed_value is not None else self.p50
            self.data = np.full(shape, val, dtype=float)
            return

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
        values: np.ndarray = metalog.r(dist, n=self.iterations * (self.years))

        self.data = values.reshape(shape).clip(self.min_value, self.max_value)

