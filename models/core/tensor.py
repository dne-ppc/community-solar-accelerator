# models.py
from __future__ import annotations
from typing import  Optional, TypeVar, Optional

import numpy as np


from pydantic import BaseModel
import numpy_financial as npf

from models.core.registry import RoleRegistry, register_role

NdArray = TypeVar("numpy.ndarray")

np.float_ = np.float64


@register_role("tensor")  # neutral base role; concrete subclasses set their own role
class TensorCore(BaseModel):

    scenario: str
    label: str
    description: Optional[str] = None
    units: Optional[str] = None
    data: NdArray  | None = None
    years: int = 25
    iterations: int = 1000

    def __add__(self, other):
        return self.data + (other.data if hasattr(other, "data") else other)

    def __sub__(self, other):
        return self.data - (other.data if hasattr(other, "data") else other)

    def __mul__(self, other):
        return self.data * (other.data if hasattr(other, "data") else other)

    def __truediv__(self, other):
        return self.data / (other.data if hasattr(other, "data") else other)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape

    def npv(self, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0], 1)), self.data], axis=1)
        return np.array([npf.npv(rates[i], arr[i, :]) for i in range(self.shape[0])])

    def npv_iter(self, i: int, rates: NdArray) -> NdArray:
        arr = np.concatenate([np.zeros((self.shape[0], 1)), self.data], axis=1)

        return np.array([npf.npv(rates[i], arr[i, :])])