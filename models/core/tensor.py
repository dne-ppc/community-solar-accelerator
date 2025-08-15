# models/core/tensor.py
from __future__ import annotations

from typing import Optional, Tuple, Union, Any
import numpy as np

ArrayLike = Union[float, int, np.ndarray]


def _is_unitless(u: str) -> bool:
    return u is None or u == "" or u.lower() == "unitless"


def _resolve_units_addsub(u1: str, u2: str) -> str:
    # Same or one unitless -> keep the other; otherwise, disallow
    if u1 == u2:
        return u1
    if _is_unitless(u1):
        return u2
    if _is_unitless(u2):
        return u1
    raise ValueError(f"Unit mismatch for add/sub: {u1!r} vs {u2!r}")


def _resolve_units_mul(u1: str, u2: str) -> str:
    if _is_unitless(u1):
        return u2
    if _is_unitless(u2):
        return u1
    return f"{u1}*{u2}"


def _resolve_units_div(u1: str, u2: str) -> str:
    if _is_unitless(u2):
        return u1
    if _is_unitless(u1):
        return f"1/{u2}"
    return f"{u1}/{u2}"


def _broadcast_other(
    self_shape: Tuple[int, int], other: Any
) -> Tuple[np.ndarray, Optional[str]]:
    """Coerce `other` (scalar/array/ModelTensor-like) to (I, T) array and (maybe) units.

    Accepts shapes:
      - scalar
      - (1, 1), (1, T), (I, 1), (I, T)
    Raises if not broadcastable.
    """
    I, T = self_shape

    # Fast path: scalar
    if isinstance(other, (int, float)):
        return (np.full((I, T), float(other), dtype=float), None)

    # numpy array
    if isinstance(other, np.ndarray):
        arr = np.asarray(other, dtype=float)
        if arr.ndim == 0:
            return (np.full((I, T), float(arr), dtype=float), None)
        if arr.ndim == 1:
            # Allow broadcasting from (T,) or (I,)
            if arr.shape[0] == T:
                arr = np.tile(arr[None, :], (I, 1))
            elif arr.shape[0] == I:
                arr = np.tile(arr[:, None], (1, T))
            else:
                raise ValueError(
                    f"Cannot broadcast array of shape {arr.shape} to {(I, T)}"
                )
        elif arr.ndim == 2:
            aI, aT = arr.shape
            if aI in (1, I) and aT in (1, T):
                if aI == 1 and I > 1:
                    arr = np.tile(arr, (I, 1))
                if aT == 1 and T > 1:
                    arr = np.tile(arr, (1, T))
            elif arr.shape == (I, T):
                pass
            else:
                raise ValueError(
                    f"Cannot broadcast array of shape {arr.shape} to {(I, T)}"
                )
        else:
            raise ValueError("other array must be 0D, 1D, or 2D")
        return (arr, None)

    # ModelTensor-like with .data, .units, .iterations, .years
    if (
        hasattr(other, "data")
        and hasattr(other, "years")
        and hasattr(other, "iterations")
    ):
        data = np.asarray(other.data, dtype=float)
        oI, oT = int(getattr(other, "iterations")), int(getattr(other, "years"))
        if (oI in (1, I)) and (oT in (1, T)):
            if oI == 1 and I > 1:
                data = np.tile(data, (I, 1))
            if oT == 1 and T > 1:
                data = np.tile(data, (1, T))
        elif data.shape == (I, T):
            pass
        else:
            raise ValueError(
                f"Cannot broadcast tensor of shape {data.shape} "
                f"({oI},{oT}) to {(I, T)}"
            )
        return (data, getattr(other, "units", None))

    raise TypeError(f"Unsupported operand type: {type(other)!r}")


class TensorOps:
    """Mixin that adds vectorized arithmetic and reductions to ModelTensor.

    Usage:
        class ModelTensor(TensorOps, BaseModel):
            ...

    All binary ops attempt to broadcast the RHS to the LHS shape using rules:
        - scalar -> full surface
        - array -> supports (I,), (T,), (1,1), (1,T), (I,1), (I,T)
        - ModelTensor-like -> same broadcasting as arrays, but years/iterations must match or be 1.
    Units:
        - add/sub require equal units or one side unitless; otherwise raises.
        - mul/div compose units (simple string form) and treat unitless as transparent.
    """

    __array_priority__ = 1000  # ensure numpy uses our ops

    # ----- indexing ------------------------------------------------------
    def __getitem__(self, idx):
        """
        NumPy-like indexing. For pure column selection (e.g., [:, j] or [:, j0:j1]),
        return a same-type tensor; otherwise return a NumPy view.
        """

        # Tuple-of-two pattern (rows, cols)
        if isinstance(idx, tuple) and len(idx) == 2:
            rows, cols = idx

            # Treat full-row selection as column slicing → preserve tensor
            full_rows = (
                rows is None
                or rows == slice(None)
                or (
                    isinstance(rows, slice)
                    and rows.start is None
                    and rows.stop is None
                    and rows.step is None
                )
            )
            if full_rows:
                sub = self.data[idx]
                # Ensure 2D shape for single-column selection
                if sub.ndim == 1:
                    sub = sub.reshape(sub.shape[0], 1)
                return self.with_data(np.asarray(sub))

        # Fallback: return ndarray view
        return self.data[idx]

    # ----- assignment ---------------------------------------------------

    def __setitem__(self, idx, value):
        """
        In-place assignment with a small convenience:
        If you're assigning a 1D array of shape (I,) into a single-column 2D slice (I,1),
        this reshapes the RHS to (I,1) to avoid NumPy's shape error.

        Examples:
        self[:, 0]   = v1d            # NumPy ok
        self[:, 1:2] = v1d            # this helper reshapes v1d -> (I,1)
        self[:, 1:2] = v2d_col        # already (I,1), ok
        """
        import numpy as np

        # Peek at the target slice's shape
        target_view = self.data[idx]
        arr = np.asarray(value)

        # If target is (I,1) but value is (I,), reshape to (I,1)
        if (
            isinstance(target_view, np.ndarray)
            and target_view.ndim == 2
            and target_view.shape[1] == 1
            and arr.ndim == 1
            and arr.shape[0] == target_view.shape[0]
        ):
            arr = arr.reshape(-1, 1)

        self.data[idx] = arr

    # Make sure numpy knows how to turn a tensor into an ndarray
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    # ----- unary ops ---------------------------------------------------------
    def __neg__(self):
        return self.with_data(-self.data)

    def __abs__(self):
        return self.with_data(np.abs(self.data))

    # ----- binary helpers ----------------------------------------------------
    def _binop(self, other: Any, *, op, unit_rule) -> "TensorOps":
        I, T = int(self.iterations), int(self.years)
        rhs, rhs_units = _broadcast_other((I, T), other)
        out_units = unit_rule(self.units, rhs_units or "")

        out = op(self.data, rhs)
        return self.with_data(out, units=out_units)

    # ----- add/sub -----------------------------------------------------------
    def __add__(self, other: Any):
        return self._binop(other, op=np.add, unit_rule=_resolve_units_addsub)

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self, other: Any):
        return self._binop(other, op=np.subtract, unit_rule=_resolve_units_addsub)

    def __rsub__(self, other: Any):
        # other - self
        I, T = int(self.iterations), int(self.years)
        lhs, lhs_units = _broadcast_other((I, T), other)
        out_units = _resolve_units_addsub(lhs_units or "", self.units)
        out = np.subtract(lhs, self.data)
        return self.with_data(out, units=out_units)

    # ----- mul/div -----------------------------------------------------------
    def __mul__(self, other: Any):
        return self._binop(other, op=np.multiply, unit_rule=_resolve_units_mul)

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __truediv__(self, other: Any):
        return self._binop(other, op=np.divide, unit_rule=_resolve_units_div)

    def __rtruediv__(self, other: Any):
        # other / self
        I, T = int(self.iterations), int(self.years)
        lhs, lhs_units = _broadcast_other((I, T), other)
        out_units = _resolve_units_div(lhs_units or "", self.units)
        out = np.divide(lhs, self.data)
        return self.with_data(out, units=out_units)

    # ----- elementwise helpers ----------------------------------------------
    def clip(
        self,
        min_value: Optional[ArrayLike] = None,
        max_value: Optional[ArrayLike] = None,
    ):
        data = self.data
        if min_value is not None:
            min_arr, _ = _broadcast_other((self.iterations, self.years), min_value)
            data = np.maximum(data, min_arr)
        if max_value is not None:
            max_arr, _ = _broadcast_other((self.iterations, self.years), max_value)
            data = np.minimum(data, max_arr)
        return self.with_data(data)

    def where(self, mask: ArrayLike, other: ArrayLike):
        mask_arr, _ = _broadcast_other((self.iterations, self.years), mask)
        other_arr, other_units = _broadcast_other((self.iterations, self.years), other)
        out_units = _resolve_units_addsub(self.units, other_units or self.units)
        out = np.where(mask_arr.astype(bool), self.data, other_arr)
        return self.with_data(out, units=out_units)

    # ----- reductions --------------------------------------------------------
    def reduce_iterations(self, func) -> "TensorOps":
        """Apply a reduction over iterations axis -> (1, years)."""
        out = func(self.data, axis=0, keepdims=True)
        return self.with_data(out)

    def reduce_years(self, func) -> "TensorOps":
        """Apply a reduction over years axis -> (iterations, 1)."""
        out = func(self.data, axis=1, keepdims=True)
        return self.with_data(out)

    def mean_over_iterations(self) -> "TensorOps":
        return self.reduce_iterations(np.mean)

    def median_over_iterations(self) -> "TensorOps":
        return self.reduce_iterations(np.median)

    def percentile_over_iterations(self, q: float) -> "TensorOps":
        out = np.percentile(self.data, q, axis=0, keepdims=True, method="linear")
        return self.with_data(out)
