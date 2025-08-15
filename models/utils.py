# models.py
from __future__ import annotations

from typing import Iterable, Callable,Any,Set, Tuple

from dataclasses import dataclass
import ast
import textwrap

import numpy as np

np.float_ = np.float64


class RoleRegistry:
    """
    Maps role -> set(classes). Lets us declare/inspect 'input', 'output',
    'calc', or any arbitrary role like 'x'.
    """

    _by_role: dict[str, set[type]] = {}

    @classmethod
    def register(cls, role: str, typ: type) -> None:
        role = role.lower()
        cls._by_role.setdefault(role, set()).add(typ)

    @classmethod
    def classes_for(cls, role: str) -> set[type]:
        return cls._by_role.get(role.lower(), set())

    @classmethod
    def all_roles(cls) -> Iterable[str]:
        return cls._by_role.keys()


def register_role(role: str) -> Callable[[type], type]:
    """
    Decorator: @register_role("input") on a class sets _role and registers it.
    """
    role = role.lower()

    def _wrap(typ: type) -> type:
        setattr(typ, "_role", role)
        RoleRegistry.register(role, typ)
        return typ

    return _wrap


def depends_on(*names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a manual dependency list to a computed-property function.

    Usage:
        @computed_field
        @property
        @depends_on("capex", "inflation_rate")
        def maintenance_cost(self) -> ModelCalculation:
            ...
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_depends_on", tuple(names))
        return func

    return _wrap


# ------------------ AST-based dependency extraction -------------------------
def _iter_self_attribute_names(py_src: str, valid_names: Set[str]) -> Set[str]:
    """Parse python source and return names accessed as `self.<name>` that are in `valid_names`."""
    try:
        tree = ast.parse(textwrap.dedent(py_src))
    except SyntaxError:
        return set()
    deps: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            attr = node.attr
            if attr in valid_names:
                deps.add(attr)
    return deps


@dataclass(frozen=True)
class GraphBuildOptions:
    include_roles: Tuple[str, ...] = ("input", "calc", "output")
    use_manual_first: bool = True  # prefer @depends_on overrides when available
